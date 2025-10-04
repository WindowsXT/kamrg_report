import os
import time
import json
import csv
import warnings

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from transformers import AutoModel  
from dataset.dataset import generate_dataset_loader_chexpert, train_transform, val_transform
from tuils.tools import search_f1, computeAUROC
from tuils.loss_function import BinaryEntropyLoss_weight_pos_2  

warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# ----------------------------------------------------------------------------
# 1. Load configuration & global variables
# ----------------------------------------------------------------------------
with open('data/iu_xray/path_configs_chexpert.json', 'r', encoding='utf-8') as f:
    path_data = json.load(f)
csv_path      = path_data['train_label_path']
kfold_path    = path_data['k_fold_path']
snapshot_root = path_data['snapshot_path']

train_batch_size = 24
val_batch_size   = 12 
workers          = 12
device           = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------------------------------------------------------
# 2. Load the RAD-DINO backbone
# ----------------------------------------------------------------------------
repo = "microsoft/rad-dino"
rad_dino = AutoModel.from_pretrained(repo).to(device)

# ----------------------------------------------------------------------------
# 3. Category header: RAD-DINO features → Fully connected → Sigmoid
# ----------------------------------------------------------------------------
class RadDinoClassifier(nn.Module):
    def __init__(self, backbone, num_classes=14, device='cuda'):
        super().__init__()
        self.backbone = backbone
        self.backbone.eval()
        with torch.no_grad():
            dummy = torch.zeros(1,3,224,224, device=device)
            feats = self.backbone(dummy).last_hidden_state[:,0]
            feature_dim = feats.shape[-1]
        self.backbone.train()
        self.classifier = nn.Linear(feature_dim, num_classes)
        self.sigmoid    = nn.Sigmoid()

    def forward(self, imgs):
        outputs = self.backbone(imgs)
        cls_feats = outputs.last_hidden_state[:,0]
        logits = self.classifier(cls_feats)
        probs  = self.sigmoid(logits)
        return probs, cls_feats

# ----------------------------------------------------------------------------
# 4. Verification function
# ----------------------------------------------------------------------------
def epochVal(model, dataLoader, loss_fn, label_names):
    model.eval()
    loss_sum = 0.0
    batches  = 0

    outGT   = torch.FloatTensor().to(device)
    outPRED = torch.FloatTensor().to(device)

    with torch.no_grad():
        for imgs, targets in dataLoader:
            imgs    = imgs.to(device, non_blocking=True)
            targets = targets.view(-1, len(label_names)).to(device, non_blocking=True)

            probs, _ = model(imgs)
            loss     = loss_fn(probs, targets)
            loss_sum += loss.item()
            batches  += 1

            outGT   = torch.cat((outGT,   targets), 0)
            outPRED = torch.cat((outPRED, probs),   0)

    avg_loss = loss_sum / batches
    aucs     = computeAUROC(outGT, outPRED, len(label_names))

    print(">>> Per-class AUC:")
    for name, auc in zip(label_names, aucs):
        print(f"    {name}: {auc:.4f}")
    mean_auc = np.mean(aucs)
    print(f">>> Mean AUC across all classes: {mean_auc:.4f}")

    # Calculate the optimal threshold (per-class)
    max_thresh, max_f1, precs, recs = search_f1(outPRED, outGT)
    return avg_loss, aucs, max_thresh, max_f1, precs, recs

# ----------------------------------------------------------------------------
# 5. The training function has added the function of saving the optimal threshold
# ----------------------------------------------------------------------------
def train_one_model(model_name):
    snapshot_path = os.path.join(snapshot_root, f"{model_name}_256_14_local_val")
    os.makedirs(snapshot_path, exist_ok=True)

    df_all      = pd.read_csv(csv_path)
    label_cols  = df_all.columns[1:].tolist()
    df_labels   = df_all[label_cols]
    num_classes = len(label_cols)
    max_epoch   = 42

    for num_fold in range(5):
        print(f"\n=== Fold {num_fold} ===")

        # 读取样本列表
        with open(os.path.join(kfold_path, f'fold{num_fold}', 'train.txt')) as f:
            c_train = [l.strip() for l in f]
        with open(os.path.join(kfold_path, f'fold{num_fold}', 'val.txt')) as f:
            c_val = [l.strip() for l in f]

        train_loader, val_loader = generate_dataset_loader_chexpert(
            df_all, c_train, train_transform, train_batch_size,
            c_val,   val_transform, val_batch_size,   workers
        )

        model = RadDinoClassifier(
            backbone=rad_dino,
            num_classes=num_classes,
            device=device
        ).to(device)

        optimizer = optim.Adamax(
            model.parameters(), lr=1e-4,
            betas=(0.9,0.999), eps=1e-8, weight_decay=1e-5
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6, verbose=True
        )

        loss_fn = BinaryEntropyLoss_weight_pos_2(df_labels, device=device)

        best_loss     = float('inf')
        best_f1_mean  = 0.0
        best_auc_mean = 0.0

        for epochID in range(max_epoch):
            t0 = time.time()
            model.train()
            train_loss_sum = 0.0
            train_batches  = 0

            for imgs, targets in train_loader:
                imgs    = imgs.to(device, non_blocking=True)
                targets = targets.view(-1, num_classes).to(device, non_blocking=True)

                logits, _ = model(imgs)
                loss_val  = loss_fn(logits, targets)

                optimizer.zero_grad()
                loss_val.backward()
                optimizer.step()

                train_loss_sum += loss_val.item()
                train_batches  += 1

            avg_train_loss = train_loss_sum / train_batches

            # Verify and obtain the threshold
            if epochID == 0 or epochID > 38 or (epochID+1) % 10 == 0:
                valLoss, valAUC, valThresh, valF1, precs, recs = epochVal(
                    model, val_loader, loss_fn, label_cols
                )
                # Save the current optimal threshold of fold to json
                thresh_file = os.path.join(snapshot_path, f"best_thresholds_fold{num_fold}.json")
                with open(thresh_file, 'w') as tf:
                    json.dump(valThresh, tf)

                f1_mean  = np.mean(valF1)
                auc_mean = np.mean(valAUC)
                scheduler.step(valLoss)

            # Build checkpoint
            ckpt = {
                'epoch':         epochID+1,
                'state_dict':    model.state_dict(),  
                'optimizer':     optimizer.state_dict(),
                'val_threshold': valThresh,
                'val_f1_mean':   f1_mean,
                'val_auc_mean':  auc_mean
            }
            # Save the model weights
            if valLoss < best_loss:
                best_loss = valLoss
                torch.save(ckpt, os.path.join(snapshot_path, f"model_min_loss_{num_fold}.pth.tar"))
            if f1_mean > best_f1_mean:
                best_f1_mean = f1_mean
                torch.save(ckpt, os.path.join(snapshot_path, f"model_max_f1_{num_fold}.pth.tar"))
            if auc_mean > best_auc_mean:
                best_auc_mean = auc_mean
                torch.save(ckpt, os.path.join(snapshot_path, f"model_max_auc_{num_fold}.pth.tar"))
            if (epochID+1) % 10 == 0:
                torch.save(ckpt, os.path.join(snapshot_path, f"model_epoch_{epochID+1}_{num_fold}.pth.tar"))

            print([epochID, round(optimizer.param_groups[0]['lr'],5),
                   round(avg_train_loss,4), round(valLoss,4), time.time()-t0,
                   round(f1_mean,3), round(auc_mean,4)])

        del model
        torch.cuda.empty_cache()


if __name__ == '__main__':
    train_one_model('rad-dino')