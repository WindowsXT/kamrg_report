import os
import json
import pandas as pd
import torch
import numpy as np
import torch.nn as nn
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from tuils.tools import computeAUROC
from tqdm import tqdm

# ----------------------------------------------------------------------------
# 1. Configuration & Equipment
# ----------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------------------------------------------------------
# 2. Load annotation.json and extract the list of test ids
# ----------------------------------------------------------------------------
with open("data/iu_xray/annotation.json", "r", encoding="utf-8") as f:
    ann = json.load(f)
test_ids = [entry["id"] for entry in ann["test"]]

# ----------------------------------------------------------------------------
# 3. Load the ground-truth CSV
# ----------------------------------------------------------------------------
csv_path   = "data/iu_xray/iu_id_labeled_front.csv"
labels_df  = pd.read_csv(csv_path, dtype={"Report ID": str})
id_col     = "Report ID"
label_cols = [c for c in labels_df.columns if c != id_col]
num_classes= len(label_cols)

# ----------------------------------------------------------------------------
# 4. Load the RAD-DINO backbone & preprocessor
# ----------------------------------------------------------------------------
repo      = "microsoft/rad-dino"
processor = AutoImageProcessor.from_pretrained(repo)
backbone  = AutoModel.from_pretrained(repo).to(device)

# ----------------------------------------------------------------------------
# 5. 定义分类头
# ----------------------------------------------------------------------------
class RadDinoClassifier(nn.Module):
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone  = backbone
        feature_dim    = backbone.config.hidden_size
        self.classifier= nn.Linear(feature_dim, num_classes)
        self.sigmoid   = nn.Sigmoid()

    def forward(self, pixel_values):
        outputs = self.backbone(pixel_values=pixel_values)
        feat    = outputs.last_hidden_state[:, 0]
        logits  = self.classifier(feat)
        probs   = self.sigmoid(logits)
        return probs

# ----------------------------------------------------------------------------
# 6. Instantiate the model & load the optimal weights + thresholds
# ----------------------------------------------------------------------------
model     = RadDinoClassifier(backbone, num_classes=num_classes)
model     = nn.DataParallel(model).to(device)

ckpt_path = os.path.join(
    "data/iu_xray/snapshot_path",
    "rad-dino_256_14_local_val",
    "model_max_auc_4.pth.tar"
)
ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
state_dict = ckpt['state_dict']
new_state_dict = {}
for key, value in state_dict.items():
    new_key = key if key.startswith('module.') else f'module.{key}'
    new_state_dict[new_key] = value
model.load_state_dict(new_state_dict, strict=False)
model.eval()
thresholds = np.array(ckpt['val_threshold'])  # shape [num_classes]

# ----------------------------------------------------------------------------
# 7. Build the prefix -> file name list mapping
# ----------------------------------------------------------------------------
image_dir = "data/iu_xray/iu_front_images"
image_map = {}
for fname in os.listdir(image_dir):
    if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue
    prefix = os.path.splitext(fname)[0].split('_', 1)[0]
    image_map.setdefault(prefix, []).append(fname)

# ----------------------------------------------------------------------------
# 8. Infer, collect, statistically analyze missing IDs
# ----------------------------------------------------------------------------
all_preds       = []
all_binaries    = []
all_gts         = []
pred_image_ids  = []
pred_filenames  = []
tested_ids_set  = set()
skipped_ids_set = set()

for image_id in tqdm(test_ids, desc="Processing test IDs"):
    test_prefix = image_id.split('_',1)[0]
    fnames = image_map.get(test_prefix)
    if not fnames:
        skipped_ids_set.add(image_id)
        continue
    tested_ids_set.add(image_id)

    for fname in tqdm(fnames, desc=f"Images for {image_id}", leave=False):
        img      = Image.open(os.path.join(image_dir, fname)).convert("RGB")
        inputs   = processor(images=img, return_tensors="pt")
        pixel_v  = inputs.pixel_values.to(device)

        with torch.no_grad():
            probs = model(pixel_v)
        probs_np   = probs.squeeze(0).cpu().numpy()

        binary_np = (probs_np >= thresholds).astype(int)

        all_preds.append(probs_np)
        all_binaries.append(binary_np)
        pred_image_ids.append(image_id)
        pred_filenames.append(fname)

        row       = labels_df[labels_df[id_col] == image_id]
        gt_series = row[label_cols].iloc[0].fillna(0).replace(-1,1)
        all_gts.append(gt_series.values.astype(float))

# ----------------------------------------------------------------------------
# 9. Print statistical information
# ----------------------------------------------------------------------------
print(f">>> Unique tested IDs       : {len(tested_ids_set)}")
print(f">>> Missing test IDs        : {len(skipped_ids_set)}")
if skipped_ids_set:
    print(">>> Missing test ID list   :", ", ".join(sorted(skipped_ids_set)))
print(f">>> Total image files tested: {len(all_preds)}")
print("---------------------------------------------------")

# ----------------------------------------------------------------------------
# 10. Save only the binarization results to CSV (drop 'filename', no '_bin' suffix)
# ----------------------------------------------------------------------------
id_df = pd.DataFrame({'Report ID': pred_image_ids})

bin_df = pd.DataFrame(all_binaries, columns=label_cols)

out_df = pd.concat([id_df, bin_df], axis=1)
out_df.to_csv("data/iu_xray/iu_14_prediction_label.csv", index=False)
print(">>> Saved binary labels to iu_14_prediction_label.csv")

# ----------------------------------------------------------------------------
# 11. Calculate and print the AUC
# ----------------------------------------------------------------------------
preds_tensor = torch.tensor(all_preds)
gts_tensor   = torch.tensor(all_gts)
aucs         = computeAUROC(gts_tensor.to(device), preds_tensor.to(device), num_classes)
mean_auc     = np.mean(aucs)

print(">>> Per-class AUC:")
for name, auc in zip(label_cols, aucs):
    print(f"    {name}: {auc:.4f}")
print(f">>> Mean AUC: {mean_auc:.4f}")
