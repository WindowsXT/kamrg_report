import os
import json
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import PIL.Image
import albumentations as A
from transformers import AutoModel

# =========================================
# 0. Global Config
# =========================================
CSV_PATH           = "data/iu_id_labeled_front.csv"           
INPUT_DIR          = "data/iu_xray/iu_front_images"                
OUTPUT_DIR         = "data/iu_xray/heatmaps/agcam_kl"  
MATCHES_JSON       = "data/iu_xray/iu_matches_top30.json"            
P_OUT_DIR          = os.path.join(OUTPUT_DIR, "P")      
Q_OUT_DIR          = os.path.join(OUTPUT_DIR, "Q")      
CKPT_PATH          = os.path.join("data/iu_xray/snapshot_path", "rad-dino_256_14_local_val", "fold4_maxauc.pth.tar")
REPO               = "microsoft/rad-dino"
INCLUDE_UNCERTAIN  = False   
os.makedirs(P_OUT_DIR, exist_ok=True)
os.makedirs(Q_OUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================
# 1. Rad-DINO classification model encapsulation
# =========================================
class RadDinoClassifier(nn.Module):
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone   = backbone
        self.classifier = nn.Linear(backbone.config.hidden_size, num_classes)

    def forward(self, pixel_values, return_attentions=False):
        outputs = self.backbone(pixel_values=pixel_values,
                                output_attentions=return_attentions)
        cls_feat = outputs.last_hidden_state[:, 0]
        logits   = self.classifier(cls_feat)
        if return_attentions:
            return logits, outputs.attentions
        return logits, None

# =========================================
# 2. AGCAM (Paper-faithful Multi-layer + Multi-head Aggregation)
# =========================================
class AGCAM:
    def __init__(self,
                 model,
                 head_fusion='sum',
                 layer_fusion='sum',
                 apply_sigmoid=True,
                 target_layers=None):
        self.model        = model
        self.head_fusion  = head_fusion
        self.layer_fusion = layer_fusion
        self.apply_sigmoid = apply_sigmoid
        self.target_layers = target_layers
        self.attn_list    = []
        # Capture all attention.attention modules
        for name, module in self.model.named_modules():
            if name.endswith("attention.attention"):
                module.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module, inputs, output):
        attn = None
        if isinstance(output, tuple):
            for o in output:
                if isinstance(o, torch.Tensor) and o.dim() == 4:
                    attn = o; break
        elif isinstance(output, torch.Tensor) and output.dim() == 4:
            attn = output
        if attn is None:
            return
        attn = attn.requires_grad_()
        attn.retain_grad()
        self.attn_list.append(attn)

    def _select_layer_indices(self, total_layers):
        if self.target_layers is None:
            return list(range(total_layers))
        return [i for i in self.target_layers if 0 <= i < total_layers]

    def generate(self, pixel_values, cls_idx):
        self.attn_list.clear()
        logits, _ = self.model(pixel_values, return_attentions=True)
        B = logits.shape[0]
        pred_labels = logits.argmax(dim=1)
        # Target Score
        target_indices = torch.full((B,), cls_idx, device=logits.device, dtype=torch.long)
        scores = logits[torch.arange(B, device=logits.device), target_indices]
        self.model.zero_grad(set_to_none=True)
        scores.sum().backward()
        if not self.attn_list:
            raise RuntimeError("No attention captured.")
        total_layers = len(self.attn_list)
        layer_ids    = self._select_layer_indices(total_layers)
        attn_stack = torch.stack([self.attn_list[i] for i in layer_ids], dim=0)
        grad_stack = torch.stack([self.attn_list[i].grad for i in layer_ids], dim=0)
        grad_pos   = F.relu(grad_stack)
        if self.apply_sigmoid:
            attn_proc = torch.sigmoid(attn_stack)
        else:
            attn_proc = attn_stack
        weighted = grad_pos * attn_proc
        # Extract CLS->patch
        cls_rows = weighted[..., 0, 1:]
        L, B2, H, P = cls_rows.shape
        side = int(P ** 0.5)
        if side*side != P:
            raise ValueError(f"Patches {P} not square.")
        # Header Aggregation
        if self.head_fusion=='sum':
            fused_heads = cls_rows.sum(dim=2)
        else:
            fused_heads = cls_rows.mean(dim=2)
        # Layer Aggregation
        if self.layer_fusion=='sum':
            fused = fused_heads.sum(dim=0)
        else:
            fused = fused_heads.mean(dim=0)
        # Normalization
        cams = []
        for b in range(B):
            v = fused[b]
            vmin, vmax = v.min(), v.max()
            if vmax-vmin<1e-8:
                v_norm = torch.zeros_like(v)
            else:
                v_norm = (v-vmin)/(vmax-vmin+1e-6)
            cams.append(v_norm.view(1, side, side))
        cam = torch.stack(cams, dim=0)
        return pred_labels, cam

# =========================================
# 3. Load the tag CSV and match the JSON
# =========================================

labels_df = pd.read_csv(CSV_PATH, dtype={ 'Report ID': str })
labels_df['Report ID'] = labels_df['Report ID'].str.strip()
labels_df = labels_df.set_index('Report ID')
label_cols = labels_df.columns.tolist()

with open(MATCHES_JSON, 'r', encoding='utf-8') as f:
    matches = json.load(f)['exact_matches']

# =========================================
# 4. Load model
# =========================================
backbone = AutoModel.from_pretrained(REPO, attn_implementation="eager").to(device)
model    = RadDinoClassifier(backbone, num_classes=len(label_cols))
model    = nn.DataParallel(model).to(device)
ckpt     = torch.load(CKPT_PATH, map_location=device)
state_dict = ckpt.get('state_dict', ckpt)

new_sd = {}
for k,v in state_dict.items():
    new_sd['module.'+k if not k.startswith('module.') else k] = v
model.load_state_dict(new_sd, strict=False)
model.eval()

# =========================================
# 5. Initialize AGCAM
# =========================================
agcam = AGCAM(
    model=model.module,
    head_fusion='sum',
    layer_fusion='sum',
    apply_sigmoid=True,
    target_layers=None
)

# =========================================
# 6. Pre-treatment
# =========================================
val_transform = A.Compose([
    A.Resize(224,224, interpolation=cv2.INTER_LINEAR),
    A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225), max_pixel_value=255.0, p=1.0),
])

# =========================================
# 7. Generate and save the P&Q heat map
# =========================================
def generate_P_Q_heatmaps():
    for test_id, train_list in matches.items():

        test_files = [f for f in os.listdir(INPUT_DIR) if f.startswith(test_id)]
        if not test_files or test_id not in labels_df.index:
            continue

        row_t = labels_df.loc[test_id]
        if INCLUDE_UNCERTAIN:
            test_labels = [lbl for lbl in label_cols if row_t[lbl] in (1, -1)]
        else:
            test_labels = [lbl for lbl in label_cols if row_t[lbl]==1]
        if not test_labels:
            continue

        for fname in test_files:
            raw = PIL.Image.open(os.path.join(INPUT_DIR, fname)).convert('RGB')
            aug = val_transform(image=np.array(raw))
            pixel = torch.from_numpy(aug['image'].transpose(2,0,1)).unsqueeze(0).to(device)
            for lbl in test_labels:
                idx = label_cols.index(lbl)
                _, cam = agcam.generate(pixel, cls_idx=idx)
                cam_up = F.interpolate(cam, (224,224), mode='bilinear', align_corners=False)[0,0]
                P = cam_up.detach().cpu().numpy()
                P = (P - P.min())/(P.max()-P.min()+1e-6)
                np.save(os.path.join(P_OUT_DIR, f"{test_id}_{lbl.replace(' ','_')}.npy"), P)

        for tr_id in train_list:
            if tr_id not in labels_df.index:
                continue
            row_tr = labels_df.loc[tr_id]
            common = [lbl for lbl in test_labels if row_tr[lbl]==1]
            if not common:
                continue
            tr_files = [f for f in os.listdir(INPUT_DIR) if f.startswith(tr_id)]
            if not tr_files:
                continue
            for fname in tr_files:
                raw = PIL.Image.open(os.path.join(INPUT_DIR, fname)).convert('RGB')
                aug = val_transform(image=np.array(raw))
                pixel = torch.from_numpy(aug['image'].transpose(2,0,1)).unsqueeze(0).to(device)
                for lbl in common:
                    idx = label_cols.index(lbl)
                    _, cam = agcam.generate(pixel, cls_idx=idx)
                    cam_up = F.interpolate(cam, (224,224), mode='bilinear', align_corners=False)[0,0]
                    Q = cam_up.detach().cpu().numpy()
                    Q = (Q - Q.min())/(Q.max()-Q.min()+1e-6)
                    np.save(os.path.join(Q_OUT_DIR, f"{tr_id}_{lbl.replace(' ','_')}.npy"), Q)
    print("Finished generating P and Q heatmaps.")

if __name__ == "__main__":
    generate_P_Q_heatmaps()
