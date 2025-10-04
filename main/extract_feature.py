import os
import json
import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
import albumentations as A

# -----------------------------------------------------------------------------
# 1. Read the configuration and preprocessing pipeline
# -----------------------------------------------------------------------------
with open('data/iu_xray/path_configs_chexpert.json', encoding='utf-8') as f:
    path_data = json.load(f)
image_dir = path_data['train_img_path']

IMAGENET_SIZE = 224

val_transform = A.Compose([
    A.Resize(IMAGENET_SIZE, IMAGENET_SIZE),
    A.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        max_pixel_value=255.0,
        p=1.0
    ),
])

# ------------------ Custom classifier (for loading training weights) ------------------
class RadDinoClassifier(nn.Module):
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone   = backbone
        hidden_size     = backbone.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_classes)
        self.sigmoid    = nn.Sigmoid()

    def forward(self, pixel_values):
        outputs = self.backbone(pixel_values=pixel_values)
        return outputs.last_hidden_state[:, 0]

# ------------------ Image feature extraction ------------------
def extract_image_features(image_paths, transform, model, device):
    model.eval()
    features = {}
    for img_id, path in tqdm(image_paths.items(), desc="Extracting Features"):
        # 1) load & augment
        img = np.array(Image.open(path).convert("RGB"))
        aug = transform(image=img)
        # 2) to torch tensor
        tensor = torch.tensor(aug['image'].transpose(2,0,1), dtype=torch.float32)
        pixel_values = tensor.unsqueeze(0).to(device)
        # 3) forward
        with torch.no_grad():
            feat = model(pixel_values)       # (1, D)
        features[img_id] = feat.cpu().numpy().squeeze()
    return features

# ------------------ Main process ------------------
def main():
    image_paths = {
        os.path.splitext(fn)[0]: os.path.join(image_dir, fn)
        for fn in os.listdir(image_dir)
        if fn.lower().endswith(('.png','.jpg','.jpeg'))
    }

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    repo     = "microsoft/rad-dino"
    from transformers import AutoModel
    backbone = AutoModel.from_pretrained(repo).to(device)

    num_classes = 14
    model = RadDinoClassifier(backbone, num_classes)
    model = nn.DataParallel(model).to(device)

    ckpt_path = os.path.join("snapshot_path", "rad-dino_256_14_local_val", "model_max_auc_4.pth.tar")
    ckpt      = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd        = ckpt['state_dict']
    new_sd    = {('module.'+k if not k.startswith('module.') else k):v for k,v in sd.items()}
    model.load_state_dict(new_sd, strict=False)


    features = extract_image_features(image_paths, val_transform, model, device)

    ids   = np.array(list(features.keys()))
    feats = np.stack(list(features.values()), axis=0)
    np.savez('data/iu_xray/iu_image_features.npz', ids=ids, features=feats)
    print(f"Extracted {len(ids)} features to 'iu_image_features.npz'")

if __name__ == "__main__":
    main()
