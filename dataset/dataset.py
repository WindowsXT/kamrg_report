import os
import json
import cv2
import numpy as np
import torch
import torch.utils.data as data
import albumentations
from PIL import Image
from torch.utils.data import DataLoader

# -----------------------------------------------------------------------------
# 1. 读取配置与预处理管线
# -----------------------------------------------------------------------------
with open('path_configs_chexpert.json', encoding='utf-8') as f:
    path_data = json.load(f)
train_img_path = path_data['train_img_path']

IMAGENET_SIZE = 224
train_transform = albumentations.Compose([
    albumentations.Resize(IMAGENET_SIZE, IMAGENET_SIZE),
    albumentations.OneOf([
        albumentations.RandomGamma(gamma_limit=(60, 120), p=0.9),
        albumentations.HueSaturationValue(
            hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.9
        ),
        albumentations.RandomBrightnessContrast(
            brightness_limit=0.2, contrast_limit=0.2, p=0.9
        ),
    ], p=0.9),
    albumentations.HorizontalFlip(p=0.5),
    albumentations.Affine(
        scale=(0.8, 1.2), translate_percent=(-0.2, 0.2), rotate=(-20, 20),
        interpolation=cv2.INTER_LINEAR, mode=cv2.BORDER_CONSTANT, p=1.0
    ),
    albumentations.OneOf([
        albumentations.GridDistortion(
            num_steps=5, distort_limit=0.3,
            interpolation=cv2.INTER_LINEAR, p=1.0
        ),
        albumentations.ElasticTransform(
            alpha=1, sigma=50, p=1.0
        ),
    ], p=0.5),
    albumentations.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        max_pixel_value=255.0, p=1.0
    ),
])

val_transform = albumentations.Compose([
    albumentations.Resize(IMAGENET_SIZE, IMAGENET_SIZE),
    albumentations.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        max_pixel_value=255.0, p=1.0
    ),
])

# -----------------------------------------------------------------------------
# 2. 数据集类：按前缀匹配标签，并兼容不同 transform 类型
# -----------------------------------------------------------------------------
class Chexnet_dataset_chexpert(data.Dataset):
    def __init__(self, df, name_list, transform=None):
        self.df = df.copy()
        self.name_list = name_list
        self.transform = transform

        # 1) 提取 prefix 列
        self.df['prefix'] = self.df['Report ID'].str.split('_', n=1).str[0]

        # 2) 确定标签列：从 'No Finding' 到 'Support Devices'
        cols = list(self.df.columns)
        start = cols.index('No Finding')
        end   = cols.index('Support Devices')
        self.label_cols = cols[start:end+1]

        # 3) 构建 prefix -> label 向量 映射
        self.label_map = {}
        for _, row in self.df.iterrows():
            p = row['prefix']
            if p not in self.label_map:
                labels = row[self.label_cols].values.astype(np.float32)
                labels = np.nan_to_num(labels)
                labels[labels == -1] = 1.0
                self.label_map[p] = labels

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):
        # 1) 取图像路径与文件名
        full_path = self.name_list[idx]
        fname     = os.path.basename(full_path)

        # 2) 计算 prefix
        prefix = os.path.splitext(fname)[0].split('_', 1)[0]

        # 3) 获取标签向量
        if prefix not in self.label_map:
            raise KeyError(f"Prefix {prefix} not found in CSV")
        labels = self.label_map[prefix]

        # 4) 读取图像
        img_path = os.path.join(train_img_path, fname)
        image_bgr = cv2.imread(img_path)
        if image_bgr is None:
            image_bgr = np.zeros((IMAGENET_SIZE, IMAGENET_SIZE, 3), dtype=np.uint8)

        # 5) 根据 transform 类型进行处理
        if isinstance(self.transform, albumentations.core.composition.Compose):
            # Albumentations 接口
            image = self.transform(image=image_bgr)['image']
            # HWC -> CHW
            image = image.transpose(2, 0, 1).astype(np.float32)
            image_tensor = torch.FloatTensor(image)
        else:
            # TorchVision/CLIP preprocess 接口
            # cv2.imread 得到 BGR，需要转 RGB
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(image_rgb)
            image_tensor = self.transform(pil_img)
            if not isinstance(image_tensor, torch.FloatTensor):
                image_tensor = image_tensor.float()

        return image_tensor, torch.FloatTensor(labels)

# class Chexnet_dataset_chexpert(data.Dataset):
#     def __init__(self, df, name_list, transform=None):
#         self.df = df.copy()
#         self.name_list = name_list
#         self.transform = transform

#         self.df['prefix'] = self.df['Report ID'].str.split('_', n=1).str[0]
#         cols = list(self.df.columns)
#         start = cols.index('No Finding')
#         end = cols.index('Support Devices')
#         self.label_cols = cols[start:end+1]

#         self.label_map = {}
#         for _, row in self.df.iterrows():
#             p = row['prefix']
#             if p not in self.label_map:
#                 labels = row[self.label_cols].values.astype(np.float32)
#                 labels = np.nan_to_num(labels)
#                 labels[labels == -1] = 1.0
#                 self.label_map[p] = labels

#     def __len__(self):
#         return len(self.name_list)

#     def __getitem__(self, idx):
#         full_path = self.name_list[idx]
#         fname = os.path.basename(full_path)
#         prefix = os.path.splitext(fname)[0].split('_', 1)[0]

#         if prefix not in self.label_map:
#             raise KeyError(f"Prefix {prefix} not found in CSV")
#         labels = self.label_map[prefix]

#         img_path = os.path.join(train_img_path, fname)
#         image_bgr = cv2.imread(img_path)
#         if image_bgr is None:
#             image_bgr = np.zeros((IMAGENET_SIZE, IMAGENET_SIZE, 3), dtype=np.uint8)

#         if isinstance(self.transform, albumentations.core.composition.Compose):
#             image = self.transform(image=image_bgr)['image']
#             image = image.transpose(2, 0, 1).astype(np.float32)
#             image_tensor = torch.FloatTensor(image)
#         else:
#             image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
#             pil_img = Image.fromarray(image_rgb)
#             image_tensor = self.transform(pil_img)
#             if not isinstance(image_tensor, torch.FloatTensor):
#                 image_tensor = image_tensor.float()

#         return image_tensor, torch.FloatTensor(labels), prefix


# -----------------------------------------------------------------------------
# 3. 生成 DataLoader 的函数
# -----------------------------------------------------------------------------
def generate_dataset_loader_chexpert(df_all, c_train,  train_transform,
                                     train_batch_size, c_val,   val_transform,
                                     val_batch_size, workers):
    train_dataset = Chexnet_dataset_chexpert(df_all, c_train, train_transform)
    val_dataset   = Chexnet_dataset_chexpert(df_all, c_val,   val_transform)

    train_loader = data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        drop_last=False
    )
    val_loader = data.DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        drop_last=False
    )
    return train_loader, val_loader

