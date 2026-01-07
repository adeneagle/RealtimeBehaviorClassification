import numpy as np
import torch
from torch import nn
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from omegaconf import OmegaConf
from torchvision import models
from timm.models.layers import DropPath
import albumentations as A
from lightning.pytorch.loggers.wandb import WandbLogger
from torch.utils.data import Dataset, DataLoader
import os
import cv2

# Dataset class
class MultiImageDataset(Dataset):
    def __init__(self, data_list, transform=None):
        """
        Args:
            data_list: List of tuples (img1_path, img2_path, img3_path, label)
            transform: Albumentations transform
        """
        self.data_list = data_list
        self.transform = transform
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        img1_path, img2_path, img3_path, label = self.data_list[idx]
        
        # Load images with OpenCV and convert BGR to RGB
        img1 = cv2.cvtColor(cv2.imread(img1_path), cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(cv2.imread(img2_path), cv2.COLOR_BGR2RGB)
        img3 = cv2.cvtColor(cv2.imread(img3_path), cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(
                image=img1,
                image1=img2,
                image2=img3
            )
            img1 = transformed['image']
            img2 = transformed['image1']
            img3 = transformed['image2']
        
        triplet = torch.stack([img1, img2, img3])

        return triplet, label

# DataModule
class MultiImageDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_data,
        val_data,
        test_data=None,
        batch_size=32,
        num_workers=4,
        image_size=224,
    ):
        super().__init__()
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        
        # Training transforms with augmentation
        self.train_transform = A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.Rotate(limit=15, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            A.pytorch.ToTensorV2(),
        ], additional_targets={'image1': 'image', 'image2': 'image'})
        
        # Validation/test transforms without augmentation
        self.val_transform = A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            A.pytorch.ToTensorV2(),
        ], additional_targets={'image1': 'image', 'image2': 'image'})
    
    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = MultiImageDataset(self.train_data, self.train_transform)
            self.val_dataset = MultiImageDataset(self.val_data, self.val_transform)
        
        if stage == 'test' or stage is None:
            if self.test_data is not None:
                self.test_dataset = MultiImageDataset(self.test_data, self.val_transform)
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )
    
    def test_dataloader(self):
        if self.test_data is not None:
            return DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
            )

def train_test_split(file_paths, labels, prefixes, train_percent=0.7, val_percent=0.15, test_percent=0.15):
    """
    Split data from multiple videos into train/val/test sets.
    
    Args:
        sf: List of file paths
        il: List of labels
        prefixes: List of video prefixes to filter by
        train_percent: Percentage of data for training (0-1)
        val_percent: Percentage of data for validation (0-1)
        test_percent: Percentage of data for testing (0-1)
    
    Returns:
        xt, xv, xtest, yt, yv, ytest: Train, val, test splits for data and labels
    """
    assert abs(train_percent + val_percent + test_percent - 1.0) < 1e-6, \
        "train_percent + val_percent + test_percent must equal 1.0"
    
    x_train, x_valid, x_test = [], [], []
    y_train, y_valid, y_test = [], [], []
    
    for p in prefixes:
        combined = list(zip(file_paths, labels))
        fd = [i for i in combined if os.path.basename(i[0]).startswith(p)]
        if not fd:
            continue
        
        # Sort by frame number (assuming format: prefix_framenum_...)
        sd = sorted(fd, key=lambda x: int(os.path.basename(x[0]).split('_')[1]))
        sfo, sla = zip(*sd)
        
        total = len(sfo)
        
        # Calculate split indices
        train_end = int(total * train_percent)
        val_end = int(total * (train_percent + val_percent))
        # test goes from val_end to end
        
        # Split the data
        train_files = list(sfo[:train_end])
        train_labels = list(sla[:train_end])
        
        val_files = list(sfo[train_end:val_end])
        val_labels = list(sla[train_end:val_end])
        
        test_files = list(sfo[val_end:])
        test_labels = list(sla[val_end:])
        
        # Add to accumulator lists
        x_train.extend(train_files)
        y_train.extend(train_labels)
        x_valid.extend(val_files)
        y_valid.extend(val_labels)
        x_test.extend(test_files)
        y_test.extend(test_labels)
        
        print(f"Prefix {p}: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")
    
    print(f"\nTotal: {len(x_train)} train, {len(x_valid)} val, {len(x_test)} test")
    
    return x_train, x_valid, x_test, y_train, y_valid, y_test

## Transformers
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        
    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Transformer(nn.Module):
    def __init__(self, embed_dim=768, depth=4, num_heads=12, mlp_ratio=1., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        super().__init__()
        
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, 
                drop_path = drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
                )
            for i in range(depth)])

    def forward(self, x, pos):
        for _, block in enumerate(self.blocks):
            x = block(x + pos)
        
        return x

class MultiImageTransformerClassifier(L.LightningModule):
    def __init__(self, num_classes, num_images=3, d_model=512, num_heads=8, num_layers=1, dropout=0.1, lr=1e-4, weight_decay=1e-4, max_epochs=100,
                 image_size=300
    ):
        super().__init__()
        self.save_hyperparameters()

        self.efficientnet = models.efficientnet_b3(weights='DEFAULT')
        self.efficientnet.classifier = nn.Identity() # 1536 input features

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_images+1, d_model))
        self.transformer = Transformer(
            embed_dim=d_model,
            depth=num_layers,
            num_heads=num_heads,
            mlp_ratio=1.,
            qkv_bias=True,
        )

        self.decoder = nn.Sequential([nn.Linear(d_model, d_model//2), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model//2, num_classes)])

        self.criterion = nn.CrossEntropyLoss()

        self.preprocess_transforms = A.Compose([
            A.Resize(height=image_size, width=image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            A.pytorch.ToTensorV2()
        ], additional_targets={'image1': 'image', 'image2': 'image'})

        self.train_transforms = A.Compose([
            A.Rotate(limit=(-180, 180), p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.ColorJitter(p=0.5),
        ], additional_targets={'image1': 'image', 'image2': 'image'})

    def forward(self, x):
        batch_size = x.shape[0]

        transformed = self.preprocess_transforms(image=x[0], image1=x[1], image2=x[2])
        transformed_images = torch.stack([transformed['image'], transformed['image1'], transformed['image2']])
        feats = self.efficientnet(transformed_images)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        feats = torch.cat([cls_tokens, feats], dim=1)

        feats = self.transformer(feats, self.pos_embedding)
        feats = feats[:, 0]
        logits = self.decoder(feats)

        return logits
    
    def training_step(self, batch, batch_idx):
        images, labels = batch

        transformed = self.train_transforms(image=images[0], image1=images[1], image2=images[2])
        transformed_images = torch.stack([transformed['image'], transformed['image1'], transformed['image2']])
        logits = self(transformed_images)

        loss = self.criterion(logits, labels)
        self.log('train_loss', loss)

        return loss
    
    def validation_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)

        loss = self.criterion(logits, labels)
        self.log('val_loss', loss)

        return loss
    
    def test_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)

        loss = self.criterion(logits, labels)
        self.log('test_loss', loss)

        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.max_epochs, eta_min=1e-6)
        
        out = {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch'
            }
        }

        return out

def main():
    config = OmegaConf.load('config.yaml')
    wandb_logger = WandbLogger(project=config.project_name, name=config.run_name)

    checkpoint_callback = ModelCheckpoint(monitor='val_loss', mode='min', save_top_k=1)
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=config.training.early_stopping_patience, mode='min')

    trainer = L.Trainer(
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stopping_callback],
        max_epochs=config.training.max_epochs,
        batch_size=config.training.batch_size,
        num_workers=config.data.num_workers,
    )

    (x_train, x_valid, x_test, y_train, y_valid, y_test) = train_test_split(config.data.frames_path, config.data.labels_path, config.data.prefixes, config.data.train_percent, config.data.val_percent, config.data.test_percent)

    data_module = MultiImageDataModule(x_train, x_valid, x_test, config.training.batch_size, config.data.num_workers, config.training.image_size)

    model = MultiImageTransformerClassifier(config.model.num_classes, config.model.d_model, config.model.nhead, config.model.num_layers, config.training.dropout, config.training.lr, config.training.weight_decay, config.training.max_epochs, config.training.image_size)

    trainer.fit(model, data_module)

    trainer.test(model, data_module, ckpt_path='best')

if __name__ == "__main__":
    main()

