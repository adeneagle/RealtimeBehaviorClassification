import numpy as np
import matplotlib.pyplot as plt
from torchmetrics import Accuracy, MatthewsCorrCoef, ConfusionMatrix
import torch
from torch import nn
from torchvision.ops import Conv2dNormActivation
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateFinder
from omegaconf import OmegaConf
from torchvision import models
from timm.layers import DropPath
import albumentations as A
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.tuner import Tuner
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import pandas as pd
import wandb

def load_data(file_name, make_binary=False, binary_behavior=None, root_dir=r"C:\Users\Jimi\Desktop\Brayden\RealtimeLightning\RealtimeBehaviorClassification"):
    vid_path = os.path.join(root_dir, 'processed_frames', file_name)
    data_path = os.path.join(root_dir, 'labels', file_name + '.csv')

    # Get number of frames per group (should be 3)
    first_group = os.listdir(vid_path)[0]
    num_frames_per_group = len(os.listdir(os.path.join(vid_path, first_group)))
    
    # Load labels
    labels = pd.read_csv(data_path)
    class_names = list(labels.columns[1:])

    labels = labels.to_numpy()
    labels_frames = labels[:, 0].astype(int)  # Frame indices
    max_labeled_frame = int(np.max(labels_frames))
    
    # Get folder paths for frames that have labels
    # We need folders starting from index (num_frames_per_group - 1) 
    # because that's the first folder where the last frame (frame i) has full context
    # and going up to max_labeled_frame
    all_folders = sorted(os.listdir(vid_path), key=lambda x: int(x))
    min_folder_idx = num_frames_per_group - 1
    
    # Filter folders where the last frame (folder index i) has a label
    frame_group_paths = [
        os.path.join(vid_path, f) for f in all_folders 
        if min_folder_idx <= int(f) <= max_labeled_frame
    ]

    frame_group_paths_indiv = []

    for path in frame_group_paths:
        paths_full = [os.path.join(path, f"{idx}.png") for idx in [0, 1, 2]]
        frame_group_paths_indiv.append(paths_full)
    
    # Extract integer labels for these frames
    # Get labels for frames [min_folder_idx, ..., max_labeled_frame]
    labels_clipped = labels[min_folder_idx:max_labeled_frame + 1, 1:]
    labels_int = labels_clipped.argmax(axis=1)

    label_ids, counts = np.unique(labels_int, return_counts=True)

    if len(label_ids) != len(class_names):
        missing_idxs = [idx for idx in range(len(class_names)) if idx not in label_ids]

        for missing_idx in missing_idxs:
            print(f"WARNING: removing {class_names[missing_idx]} due to lack of samples")
            class_names.pop(missing_idx)
        
        for idx in range(len(class_names)):
            label_id = label_ids[idx]
            labels_int[labels_int == label_id] = idx


    if make_binary:
        beh_idx = class_names.index(binary_behavior)

        included_beh_idxs = labels_int == beh_idx
        labels_int[:] = 0
        labels_int[included_beh_idxs] = 1

        class_names = ['other', binary_behavior]
    
    labels_int = labels_int.tolist()

    return frame_group_paths_indiv, labels_int, class_names

# Dataset class
class MultiImageDataset(Dataset):
    def __init__(self, data, transform=None):
        self.image_paths, self.labels = data
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img1_path, img2_path, img3_path = self.image_paths[idx]
        label = self.labels[idx]

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
        batch_size=64,
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
            A.Rotate(limit=180, p=.5, border_mode=cv2.BORDER_REPLICATE),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.2),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.2),
            A.GaussNoise(p=0.3),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0
            ),
            A.pytorch.ToTensorV2(),
        ], additional_targets={'image1': 'image', 'image2': 'image'})
        
        # Validation/test transforms without augmentation
        self.val_transform = A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0
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

def train_test_split(file_paths, labels, train_percent=0.7, val_percent=0.15, test_percent=0.15, 
                     make_binary=False, binary_behavior=None):
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

    total = len(labels)
    
    # Calculate split indices
    train_end = int(total * train_percent)
    val_end = int(total * (train_percent + val_percent))
    # test goes from val_end to end
    
    # Split the data
    train_files = list(file_paths[:train_end])
    train_labels = list(labels[:train_end])
    
    val_files = list(file_paths[train_end:val_end])
    val_labels = list(labels[train_end:val_end])
    
    test_files = list(file_paths[val_end:])
    test_labels = list(labels[val_end:])
    
    # Add to accumulator lists
    x_train.extend(train_files)
    y_train.extend(train_labels)
    x_valid.extend(val_files)
    y_valid.extend(val_labels)
    x_test.extend(test_files)
    y_test.extend(test_labels)
    
    print(f"\nTotal: {len(x_train)} train, {len(x_valid)} val, {len(x_test)} test")
    
    return (x_train, y_train), (x_valid, y_valid), (x_test, y_test)

## Transformers
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.1):
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
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0.1, attn_drop=0.1,
                 drop_path=0.1, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
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
    def __init__(self, embed_dim=768, depth=4, num_heads=12, mlp_ratio=2., qkv_bias=False, qk_scale=None,
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
    def __init__(self, num_images=3, d_model=512, num_heads=8, num_layers=1, 
                 attn_drop_rate=0.1, dropout=0.1, drop_path=0.1, lr=1e-4, no_transformer=False,
                 use_dino=True, weight_decay=1e-3, max_epochs=100, weights=None, class_names=None
    ):
        super().__init__()
        self.save_hyperparameters()

        self.num_classes = len(class_names)

        if self.num_classes > 2:
            task = 'multiclass'
        else:
            task = 'binary'

        if use_dino:
            self.feature_extractor = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
            feature_size = self.feature_extractor.embed_dim
        else:
            self.feature_extractor = models.efficientnet_b0(weights='DEFAULT')
            feature_size = self.feature_extractor.classifier[1].in_features
            self.feature_extractor.classifier = nn.Identity()

        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        if not use_dino:
            self.feature_extractor.features[-1] = Conv2dNormActivation(
                in_channels=320,
                out_channels=feature_size,
                kernel_size=1,
                norm_layer=nn.BatchNorm2d,
                activation_layer=nn.ReLU
            )
            for param in self.feature_extractor.features[-1].parameters():
                if not param.requires_grad:
                    raise ValueError("Should be trainable")

        # self.dimension_changer = nn.Linear(feature_size, d_model)

        # self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_images, d_model))
        if no_transformer:
            self.encoder = nn.Sequential(
                nn.Linear(d_model*3, d_model*2),
                nn.BatchNorm1d(d_model*2),
                nn.Dropout(dropout),
                nn.Linear(d_model*2, d_model),
                nn.BatchNorm1d(d_model),
                nn.Dropout(dropout)
            )
        else:
            self.encoder = Transformer(
                embed_dim=d_model,
                depth=num_layers,
                num_heads=num_heads,
                qkv_bias=False,
                drop_rate=dropout,
                drop_path_rate=drop_path,
                attn_drop_rate=attn_drop_rate
            )

        self.decoder = nn.Sequential(
                nn.Linear(d_model, self.num_classes)
                )

        self.criterion = nn.CrossEntropyLoss(weight=weights)

        self.train_acc = Accuracy(task=task, num_classes=self.num_classes, average='macro')
        self.val_acc = Accuracy(task=task, num_classes=self.num_classes, average='macro')
        self.test_acc = Accuracy(task=task, num_classes=self.num_classes, average='macro')
        
        self.train_mcc = MatthewsCorrCoef(task=task, num_classes=self.num_classes)
        self.val_mcc = MatthewsCorrCoef(task=task, num_classes=self.num_classes)
        self.test_mcc = MatthewsCorrCoef(task=task, num_classes=self.num_classes)

        self.test_confusion = ConfusionMatrix(task=task, num_classes=self.num_classes)
        self.class_names = class_names or [f"Class_{i}" for i in range(self.num_classes)]

    def forward(self, x):
        # x shape: (batch_size, num_images, channels, height, width)
        batch_size, num_images, c, h, w = x.shape
        
        # Reshape to process all images at once
        x = x.view(batch_size * num_images, c, h, w)
        
        # Extract features
        feats = self.feature_extractor(x)
        # feats = self.dimension_changer(feats)
        
        # Reshape back to (batch_size, num_images, d_model)
        feats = feats.view(batch_size, num_images, -1)
        
        # Add CLS token
        # cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        # feats = torch.cat([cls_tokens, feats], dim=1)
        
        # Transformer
        if self.hparams.no_transformer:
            feats = feats.view(batch_size, -1)
            feats = self.encoder(feats)
        else:
            feats = self.encoder(feats, self.pos_embedding)
            feats = feats[:, -1]

        logits = self.decoder(feats)
        
        return logits
    
    def training_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)

        loss = self.criterion(logits, labels)
        preds = torch.argmax(logits, dim=1)
        
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_balanced_acc', self.train_acc(preds, labels), on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_mcc', self.train_mcc(preds, labels), on_step=False, on_epoch=True)

        return loss 
    
    def validation_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)

        loss = self.criterion(logits, labels)
        preds = torch.argmax(logits, dim=1)
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_balanced_acc', self.val_acc(preds, labels), on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_mcc', self.val_mcc(preds, labels), on_step=False, on_epoch=True)

        return loss
    
    def test_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)

        loss = self.criterion(logits, labels)
        preds = torch.argmax(logits, dim=1)
        
        self.log('test_loss', loss)
        self.log('test_balanced_acc', self.test_acc(preds, labels), on_step=False, on_epoch=True)
        self.log('test_mcc', self.test_mcc(preds, labels), on_step=False, on_epoch=True)

        self.test_confusion.update(preds, labels)

        return loss

    def on_test_epoch_end(self):
        cm = self.test_confusion.compute().cpu().numpy()

        fig, ax = plt.subplots(figsize=(10, 8), dpi=200)

        cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)

        im = ax.imshow(cm_normalized, vmin=0, vmax=1)

        plt.colorbar(im, ax=ax, label='Fraction of True')

        ax.set_xticks(range(self.num_classes), labels=self.class_names, rotation=45, ha='right')
        ax.set_yticks(range(self.num_classes), labels=self.class_names)
        
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title('Test Confusion Matrix')

        for i in range(self.num_classes):
            for j in range(self.num_classes):
                ax.text(j, i, int(cm[i, j]), ha='center', va='center', color='black')
        
        self.logger.experiment.log({"Confusion Matrix - Test Set": wandb.Image(fig)})
        plt.close(fig)
        
        self.test_confusion.reset()

        test_dataloader = self.trainer.datamodule.test_dataloader()
        self.log_test_video(test_dataloader, fps=30)
    
    def log_test_video(self, dataloader, fps=30, output_size=(640, 480)):
        """
        Create a continuous video from test set predictions.
        Takes the last frame from each triplet to reconstruct the original video.
        
        Args:
            dataloader: Test dataloader (with shuffle=False)
            fps: Frames per second for the output video
            output_size: (width, height) for output frames
        """

        self.eval()
        annotated_frames = []
        
        # Process all batches
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(dataloader):
                images = images.to(self.device)
                logits = self(images)
                preds = torch.argmax(logits, dim=1)
                
                # Process each sample in the batch
                for i in range(len(images)):
                    # Get the last frame (index 2) from the triplet
                    triplet = images[i]  # Shape: (3, C, H, W)
                    last_frame = triplet[2]  # Shape: (C, H, W)
                    
                    pred_label = self.class_names[preds[i]]
                    true_label = self.class_names[labels[i]]
                    
                    # Denormalize
                    frame = last_frame.cpu().permute(1, 2, 0).numpy()
                    mean = np.array([0.485, 0.456, 0.406])
                    std = np.array([0.229, 0.224, 0.225])
                    frame = frame * std + mean
                    frame = np.clip(frame * 255, 0, 255).astype(np.uint8)
                    
                    # Resize frame to consistent size
                    frame = cv2.resize(frame, output_size, interpolation=cv2.INTER_LINEAR)
                    
                    # Get dimensions
                    h, w = frame.shape[:2]
                    
                    # Add semi-transparent background for text
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (0, 0), (w, 60), (0, 0, 0), -1)
                    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
                    
                    # Add text
                    text_true = f"True: {true_label}"
                    text_pred = f"Pred: {pred_label}"
                    
                    # Color code: green if correct, red if wrong
                    color = (0, 255, 0) if pred_label == true_label else (255, 0, 0)
                    
                    cv2.putText(frame, text_true, (10, 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.putText(frame, text_pred, (10, 45), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    annotated_frames.append(frame)
        
        # Convert to video format: (num_frames, C, H, W)
        video_array = np.stack(annotated_frames)
        video_array = video_array.transpose(0, 3, 1, 2)
        
        # Log to wandb
        self.logger.experiment.log({
            "test_set_video": wandb.Video(
                video_array, 
                fps=fps,
                format="mp4",
                caption=f"Continuous test set predictions ({len(annotated_frames)} frames)"
            )
        })
        
        print(f"Logged video with {len(annotated_frames)} frames at {fps} fps")
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
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
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=config.callbacks.early_stopping_patience, mode='min')

    trainer = L.Trainer(
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stopping_callback],
        max_epochs=config.max_epochs,
    )

    file_paths, labels, class_names = load_data(**config.data_loading)

    train_data, valid_data, test_data = train_test_split(file_paths, labels, **config.data)

    class_ids, counts = torch.unique(torch.tensor(train_data[1]), return_counts=True)
    weights = counts.sum() / counts

    print(class_names, class_ids, counts)

    assert len(counts) == len(class_names), f"Number of classes ({len(class_names)}) should equal number of counts ({len(counts)})"

    for weight, name in zip(weights, class_names):
        print(f"{name}: {weight:.3f}")

    data_module = MultiImageDataModule(train_data, valid_data, test_data)

    model = MultiImageTransformerClassifier(**config.model, max_epochs=config.max_epochs, weights=weights,
                                            class_names=class_names)

    if config.fine_tune_lr:
        tuner = Tuner(trainer)
        lr_finder = tuner.lr_find(model, datamodule=data_module, min_lr=1e-6, max_lr=1e-2, num_training=200)
        fig = lr_finder.plot(suggest=True)
        plt.savefig('lr_finder.png')
    else:
        trainer.fit(model, data_module)
        trainer.test(model, data_module, ckpt_path='best')

if __name__ == "__main__":
    main()

