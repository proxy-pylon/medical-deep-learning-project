import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from PIL import Image
import cv2
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, confusion_matrix, roc_curve, auc
)
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchvision.models as models
from torchvision.models import ResNet50_Weights, EfficientNet_B0_Weights

import torch
import cv2
from torch.utils.data import Dataset
import torch
from tqdm import tqdm

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

import torch
import torch.nn as nn
import torch.nn.functional as F

import warnings
warnings.filterwarnings('ignore')

class Config:
    # Data Paths
    HAM10000_BASE = './data'
    ISIC_BASE =  'not defined lol'
    # Output Paths
    OUTPUT_DIR = './output/'
    CHECKPOINT_DIR = OUTPUT_DIR + 'checkpoints'
    RESULTS_DIR = OUTPUT_DIR + 'results'

    # Model configurations
    MODEL_NAME = 'resnet50' # Options: 'resnet50', 'efficientnet', 'vgg16'
    IMG_SIZE = 224
    NUM_CLASSES = 2 # Binary: melanoma vs benign
    PRETRAINED = True

    # Training configurations
    BATCH_SIZE = 32
    NUM_EPOCHS = 250
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4
    EARLY_STOPPING_PATIENCE = 30

    # Warmup / fine-tune
    FREEZE_EPOCHS = 3            # 2–3 as you like
    HEAD_LR_WARMUP = 1e-3        # head-only phase
    HEAD_LR_FINETUNE = 1e-4      # during full fine-tune

    # Discriminative LRs for backbone (lower -> earlier layers)
    BACKBONE_LR_LOW = 1e-5
    BACKBONE_LR_MID = 2e-5
    BACKBONE_LR_HIGH = 3e-5

    #Dataset split ratio and seeding
    TEST_SIZE = 0.30
    VAL_SIZE = 0.20
    RANDOM_STATE = 42

    # Augmentation
    USE_MIXUP = False
    USE_CUTMIX = False

    # Device
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    NUM_WORKERS = 2
      
def load_ham10000_data(base_path):
    """Loading HAM10000 dataset"""
    print('Loading HAM10000 dataset....')

    # Load metadata
    metadata_path = os.path.join(base_path, 'HAM10000_metadata.csv')
    df = pd.read_csv(metadata_path)

    # Create image paths
    def get_image_path(image_id):
        # Check for image directories
        part1 = os.path.join(base_path, 'HAM10000_images_part_1', f'{image_id}.jpg')
        part2 = os.path.join(base_path, 'HAM10000_images_part_2', f'{image_id}.jpg')

        if os.path.exists(part1):
            return part1
        elif os.path.exists(part2):
            return part2
        else: 
            return None
            
    # Getting image path
    df['image_path'] = df['image_id'].apply(get_image_path)

    # Remove missing images
    df = df[df['image_path'].notna()].reset_index(drop=True)

    # Binary classification: melanoma (mel) vs others
    df['binary_label'] = (df['dx'] == 'mel').astype(int)

    print(f"Loaded {len(df)} images")
    print(f"Melanoma: {df['binary_label'].sum()}")
    print(f"Benign: {len(df) - df['binary_label'].sum()}")
    print(f"\nClass Distribution:")
    print(df['dx'].value_counts())

    return df

class MelanomaDataset(Dataset):
    """Custom dataset for melanoma classification"""

    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.loc[idx, 'image_path']
        label = self.df.loc[idx, 'binary_label']

        # Load image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        return {
            'image': image,
            'label': torch.tensor(label, dtype=torch.long)  # ✅ fixed typo
        }

def get_train_transform(img_size=224):
    return A.Compose([
        A.RandomResizedCrop(
                size=(img_size, img_size),
                scale=(0.90, 1.00),
                ratio=(0.9, 1.1),
                p=1.0
            ),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=20, border_mode=cv2.BORDER_REFLECT_101, p=0.7),
        # Small center-ish crop to shave off potential reflected slivers
        A.CenterCrop(height=img_size, width=img_size, p=1.0),

        A.Affine(scale=(0.95, 1.05), translate_percent=(-0.02, 0.02),
                 shear=(-5, 5), mode=cv2.BORDER_REFLECT_101, p=0.5),

        A.RandomBrightnessContrast(0.10, 0.10, p=0.3),
        A.ColorJitter(0.05, 0.05, 0.05, 0.02, p=0.2),
        A.CLAHE(clip_limit=(1, 2), tile_grid_size=(8, 8), p=0.2),

        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0)),
            A.GaussianBlur(blur_limit=(3, 7)),
            A.MedianBlur(blur_limit=5),
        ], p=0.3),
        A.CoarseDropout(max_holes=1, min_holes=1,
                        max_height=int(0.08*img_size), max_width=int(0.08*img_size),
                        min_height=8, min_width=8, fill_value=0, p=0.15),

        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


def get_val_transform(img_size=224):
    """Validation/test augmentation pipeline"""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
        ToTensorV2()
    ])

class MelanomaClassifier(nn.Module):
    """CNN classifier with transfer learning"""
    def __init__(self, model_name='resnet50', num_classes=2, pretrained=True):
        super(MelanomaClassifier, self).__init__()
        if model_name == 'resnet50':
            if pretrained:
                self.backbone = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            else:
                self.backbone = models.resnet50(weights=None)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif model_name == 'efficientnet':
            if pretrained:
                self.backbone = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
            else:
                self.backbone = models.efficientnet_b0(weights=None)
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        #classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    def forward(self, x):
        features = self.backbone(x)
        output = self.classifier(features)
        return output, features

def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    """Freeze or unfreeze backbone params in place."""
    for p in model.backbone.parameters():
        p.requires_grad = trainable

def get_param_groups_discriminative(model: nn.Module, config: Config):
    """
    Build parameter groups with discriminative LRs for supported backbones.
    Head gets HEAD_LR_FINETUNE, backbone gets tiered 1e-5..3e-5.
    """
    param_groups = []

    # 1) Classifier / head
    param_groups.append({
        "params": list(model.classifier.parameters()),
        "lr": config.HEAD_LR_FINETUNE,
        "weight_decay": config.WEIGHT_DECAY
    })

    # 2) Backbone tiers
    bb = model.backbone
    low = config.BACKBONE_LR_LOW
    mid = config.BACKBONE_LR_MID
    high = config.BACKBONE_LR_HIGH

    if isinstance(bb, models.ResNet):
        # Early layers: tiny LR
        tiers = [
            (["conv1", "bn1"], low),
            (["layer1"], low),
            (["layer2"], mid),
            (["layer3"], high),
            (["layer4"], high),
        ]
        for names, lr in tiers:
            params = []
            for n in names:
                m = getattr(bb, n)
                params += list(m.parameters())
            param_groups.append({"params": params, "lr": lr, "weight_decay": config.WEIGHT_DECAY})

    elif hasattr(bb, "features"):  # EfficientNet-style
        # Rough split of stages from shallow to deep
        feat = bb.features
        n = len(feat)
        cut1 = max(1, n // 3)
        cut2 = max(cut1 + 1, (2 * n) // 3)

        early = list(feat[:cut1].parameters())             # low
        middle = list(feat[cut1:cut2].parameters())        # mid
        late = list(feat[cut2:].parameters())              # high

        param_groups += [
            {"params": early,  "lr": low,  "weight_decay": config.WEIGHT_DECAY},
            {"params": middle, "lr": mid,  "weight_decay": config.WEIGHT_DECAY},
            {"params": late,   "lr": high, "weight_decay": config.WEIGHT_DECAY},
        ]
    else:
        # Fallback: if we can’t tier, at least give the whole backbone a sane small LR
        param_groups.append({
            "params": [p for p in model.backbone.parameters() if p.requires_grad],
            "lr": mid, "weight_decay": config.WEIGHT_DECAY
        })

    return param_groups

def build_optimizer_warmup(model: nn.Module, config: Config):
    """
    AdamW on the head only, LR = 1e-3. Backbone is expected to be frozen.
    """
    head_params = [p for p in model.classifier.parameters() if p.requires_grad]
    return optim.AdamW(head_params, lr=config.HEAD_LR_WARMUP, weight_decay=config.WEIGHT_DECAY)

def build_optimizer_finetune(model: nn.Module, config: Config):
    """
    AdamW with discriminative LRs across backbone tiers + head.
    """
    groups = get_param_groups_discriminative(model, config)
    return optim.AdamW(groups)  # each group already sets its own lr/wd


# Bigger gamma to focus more on hard examples
# Bigger alpha to balance classes
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # tensor of shape [num_classes] or scalar
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Cross entropy per sample
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        # Get predicted probabilities for the correct class
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc='Training')

    for batch in pbar:
        images = batch['image'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        outputs, _ = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix({
            'loss': running_loss / total,
            'acc': 100 * correct / total
        })
    
    # FIXED: Return statement moved outside the loop
    return running_loss / total, correct / total

# def validate(model, loader, criterion, device):
#     """
#     Validate the model on the given data loader.

#     Args:
#         model: The neural network model.
#         loader: DataLoader for the validation dataset.
#         criterion: Loss function.
#         device: Device to run the model on (e.g., 'cuda' or 'cpu').

#     Returns:
#         Tuple containing average loss and accuracy.
#     """
#     model.eval()
#     running_loss = 0.0
#     correct = 0
#     total = 0

#     with torch.no_grad():
#         for batch in tqdm(loader, desc='Validation'):
#             images = batch['image'].to(device)
#             labels = batch['label'].to(device)

#             outputs, _ = model(images)
#             loss = criterion(outputs, labels)

#             running_loss += loss.item() * images.size(0)

#             # Fix: correctly unpack the max output
#             _, predicted = outputs.max(1)

#             total += labels.size(0)
#             correct += predicted.eq(labels).sum().item()

#     avg_loss = running_loss / total
#     accuracy = correct / total
#     return avg_loss, accuracy

from sklearn.metrics import f1_score

# Validation function with F1 score instead of accuracy
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch in tqdm(loader, desc='Validation'):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            outputs, _ = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)

            _, predicted = outputs.max(1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    avg_loss = running_loss / len(loader.dataset)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    accuracy = (np.array(all_preds) == np.array(all_labels)).mean()
    return avg_loss, accuracy, f1


def evaluate_model(model, loader, device):
    """Testing"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in tqdm(loader, desc='Evaluating'):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)

            outputs, _ = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = outputs.max(1)

            all_preds.extend(preds.cpu().numpy())  # FIXED: .cpu() not .cput()
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    roc_auc = roc_auc_score(all_labels, all_probs)
    cm = confusion_matrix(all_labels, all_preds)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': cm,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs
    }

def plot_training_history(history, save_path):
    """Training curves"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))  # FIXED: subplots not subplts
    
    # Loss
    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy
    axes[1].plot(history['train_acc'], label='Train Acc')
    axes[1].plot(history['val_acc'], label='Val Acc')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(cm, save_path):
    'plt confusion matrix'
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=['Benign', 'Melanoma'],
               yticklabels=['Benign','Melanoma'])
    plt.ylabel('True Label')
    plt.xlabel('Prdicted Label')
    plt.title('Congusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_roc_curve(labels, probs, save_path):
    """
    Plot and save the ROC curve.

    Args:
        labels: Ground truth binary labels (0 or 1).
        probs: Predicted probabilities for the positive class.
        save_path: Path to save the ROC curve image.
    """
    # Get false positive rate, true positive rate, and thresholds
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)

    # Create the plot
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save and close the plot
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def main(use_ham10000=True, use_isic=False):
    'Main training pipeline'
    config = Config()
    # Create output directories
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    print('=' * 70)
    print("MELANOMA CLASSIFICATION - TRAINING PIPELINE")
    print('=' * 70)
    print(f"Device: {config.DEVICE}")
    print(f"Model: {config.MODEL_NAME}")
    print(f"Image size: {config.IMG_SIZE}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Learning rate: {config.LEARNING_RATE}")
    print("=" * 70)
    # Load data
    if use_ham10000:
        df = load_ham10000_data(config.HAM10000_BASE)
    elif use_isic:
        df = load_isic_data(config.ISIC_BASE)
        # Filter to train split for now
        df = df[df['split'] == 'train'].reset_index(drop=True)
    else:
        raise ValueError("Must sepcify eiter HAM10000 or ISIC dataset")
    #split data
    print("\nSplitting dataset...")
    train_df, temp_df = train_test_split(
        df, test_size = config.TEST_SIZE + config.VAL_SIZE,
        random_state = config.RANDOM_STATE, stratify=df['binary_label']
    )
    val_df, test_df = train_test_split(
        temp_df, test_size = config.TEST_SIZE / (config.TEST_SIZE + config.VAL_SIZE),
        random_state = config.RANDOM_STATE, stratify=temp_df['binary_label']
    )
    print(f"Train: {len(train_df)} | val: {len(val_df)}| Test: {len(test_df)}")

    # Create datasets
    train_dataset = MelanomaDataset(train_df, get_train_transform(config.IMG_SIZE))
    val_dataset = MelanomaDataset(val_df, get_val_transform(config.IMG_SIZE))
    test_dataset = MelanomaDataset(test_df, get_val_transform(config.IMG_SIZE))
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, 
                             shuffle=True, num_workers=config.NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, 
                           shuffle=False, num_workers=config.NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, 
                            shuffle=False, num_workers=config.NUM_WORKERS)
    
    # Calculate class weights for imbalanced data
    class_counts = train_df['binary_label'].value_counts()
    total = len(train_df)
    class_weights = {
        0: total/(2*class_counts[0]), 
        1: total/(2*class_counts[1]) * 1 # multiply by some scalar to give more weight to melanoma
    }
    weights = torch.FloatTensor([class_weights[0], class_weights[1]]).to(config.DEVICE)
    
    print(f"\nClass weights: {class_weights}")
    
    # Create model
    print(f"\nCreating {config.MODEL_NAME} model...")
    model = MelanomaClassifier(config.MODEL_NAME, config.NUM_CLASSES, config.PRETRAINED)
    model = model.to(config.DEVICE)
    
    # Loss and optimizer
    # criterion = nn.CrossEntropyLoss(weight=weights) # This is bad with imbalanced data
    criterion = FocalLoss(alpha=weights, gamma=2.0)
    
    # Phase 1: freeze backbone, head-only AdamW @ 1e-3
    set_backbone_trainable(model, trainable=False)
    optimizer = build_optimizer_warmup(model, config)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    current_phase = "warmup"
    
    # Training loop
    print("\nStarting training...")
    print("=" * 70)
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_acc = 0.0
    best_val_f1 = 0.0
    patience_counter = 0
    
    for epoch in range(config.NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{config.NUM_EPOCHS}")
        print("-" * 70)
        
        # Switch from warmup to fine-tune after FREEZE_EPOCHS
        if current_phase == "warmup" and epoch == config.FREEZE_EPOCHS:
            print("\nUnfreezing backbone and switching to discriminative LRs...")
            set_backbone_trainable(model, trainable=True)
            optimizer = build_optimizer_finetune(model, config)  # AdamW with param groups
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
            current_phase = "finetune"

        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, 
                                           optimizer, config.DEVICE)
        
        # Validate
        val_loss, val_acc, val_f1 = validate(model, val_loader, criterion, config.DEVICE)
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Store history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        print(f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        # if val_acc > best_val_acc:
        #     best_val_acc = val_acc
        #     patience_counter = 0
        #     torch.save({
        #         'epoch': epoch,
        #         'model_state_dict': model.state_dict(),
        #         'optimizer_state_dict': optimizer.state_dict(),
        #         'val_acc': val_acc,
        #         'val_loss': val_loss,
        #     }, os.path.join(config.CHECKPOINT_DIR, 'best_model.pth'))
        #     print(f"✓ Best model saved! Val Acc: {val_acc:.4f}")
        # else:
        #     patience_counter += 1

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_f1,
                'val_loss': val_loss,
            }, os.path.join(config.CHECKPOINT_DIR, 'best_model.pth'))
            print(f"✓ Best model saved! Val F1: {val_f1:.4f}")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= config.EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    # Plot training history
    plot_training_history(history, os.path.join(config.RESULTS_DIR, 'training_history.png'))
    
    # Load best model and evaluate
    print("\n" + "=" * 70)
    print("EVALUATING BEST MODEL ON TEST SET")
    print("=" * 70)
    
    checkpoint = torch.load(os.path.join(config.CHECKPOINT_DIR, 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    metrics = evaluate_model(model, test_loader, config.DEVICE)
    
    print(f"\nTest Results:")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")
    print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(f"\nConfusion Matrix:")
    print(metrics['confusion_matrix'])
    
    # Plot results
    plot_confusion_matrix(metrics['confusion_matrix'], 
                         os.path.join(config.RESULTS_DIR, 'confusion_matrix.png'))
    plot_roc_curve(metrics['labels'], metrics['probabilities'],
                  os.path.join(config.RESULTS_DIR, 'roc_curve.png'))
    
    print(f"\nResults saved to: {config.RESULTS_DIR}")
    print("Training complete!")
    
    return model, metrics, history

if __name__ == "__main__":
    # Run training
    model, metrics, history = main(use_ham10000=True, use_isic=True)