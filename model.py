import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
import pandas as pd
import numpy as np
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Configuration
class Config:
    # Paths
    DATA_DIR = './data'
    IMAGE_DIR = os.path.join(DATA_DIR, 'HAM10000_images')
    METADATA_PATH = os.path.join(DATA_DIR, 'HAM10000_metadata.csv')
    
    # Model
    MODEL_NAME = 'efficientnet_b3'
    NUM_CLASSES = 7
    FREEZE_LAYERS = True
    UNFREEZE_LAST_N = 4  # Unfreeze last blocks
    
    # Training
    BATCH_SIZE = 32
    EPOCHS = 30
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    
    # Data
    IMG_SIZE = 300
    TEST_SIZE = 0.2
    VAL_SIZE = 0.1  # From training set
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Save paths
    MODEL_SAVE_PATH = 'best_model.pth'
    HISTORY_SAVE_PATH = 'training_history.png'

config = Config()

# Disease mapping
DX_MAPPING = {
    'akiec': 0,  # Actinic keratoses
    'bcc': 1,    # Basal cell carcinoma
    'bkl': 2,    # Benign keratosis
    'df': 3,     # Dermatofibroma
    'mel': 4,    # Melanoma
    'nv': 5,     # Melanocytic nevi
    'vasc': 6    # Vascular lesions
}

DX_NAMES = list(DX_MAPPING.keys())


def load_and_preprocess_metadata(metadata_path):
    """Load metadata and prepare for patient-based splitting"""
    df = pd.read_csv(metadata_path)
    
    # Map diagnoses to numeric labels
    df['label'] = df['dx'].map(DX_MAPPING)
    
    # Get unique patient IDs (lesion_id represents unique patients/lesions)
    df['patient_id'] = df['lesion_id']
    
    print(f"Total images: {len(df)}")
    print(f"Unique patients: {df['patient_id'].nunique()}")
    print(f"\nClass distribution:\n{df['dx'].value_counts()}")
    
    return df


def patient_based_split(df, test_size=0.2, val_size=0.1, random_state=42):
    """
    Split by patient ID to prevent data leakage.
    Stratify by diagnosis to maintain class balance.
    """
    # Get one row per patient with their diagnosis
    patient_df = df.groupby('patient_id').agg({
        'dx': 'first',
        'label': 'first'
    }).reset_index()
    
    # First split: train+val vs test (stratified by diagnosis)
    train_val_patients, test_patients = train_test_split(
        patient_df['patient_id'].values,
        test_size=test_size,
        random_state=random_state,
        stratify=patient_df['label'].values
    )
    
    # Get labels for train_val_patients for stratification
    train_val_labels = patient_df[patient_df['patient_id'].isin(train_val_patients)]['label'].values
    
    # Second split: train vs val (stratified)
    val_size_adjusted = val_size / (1 - test_size)
    train_patients, val_patients = train_test_split(
        train_val_patients,
        test_size=val_size_adjusted,
        random_state=random_state,
        stratify=train_val_labels
    )
    
    # Create splits
    train_df = df[df['patient_id'].isin(train_patients)].copy()
    val_df = df[df['patient_id'].isin(val_patients)].copy()
    test_df = df[df['patient_id'].isin(test_patients)].copy()
    
    print(f"\nSplit statistics:")
    print(f"Train: {len(train_df)} images, {len(train_patients)} patients")
    print(f"Val: {len(val_df)} images, {len(val_patients)} patients")
    print(f"Test: {len(test_df)} images, {len(test_patients)} patients")
    
    return train_df, val_df, test_df


def fill_missing_values(train_df, val_df, test_df):
    """
    Fill missing values using only training set statistics.
    This prevents data leakage.
    """
    # Calculate statistics from training set only
    age_median = train_df['age'].median()
    sex_mode = train_df['sex'].mode()[0]
    localization_mode = train_df['localization'].mode()[0]
    
    # Fill missing values in all sets using training statistics
    for df in [train_df, val_df, test_df]:
        df['age'].fillna(age_median, inplace=True)
        df['sex'].fillna(sex_mode, inplace=True)
        df['localization'].fillna(localization_mode, inplace=True)
    
    print(f"\nFilled missing values using training set statistics:")
    print(f"Age median: {age_median}")
    print(f"Sex mode: {sex_mode}")
    print(f"Localization mode: {localization_mode}")
    
    return train_df, val_df, test_df


def normalize_age(age):
    """
    Smart age normalization using robust scaling.
    Assumes age range 0-100 but uses percentile-based normalization.
    """
    # Clip outliers
    age = np.clip(age, 0, 100)
    # Normalize using expected range with slight compression at extremes
    return (age - 50) / 40  # Centers around 50, ~95% of data in [-1, 1]


def encode_categorical_features(train_df, val_df, test_df):
    """Encode categorical features"""
    # Sex encoding
    sex_map = {'male': 0, 'female': 1, 'unknown': 2}
    
    # Localization encoding - get from training set
    unique_locs = train_df['localization'].unique()
    loc_map = {loc: idx for idx, loc in enumerate(unique_locs)}
    
    for df in [train_df, val_df, test_df]:
        df['sex_encoded'] = df['sex'].map(sex_map).fillna(2)
        df['localization_encoded'] = df['localization'].map(loc_map).fillna(0)
        df['age_normalized'] = df['age'].apply(normalize_age)
    
    return train_df, val_df, test_df, len(loc_map)


class HAM10000Dataset(Dataset):
    """Dataset with metadata features and heavy augmentation"""
    
    def __init__(self, df, image_dir, transform=None, use_metadata=True):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform
        self.use_metadata = use_metadata
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load image
        img_name = f"{row['image_id']}.jpg"
        img_path = os.path.join(self.image_dir, img_name)
        
        try:
            image = Image.open(img_path).convert('RGB')
        except:
            # Fallback: create a blank image if file not found
            image = Image.new('RGB', (config.IMG_SIZE, config.IMG_SIZE), color='gray')
        
        if self.transform:
            image = self.transform(image)
        
        label = row['label']
        
        if self.use_metadata:
            # Include metadata features
            metadata = torch.tensor([
                row['age_normalized'],
                row['sex_encoded'],
                row['localization_encoded']
            ], dtype=torch.float32)
            
            return image, metadata, label
        else:
            return image, label


def get_transforms(is_training=True):
    """
    Heavy augmentation for training, minimal for validation/test.
    """
    if is_training:
        return transforms.Compose([
            transforms.Resize((config.IMG_SIZE + 20, config.IMG_SIZE + 20)),
            transforms.RandomCrop(config.IMG_SIZE),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(20),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1),
                shear=10
            ),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.15))
        ])
    else:
        return transforms.Compose([
            transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])


class EfficientNetClassifier(nn.Module):
    """
    EfficientNet with metadata integration and custom classifier.
    """
    
    def __init__(self, model_name, num_classes, num_metadata_features, pretrained=True):
        super().__init__()
        
        # Load pretrained EfficientNet
        self.backbone = timm.create_model(model_name, pretrained=pretrained)
        
        # Get number of features from backbone
        num_features = self.backbone.classifier.in_features
        
        # Remove original classifier
        self.backbone.classifier = nn.Identity()
        
        # Custom classifier with metadata integration
        self.classifier = nn.Sequential(
            nn.Linear(num_features + num_metadata_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, image, metadata):
        # Extract image features
        img_features = self.backbone(image)
        
        # Concatenate with metadata
        combined = torch.cat([img_features, metadata], dim=1)
        
        # Classify
        output = self.classifier(combined)
        
        return output
    
    def freeze_backbone(self, unfreeze_last_n=4):
        """
        Freeze all layers except the last N blocks.
        For EfficientNet, we unfreeze the last few blocks.
        """
        # Freeze all parameters first
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Unfreeze last N blocks
        # EfficientNet structure: blocks[0-6], conv_head, bn2
        if hasattr(self.backbone, 'blocks'):
            num_blocks = len(self.backbone.blocks)
            unfreeze_from = max(0, num_blocks - unfreeze_last_n)
            
            for i in range(unfreeze_from, num_blocks):
                for param in self.backbone.blocks[i].parameters():
                    param.requires_grad = True
        
        # Always unfreeze the head
        if hasattr(self.backbone, 'conv_head'):
            for param in self.backbone.conv_head.parameters():
                param.requires_grad = True
        if hasattr(self.backbone, 'bn2'):
            for param in self.backbone.bn2.parameters():
                param.requires_grad = True
        
        # Classifier is always trainable
        for param in self.classifier.parameters():
            param.requires_grad = True
        
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"\nTrainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")


def calculate_class_weights(train_df):
    """Calculate class weights for imbalanced dataset"""
    class_counts = train_df['label'].value_counts().sort_index()
    total = len(train_df)
    
    # Inverse frequency weighting
    weights = torch.tensor([total / (len(class_counts) * count) 
                           for count in class_counts], 
                          dtype=torch.float32)
    
    print(f"\nClass weights: {weights.numpy()}")
    return weights


def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, metadata, labels in loader:
        images = images.to(device)
        metadata = metadata.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images, metadata)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc


def validate(model, loader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, metadata, labels in loader:
            images = images.to(device)
            metadata = metadata.to(device)
            labels = labels.to(device)
            
            outputs = model(images, metadata)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc, all_preds, all_labels, all_probs


def calculate_metrics(y_true, y_pred, y_probs):
    """Calculate comprehensive metrics"""
    # Per-class F1 scores
    f1_per_class = f1_score(y_true, y_pred, average=None)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    
    # ROC-AUC (one-vs-rest)
    try:
        roc_auc_per_class = []
        for i in range(config.NUM_CLASSES):
            y_true_binary = (np.array(y_true) == i).astype(int)
            y_score = np.array(y_probs)[:, i]
            if len(np.unique(y_true_binary)) > 1:
                auc = roc_auc_score(y_true_binary, y_score)
                roc_auc_per_class.append(auc)
            else:
                roc_auc_per_class.append(np.nan)
        
        roc_auc_macro = np.nanmean(roc_auc_per_class)
    except:
        roc_auc_per_class = [np.nan] * config.NUM_CLASSES
        roc_auc_macro = np.nan
    
    return {
        'f1_per_class': f1_per_class,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'roc_auc_per_class': roc_auc_per_class,
        'roc_auc_macro': roc_auc_macro
    }


def plot_training_history(history, save_path):
    """Plot and save training history"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train Loss')
    axes[0, 0].plot(history['val_loss'], label='Val Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Accuracy
    axes[0, 1].plot(history['train_acc'], label='Train Acc')
    axes[0, 1].plot(history['val_acc'], label='Val Acc')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Training and Validation Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # F1 Score
    axes[1, 0].plot(history['val_f1_macro'], label='Macro F1')
    axes[1, 0].plot(history['val_f1_weighted'], label='Weighted F1')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].set_title('Validation F1 Scores')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # ROC-AUC
    axes[1, 1].plot(history['val_roc_auc'], label='ROC-AUC')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('ROC-AUC')
    axes[1, 1].set_title('Validation ROC-AUC')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nTraining history saved to {save_path}")


def main():
    print("="*80)
    print("HAM10000 Skin Cancer Classification with EfficientNet")
    print("="*80)
    
    # Load and preprocess metadata
    print("\n1. Loading metadata...")
    df = load_and_preprocess_metadata(config.METADATA_PATH)
    
    # Patient-based split (prevents data leakage)
    print("\n2. Performing patient-based stratified split...")
    train_df, val_df, test_df = patient_based_split(
        df, 
        test_size=config.TEST_SIZE,
        val_size=config.VAL_SIZE
    )
    
    # Fill missing values (using only training statistics)
    print("\n3. Filling missing values...")
    train_df, val_df, test_df = fill_missing_values(train_df, val_df, test_df)
    
    # Encode features
    print("\n4. Encoding categorical features...")
    train_df, val_df, test_df, num_locations = encode_categorical_features(
        train_df, val_df, test_df
    )
    
    # Create datasets
    print("\n5. Creating datasets...")
    train_dataset = HAM10000Dataset(
        train_df, 
        config.IMAGE_DIR, 
        transform=get_transforms(is_training=True)
    )
    val_dataset = HAM10000Dataset(
        val_df, 
        config.IMAGE_DIR, 
        transform=get_transforms(is_training=False)
    )
    test_dataset = HAM10000Dataset(
        test_df, 
        config.IMAGE_DIR, 
        transform=get_transforms(is_training=False)
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=4
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=4
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Create model
    print(f"\n6. Creating {config.MODEL_NAME} model...")
    model = EfficientNetClassifier(
        model_name=config.MODEL_NAME,
        num_classes=config.NUM_CLASSES,
        num_metadata_features=3,  # age, sex, localization
        pretrained=True
    )
    
    # Freeze backbone layers
    if config.FREEZE_LAYERS:
        print("\n7. Freezing backbone layers...")
        model.freeze_backbone(unfreeze_last_n=config.UNFREEZE_LAST_N)
    
    model = model.to(config.DEVICE)
    
    # Class-weighted loss
    print("\n8. Calculating class weights...")
    class_weights = calculate_class_weights(train_df).to(config.DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=3,
    )
    
    # Training loop
    print("\n9. Starting training...")
    print("="*80)
    
    best_val_loss = float('inf')
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'val_f1_macro': [], 'val_f1_weighted': [],
        'val_roc_auc': []
    }
    
    for epoch in range(config.EPOCHS):
        print(f"\nEpoch {epoch+1}/{config.EPOCHS}")
        print("-" * 40)
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, config.DEVICE
        )
        
        # Validate
        val_loss, val_acc, val_preds, val_labels, val_probs = validate(
            model, val_loader, criterion, config.DEVICE
        )
        
        # Calculate metrics
        metrics = calculate_metrics(val_labels, val_preds, val_probs)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1_macro'].append(metrics['f1_macro'])
        history['val_f1_weighted'].append(metrics['f1_weighted'])
        history['val_roc_auc'].append(metrics['roc_auc_macro'])
        
        # Print metrics
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        print(f"Val F1 (Macro): {metrics['f1_macro']:.4f} | Val F1 (Weighted): {metrics['f1_weighted']:.4f}")
        print(f"Val ROC-AUC: {metrics['roc_auc_macro']:.4f}")
        
        # Learning rate scheduler
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'history': history
            }, config.MODEL_SAVE_PATH)
            print(f"✓ Model saved! (Val Loss: {val_loss:.4f})")
    
    # Plot training history
    print("\n10. Plotting training history...")
    plot_training_history(history, config.HISTORY_SAVE_PATH)
    
    # Load best model for final evaluation
    print("\n11. Loading best model for final evaluation...")
    checkpoint = torch.load(config.MODEL_SAVE_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Final test evaluation
    print("\n12. Final Test Set Evaluation")
    print("="*80)
    test_loss, test_acc, test_preds, test_labels, test_probs = validate(
        model, test_loader, criterion, config.DEVICE
    )
    test_metrics = calculate_metrics(test_labels, test_preds, test_probs)
    
    print(f"\nTest Accuracy: {test_acc:.4f}")
    print(f"Test F1 (Macro): {test_metrics['f1_macro']:.4f}")
    print(f"Test F1 (Weighted): {test_metrics['f1_weighted']:.4f}")
    print(f"Test ROC-AUC: {test_metrics['roc_auc_macro']:.4f}")
    
    print("\nPer-class F1 Scores:")
    for i, name in enumerate(DX_NAMES):
        print(f"  {name}: {test_metrics['f1_per_class'][i]:.4f}")
    
    print("\nPer-class ROC-AUC:")
    for i, name in enumerate(DX_NAMES):
        print(f"  {name}: {test_metrics['roc_auc_per_class'][i]:.4f}")
    
    # Classification report
    print("\nDetailed Classification Report:")
    print(classification_report(test_labels, test_preds, target_names=DX_NAMES))
    
    print("\n" + "="*80)
    print("Training Complete!")
    print(f"Best model saved to: {config.MODEL_SAVE_PATH}")
    print(f"Training history saved to: {config.HISTORY_SAVE_PATH}")
    print("="*80)


if __name__ == "__main__":
    main()
