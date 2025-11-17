# Standard library
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Third-party libraries
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm import tqdm
from numpy.typing import ArrayLike
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
    brier_score_loss,
    log_loss,
    precision_recall_curve
)
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from torchvision.models import ResNet50_Weights, EfficientNet_B0_Weights
import json
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
import time
import psutil

# Warnings
warnings.filterwarnings('ignore')


class Config:

    # Data Paths
    HAM10000_BASE = './data'
    ISIC_BASE = 'not defined lol'
    # Output Paths
    OUTPUT_DIR = './output/'
    CHECKPOINT_DIR = OUTPUT_DIR + 'checkpoints'
    RESULTS_DIR = OUTPUT_DIR + 'results'

    # Model configurations
    MODEL_NAME = 'resnet50'  # Options: 'resnet50', 'efficientnet', 'vgg16'
    IMG_SIZE = 224
    NUM_CLASSES = 2  # Binary: melanoma vs benign
    PRETRAINED = True

    # Training configurations
    BATCH_SIZE = 64
    MAX_EPOCHS = 150
    WARMUP_EPOCHS = 8
    LEARNING_RATE = 0.001
    EARLY_STOPPING_PATIENCE = 10

    # Warmup / fine-tune
    FREEZE_EPOCHS = 3
    HEAD_LR_WARMUP = 5e-5
    HEAD_LR_FINETUNE = 1e-4

    # Discriminative LRs for backbone
    HEAD_LR = 1e-3     
    BACKBONE_LR_LOW = 1e-5
    BACKBONE_LR_MID = 2e-5
    BACKBONE_LR_HIGH = 3e-5
    WEIGHT_DECAY = 2e-4

    # ENHANCED REGULARIZATION (FROM RESNET50)
    DROPOUT = 0.6
    GRADIENT_CLIP = 1.0
    LABEL_SMOOTHING = 0.1
    MIN_MELANOMA_F1 = 0.90  # For clinical validation

    # Dataset split ratio and seeding
    TEST_SIZE = 0.30
    VAL_SIZE = 0.20
    RANDOM_STATE = 42

    # Augmentation flags
    USE_MIXUP = True
    USE_CUTMIX = True

    # Multimodal configurations
    USE_MULTIMODAL = True  # Switch to enable/disable multimodality
    MULTIMODAL_DROPOUT = 0.3 #why such drop out
    CLINICAL_FEATURE_DIM = 10 #code dynamically calculates actual clinical features
    #needs 10 for model initialization
    FUSION_HIDDEN_DIM = 256

    # Device
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    NUM_WORKERS = 2


def load_ham10000_data(base_path: Union[str, os.PathLike]) -> pd.DataFrame:
    print('Loading HAM10000 dataset....')
    metadata_path = os.path.join(base_path, 'HAM10000_metadata.csv')
    df = pd.read_csv(metadata_path)

    def get_image_path(image_id: str) -> Optional[str]:
        """Return path to image file if present in either image folder."""
        possible_paths = [
            os.path.join(base_path, 'HAM10000_images_part_1', f'{image_id}.jpg'),
            os.path.join(base_path, 'HAM10000_images_part_2', f'{image_id}.jpg'),
            os.path.join(base_path, f'{image_id}.jpg')
        ]

        for img_path in possible_paths:
            if os.path.exists(img_path):
                return img_path
        return None

    df['image_path'] = df['image_id'].apply(get_image_path)
    df = df[df['image_path'].notna()].reset_index(drop=True)
    df['binary_label'] = (df['dx'] == 'mel').astype(int)

    print(f"Loaded {len(df)} images")
    print(f"Melanoma: {df['binary_label'].sum()}")
    print(f"Benign: {len(df) - df['binary_label'].sum()}")
    print(f"\nClass Distribution:")
    print(df['dx'].value_counts())

    return df


class MelanomaDataset(Dataset):

    def __init__(self, dataframe: pd.DataFrame, transform: Optional[A.Compose] = None,
                 use_multimodal: bool = False) -> None:
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform
        self.use_multimodal = use_multimodal

         # Preprocess clinical data if using multimodality
        if self.use_multimodal:
            self._preprocess_clinical_data()

    def _preprocess_clinical_data(self):
        """Handle missing clinical data with simple imputation."""

        # Age: median imputation
        if 'age' in self.df.columns:
            self.df['age'] = self.df['age'].fillna(self.df['age'].median())
            self.df['age'] = (self.df['age'] - self.df['age'].mean()) / self.df['age'].std()

        # Sex: 'unknown' category
        if 'sex' in self.df.columns:
            self.df['sex'] = self.df['sex'].fillna('unknown')

        # Localization: 'unknown' category
        if 'localization' in self.df.columns:
            self.df['localization'] = self.df['localization'].fillna('unknown')

        # One-hot encode categorical variables
        self.df = pd.get_dummies(self.df, columns=['sex', 'localization'], prefix=['sex', 'loc'])

    def __len__(self) -> int:
        """Number of samples."""
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_path: str = self.df.loc[idx, 'image_path']
        label: int = int(self.df.loc[idx, 'binary_label'])

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        sample = {
            'image': image,
            'label': torch.tensor(label, dtype=torch.long)
        }

        # Add clinical features if using multimodality
        if self.use_multimodal:
            clinical_data = self._get_clinical_features(idx)
            sample['clinical'] = clinical_data

        return sample

    def _get_clinical_features(self, idx: int) -> torch.Tensor:
        """Safely extract clinical features."""
        clinical_cols = [col for col in self.df.columns if col.startswith(('age', 'sex_', 'loc_'))]

        if not clinical_cols:
            return torch.zeros(1, dtype=torch.float)

        features = self.df.iloc[idx][clinical_cols].values.astype(np.float32)
        return torch.tensor(features, dtype=torch.float)


def get_train_transform(img_size: int = 224) -> A.Compose:
    return A.Compose([
        A.RandomResizedCrop(
            size=(img_size, img_size),
            scale=(0.90, 1.00),
            ratio=(0.9, 1.1),
            p=1.0
        ),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=20, border_mode=cv2.BORDER_REFLECT_101, p=0.7),
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
                        max_height=int(0.08 * img_size), max_width=int(0.08 * img_size),
                        min_height=8, min_width=8, fill_value=0, p=0.15),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


def get_val_transform(img_size: int = 224) -> A.Compose:
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


class MelanomaClassifier(nn.Module):

    def __init__(self, model_name: str = 'resnet50', num_classes: int = 2, pretrained: bool = True,
                 use_multimodal: bool = False,
                 clinical_dim: int = 10, fusion_dim: int = 256, dropout: float = 0.3) -> None:
        super().__init__()
        self.use_multimodal = use_multimodal

        if model_name == 'resnet50':
            self.backbone = models.resnet50(
                weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            )
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif model_name == 'efficientnet':
            self.backbone = models.efficientnet_b0(
                weights=EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            )

            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        else:
            raise ValueError(f"Unsupported model_name: {model_name}")

         #Multimodal fusion components
        if self.use_multimodal:
            self.clinical_processor = nn.Sequential(
                nn.Linear(clinical_dim, 128),
                nn.ReLU(),
                nn.BatchNorm1d(128),
                nn.Dropout(dropout),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.BatchNorm1d(64)
            )
            """
            self.clinical_processor = nn.Sequential(
                nn.Linear(clinical_dim, 32),  # Reduce from 128
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(32, 32),  # Simpler
                nn.ReLU()
            )
             """

            # Combined classifier for multimodal
            # Much simpler classifier
            combined_features = num_features + 64
            self.classifier = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(combined_features, 512),
                nn.ReLU(),
                nn.BatchNorm1d(512),
                nn.Dropout(0.2),
                nn.Linear(512, num_classes)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.BatchNorm1d(512),
                nn.Dropout(0.2),
                nn.Linear(512, num_classes)
            )

    def forward(self, x: torch.Tensor, clinical: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # Extract image features
        features = self.backbone(x)

        # Multimodal fusion
        if self.use_multimodal and clinical is not None:
            clinical_features = self.clinical_processor(clinical)
            combined_features = torch.cat([features, clinical_features], dim=1)
            output = self.classifier(combined_features)
        else:
            # Image-only path
            output = self.classifier(features)

        return output, features


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    for p in model.backbone.parameters():
        p.requires_grad = trainable


def get_param_groups_discriminative(model: nn.Module, config: Config) -> List[Dict[str, Any]]:
    param_groups: List[Dict[str, Any]] = []

    # Classifier / head
    param_groups.append({
        "params": list(model.classifier.parameters()),
        "lr": config.HEAD_LR_FINETUNE,
        "weight_decay": config.WEIGHT_DECAY
    })

    bb = model.backbone
    low = config.BACKBONE_LR_LOW
    mid = config.BACKBONE_LR_MID
    high = config.BACKBONE_LR_HIGH

    if isinstance(bb, models.ResNet):
        tiers = [
            (["conv1", "bn1"], low),
            (["layer1"], low),
            (["layer2"], mid),
            (["layer3"], high),
            (["layer4"], high),
        ]
        for names, lr in tiers:
            params: List[nn.Parameter] = []
            for n in names:
                m = getattr(bb, n)
                params += list(m.parameters())
            param_groups.append({"params": params, "lr": lr, "weight_decay": config.WEIGHT_DECAY})

    elif hasattr(bb, "features"):  # EfficientNet-style
        feat = bb.features
        n = len(feat)
        cut1 = max(1, n // 3)
        cut2 = max(cut1 + 1, (2 * n) // 3)

        early = list(feat[:cut1].parameters())      # low
        middle = list(feat[cut1:cut2].parameters()) # mid
        late = list(feat[cut2:].parameters())       # high

        param_groups += [
            {"params": early,  "lr": low,  "weight_decay": config.WEIGHT_DECAY},
            {"params": middle, "lr": mid,  "weight_decay": config.WEIGHT_DECAY},
            {"params": late,   "lr": high, "weight_decay": config.WEIGHT_DECAY},
        ]
    else:
        param_groups.append({
            "params": [p for p in model.backbone.parameters() if p.requires_grad],
            "lr": mid,
            "weight_decay": config.WEIGHT_DECAY
        })

    return param_groups


def build_optimizer_finetune(model: nn.Module, config: Config) -> optim.Optimizer:
    groups = get_param_groups_discriminative(model, config)
    return optim.AdamW(groups)
######## Added this
def error_analysis(predictions: np.ndarray, labels: np.ndarray, probabilities: np.ndarray) -> Dict[str, Any]:
    """Analyze model errors and confidence."""
    predictions = np.array(predictions)
    labels = np.array(labels)
    probabilities = np.array(probabilities)
    
    # Calculate errors
    errors = predictions != labels
    false_positives = (predictions == 1) & (labels == 0)
    false_negatives = (predictions == 0) & (labels == 1)
    
    # Confidence analysis
    fp_confidences = probabilities[false_positives] if np.any(false_positives) else np.array([0.0])
    fn_confidences = probabilities[false_negatives] if np.any(false_negatives) else np.array([0.0])
    
    return {
        'total_errors': int(errors.sum()),
        'error_rate': float(errors.mean()),
        'false_positives': int(false_positives.sum()),
        'false_negatives': int(false_negatives.sum()),
        'avg_fp_confidence': float(fp_confidences.mean()) if len(fp_confidences) > 0 else 0.0,
        'avg_fn_confidence': float(fn_confidences.mean()) if len(fn_confidences) > 0 else 0.0,
        'fp_confidence_std': float(fp_confidences.std()) if len(fp_confidences) > 0 else 0.0,
        'fn_confidence_std': float(fn_confidences.std()) if len(fn_confidences) > 0 else 0.0
    }

class FocalLoss(nn.Module):

    def __init__(self, alpha: Optional[Union[torch.Tensor, float]] = None,
                 gamma: float = 2.0, reduction: str = 'mean', label_smoothing: float = 0.1) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        #self.eps = 1e-8 is replaced by this:
        self.label_smoothing = label_smoothing

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(inputs, dim=1)
        #targets_one_hot = F.one_hot(targets, num_classes=inputs.size(1)).float()
        # Apply label smoothing if enabled
        if self.label_smoothing > 0:
            n_classes = inputs.size(1)
            targets_one_hot = F.one_hot(targets, num_classes=n_classes).float()
            targets_one_hot = targets_one_hot * (1 - self.label_smoothing) + self.label_smoothing / n_classes
        else:
            targets_one_hot = F.one_hot(targets, num_classes=inputs.size(1)).float()
        
        log_pt = (targets_one_hot * log_probs).sum(dim=1)
        pt = log_pt.exp()

        focal_term = (1 - pt) ** self.gamma

        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                alpha_t = torch.tensor([1 - self.alpha, self.alpha], device=inputs.device)[targets]
            else:
                alpha_t = self.alpha[targets]
            focal_loss = -alpha_t * focal_term * log_pt
        else:
            focal_loss = -focal_term * log_pt

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss



class TemperatureScaler(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.log_temperature = nn.Parameter(torch.zeros(1))

    @property
    def T(self) -> torch.Tensor:
        return self.log_temperature.exp()

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.T


def _nll_criterion(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Negative log-likelihood (cross-entropy) for calibration."""
    return F.cross_entropy(logits, targets)

@torch.no_grad()
def collect_logits_and_labels(
    model: nn.Module,
    loader: DataLoader,
    device: Union[str, torch.device],
    use_multimodal: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    all_logits: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []

    print("Starting calibration collection...")
    for i, batch in enumerate(loader):
        print(f"Processing batch {i+1}/{len(loader)}")

        images = batch['image'].to(device)
        labels = batch['label'].to(device)

        if use_multimodal and 'clinical' in batch:
            clinical_data = batch['clinical'].to(device)
            outputs, _ = model(images, clinical_data)
        else:
            outputs, _ = model(images)

        all_logits.append(outputs.detach().cpu())
        all_labels.append(labels.detach().cpu())

        # Break early for testing
        if i >= 2:  # Only process 3 batches
            print("Early break for debugging")
            break

    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    return logits, labels


def fit_temperature(
    model: nn.Module,
    val_loader: DataLoader,
    device: Union[str, torch.device],
    max_iter: int = 200,
    lr: float = 0.01,
    verbose: bool = True,
    use_multimodal: bool = False
) -> TemperatureScaler:
    logits, labels = collect_logits_and_labels(model, val_loader, device, use_multimodal)
    scaler = TemperatureScaler().to(device)
    optimizer = torch.optim.LBFGS(scaler.parameters(), lr=0.25, max_iter=50, line_search_fn='strong_wolfe')

    logits = logits.to(device)
    labels = labels.to(device)

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = _nll_criterion(scaler(logits), labels)
        loss.backward()
        return loss

    try:
        optimizer.step(closure)
    except Exception:
        opt2 = torch.optim.Adam([scaler.log_temperature], lr=lr)
        for _ in range(max_iter):
            opt2.zero_grad()
            loss = _nll_criterion(scaler(logits), labels)
            loss.backward()
            opt2.step()

    if verbose:
        with torch.no_grad():
            before = _nll_criterion(logits, labels).item()
            after = _nll_criterion(scaler(logits), labels).item()
            print(f"Temperature learned: T={scaler.T.item():.4f} | NLL: {before:.4f} -> {after:.4f}")
    return scaler
###added these 2 after overfitting
def mixup_data(x, y, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def apply_temperature(
    logits: torch.Tensor,
    scaler: Optional[TemperatureScaler],
    device: Union[str, torch.device]
) -> torch.Tensor:
   
    if scaler is None:
        return logits
    return scaler(logits.to(device))


def compute_ece(probs: ArrayLike, labels: ArrayLike, n_bins: int = 15) -> float:
    
    probs = np.asarray(probs)
    labels = np.asarray(labels)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    binids = np.digitize(probs, bins[1:-1], right=True)

    ece = 0.0
    for b in range(n_bins):
        mask = binids == b
        if not np.any(mask):
            continue
        conf = probs[mask].mean()
        acc = labels[mask].mean()
        w = mask.mean()
        ece += w * abs(acc - conf)
    return float(ece)


def plot_reliability_diagram(
    labels: ArrayLike,
    probs: ArrayLike,
    save_path: Union[str, os.PathLike],
    n_bins: int = 15
) -> None:
    labels = np.asarray(labels)
    probs = np.asarray(probs)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    binids = np.digitize(probs, bins[1:-1], right=True)

    bin_acc: List[float] = []
    bin_conf: List[float] = []
    for b in range(n_bins):
        mask = binids == b
        if not np.any(mask):
            bin_acc.append(0.0)
            bin_conf.append((bins[b] + bins[b + 1]) / 2.0)
        else:
            bin_acc.append(float(labels[mask].mean()))
            bin_conf.append(float(probs[mask].mean()))

    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], linestyle='--', linewidth=2)
    plt.bar(bin_conf, np.array(bin_acc) - np.array(bin_conf),
            width=1.0 / n_bins, bottom=bin_conf, align='center', alpha=0.7)
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.title('Reliability Diagram')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: Union[str, torch.device],
    use_multimodal: bool = False,
    gradient_clip: float = 1.0
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc='Training')

    for batch in pbar:
        images = batch['image'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        # Handle multimodal vs image-only forward pass
        if use_multimodal and 'clinical' in batch:
            clinical_data = batch['clinical'].to(device)
            outputs, _ = model(images, clinical_data)
        else:
            outputs, _ = model(images)

        loss = criterion(outputs, labels)

        loss.backward()
        if gradient_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix({
            'loss': running_loss / max(1, total),
            'acc': 100 * correct / max(1, total)
        })

    return running_loss / max(1, total), correct / max(1, total)


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: Union[str, torch.device],
    use_multimodal: bool = False
) -> Tuple[float, float, float]:
    model.eval()
    running_loss = 0.0
    all_labels: List[int] = []
    all_preds: List[int] = []
    all_probs: List[float] = [] 

    with torch.no_grad():
        for batch in tqdm(loader, desc='Validation'):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
             # Handle multimodal vs image-only forward pass
            if use_multimodal and 'clinical' in batch:
                clinical_data = batch['clinical'].to(device)
                outputs, _ = model(images, clinical_data)
            else:
                outputs, _ = model(images)
            probs = torch.softmax(outputs, dim=1)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)

            _, predicted = outputs.max(1)
            all_labels.extend(labels.cpu().numpy().tolist())
            all_preds.extend(predicted.cpu().numpy().tolist())
            all_probs.extend(probs[:, 1].cpu().numpy().tolist())

    all_preds_np = np.array(all_preds)
    all_labels_np = np.array(all_labels)
    all_probs_np = np.array(all_probs)  # Now it's defined

    cm = confusion_matrix(all_labels_np, all_preds_np)
    report = classification_report(all_labels_np, all_preds_np, output_dict=True, zero_division=0)
    
    avg_loss = running_loss / len(loader.dataset)
    f1 = f1_score(all_labels_np, all_preds_np, zero_division=0)
    accuracy = (all_preds_np == all_labels_np).mean().item()
    
    return avg_loss, accuracy, f1, report


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: Union[str, torch.device],
    scaler: Optional[TemperatureScaler] = None,
    threshold: Optional[float] = None,
    use_multimodal: bool = False
) -> Dict[str, Any]:
    model.eval()
    all_preds: List[int] = []
    all_labels: List[int] = []
    all_probs: List[float] = []

    with torch.no_grad():
        for batch in tqdm(loader, desc='Evaluating'):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)

            # Handle multimodal vs image-only forward pass
            if use_multimodal and 'clinical' in batch:
                clinical_data = batch['clinical'].to(device)
                logits, _ = model(images, clinical_data)
            else:
                logits, _ = model(images)

            if scaler is not None:
                logits = scaler(logits)

            probs = torch.softmax(logits, dim=1)
            if threshold is None:
                _, preds = probs.max(1)
            else:
                preds = (probs[:, 1] >= float(threshold)).long()

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
            all_probs.extend(probs[:, 1].cpu().numpy().tolist())

    all_preds_np = np.array(all_preds)
    all_labels_np = np.array(all_labels)
    all_probs_np = np.array(all_probs)

    accuracy = accuracy_score(all_labels_np, all_preds_np)
    precision = precision_score(all_labels_np, all_preds_np, zero_division=0)
    recall = recall_score(all_labels_np, all_preds_np, zero_division=0)
    f1 = f1_score(all_labels_np, all_preds_np, zero_division=0)
    roc_auc = roc_auc_score(all_labels_np, all_probs_np)
    cm = confusion_matrix(all_labels_np, all_preds_np)
    ece = compute_ece(all_probs_np, all_labels_np)

    brier = brier_score_loss(all_labels_np, all_probs_np)
    nll = log_loss(all_labels_np, np.vstack([1 - all_probs_np, all_probs_np]).T, labels=[0, 1])

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'ece': ece,
        'brier': brier,
        'nll': nll,
        'confusion_matrix_meta': cm,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs
    }


def best_threshold_for_f1(labels: ArrayLike, probs: ArrayLike) -> Tuple[float, float]:
    labels = np.asarray(labels)
    probs = np.asarray(probs)
    prec, rec, thr = precision_recall_curve(labels, probs)
    f1 = 2 * prec[1:] * rec[1:] / (prec[1:] + rec[1:] + 1e-12)
    idx = int(np.argmax(f1))
    return float(thr[idx]), float(f1[idx])


def youden_j_threshold(labels: ArrayLike, probs: ArrayLike) -> float:
    fpr, tpr, thr = roc_curve(labels, probs)
    j = tpr - fpr
    idx = int(np.argmax(j))
    return float(thr[idx])


def plot_training_history(history: Dict[str, List[float]], save_path: Union[str, os.PathLike]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)

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


def plot_confusion_matrix(cm: np.ndarray, save_path: Union[str, os.PathLike]) -> None:
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Benign', 'Melanoma'],
                yticklabels=['Benign', 'Melanoma'])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix Metadata')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_roc_curve(
    labels: ArrayLike,
    probs: ArrayLike,
    save_path: Union[str, os.PathLike]
) -> None:
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
##### Here is added additional functions
# ENHANCED TRAINING UTILITIES (FROM RESNET50)
def get_layer_groups(model: nn.Module) -> List[List[nn.Parameter]]:
    """Get parameter groups for discriminative fine-tuning."""
    groups = []

    # Layer 4 (deepest)
    groups.append(list(model.backbone.layer4.parameters()))
    # Layer 3
    groups.append(list(model.backbone.layer3.parameters()))
    # Layer 2
    groups.append(list(model.backbone.layer2.parameters()))
    # Layer 1
    groups.append(list(model.backbone.layer1.parameters()))
    # Conv1 + BN
    groups.append(list(model.backbone.conv1.parameters()) + list(model.backbone.bn1.parameters()))

    return groups

def bootstrap_ci(metric_values: np.ndarray, n_samples: int = 1000, ci: float = 0.95) -> Tuple[float, float, float]:
    """Compute bootstrap confidence interval."""
    bootstrapped = []
    for _ in range(n_samples):
        indices = np.random.choice(len(metric_values), size=len(metric_values), replace=True)
        bootstrapped.append(np.mean(metric_values[indices]))

    bootstrapped = np.array(bootstrapped)
    mean = np.mean(bootstrapped)
    alpha = 1 - ci
    lower = np.percentile(bootstrapped, alpha/2 * 100)
    upper = np.percentile(bootstrapped, (1 - alpha/2) * 100)

    return mean, lower, upper

def per_class_metrics(predictions: np.ndarray, labels: np.ndarray, probabilities: np.ndarray) -> Dict[str, Any]:
    """Compute per-class metrics."""
    predictions = np.array(predictions)
    labels = np.array(labels)
    probabilities = np.array(probabilities)
   
    metrics_by_class = {}
    class_names = {0: 'Benign', 1: 'Melanoma'}

    for class_idx in [0, 1]:
        mask = labels == class_idx
        if mask.sum() == 0:
            continue

        class_preds = predictions[mask]
        class_labels = labels[mask]
        class_probs = probabilities[mask]

        metrics_by_class[class_names[class_idx]] = {
            'count': mask.sum(),
            'precision': precision_score(class_labels, class_preds, pos_label=class_idx, zero_division=0),
            'recall': recall_score(class_labels, class_preds, pos_label=class_idx, zero_division=0),
            'f1': f1_score(class_labels, class_preds, pos_label=class_idx, zero_division=0),
            'auc': roc_auc_score((class_labels == class_idx).astype(int), class_probs) if len(np.unique(class_labels)) > 1 else np.nan
        }

    return metrics_by_class

def clinical_validation(metrics_by_class: Dict[str, Any], min_f1: float = 0.90) -> Dict[str, bool]:
    """Check if models meet clinical performance requirements."""
    validation_status = {}

    if 'Melanoma' in metrics_by_class:
        melanoma_f1 = metrics_by_class['Melanoma']['f1']
        passed = melanoma_f1 >= min_f1
        validation_status['Melanoma F1 >= 0.90'] = passed
        print(f"\n{'='*70}")
        print(f"CLINICAL VALIDATION REPORT")
        print(f"{'='*70}")
        print(f"Melanoma F1 Score: {melanoma_f1:.4f}")
        print(f"Requirement: >= {min_f1}")
        print(f"Status: {'✓ PASSED' if passed else '✗ FAILED'}")

    return validation_status

def build_optimizer_warmup(model: nn.Module, config: Config) -> optim.Optimizer:
    """Build optimizer for warmup (head only)."""
    return optim.AdamW(model.classifier.parameters(), lr=config.HEAD_LR, weight_decay=config.WEIGHT_DECAY)

def build_optimizer_finetune(model: nn.Module, config: Config) -> optim.Optimizer:
    """Build optimizer with discriminative learning rates."""
    param_groups = [
        {'params': model.classifier.parameters(), 'lr': config.HEAD_LR},
    ]

    layer_groups = get_layer_groups(model)
    lrs = [config.BACKBONE_LR_HIGH, config.BACKBONE_LR_HIGH, config.BACKBONE_LR_MID,
           config.BACKBONE_LR_MID, config.BACKBONE_LR_LOW]

    for params, lr in zip(layer_groups, lrs):
        param_groups.append({'params': params, 'lr': lr})

    return optim.AdamW(param_groups, weight_decay=config.WEIGHT_DECAY)

def freeze_backbone(model: nn.Module):
    """Freeze backbone parameters."""
    for param in model.backbone.parameters():
        param.requires_grad = False

def unfreeze_backbone(model: nn.Module):
    """Unfreeze backbone parameters."""
    for param in model.backbone.parameters():
        param.requires_grad = True

def cosine_annealing(epoch: int, max_epochs: int, base_lr: float) -> float:
    """Cosine annealing schedule."""
    return base_lr * 0.5 * (1 + np.cos(np.pi * epoch / max_epochs))

def warmup_lr(epoch: int, warmup_epochs: int, base_lr: float) -> float:
    """Warmup learning rate schedule."""
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / warmup_epochs
    return base_lr

def main(use_ham10000: bool = True, use_isic: bool = False, use_multimodal: bool = True) -> Tuple[nn.Module, Dict[str, Any], Dict[str, List[float]]]:
    config = Config()
    config.USE_MULTIMODAL = use_multimodal  # Override from function argument

    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    print('=' * 70)
    print("Fine tuned metadata MELANOMA CLASSIFICATION - TRAINING PIPELINE")
    print('=' * 70)
    print(f"Device: {config.DEVICE}")
    print(f"Model: {config.MODEL_NAME}")
    print(f"Image size: {config.IMG_SIZE}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Max epochs: {config.MAX_EPOCHS}")
    print(f"Warmup epochs: {config.WARMUP_EPOCHS}")
    print(f"Multimodal: {config.USE_MULTIMODAL}")
    print(f"Head LR: {config.HEAD_LR}")
    print(f"Dropout: {config.DROPOUT}")
    print(f"Learning rate: {config.LEARNING_RATE}")
    print("=" * 70)

    if use_ham10000:
        df = load_ham10000_data(config.HAM10000_BASE)
    elif use_isic:
        # Placeholder: function not defined in this file.
        df = load_isic_data(config.ISIC_BASE)  # type: ignore[name-defined]
        df = df[df['split'] == 'train'].reset_index(drop=True)
    else:
        raise ValueError("Must specify either HAM10000 or ISIC dataset")

    print("\nSplitting dataset...")
    train_df, temp_df = train_test_split(
        df, test_size=config.TEST_SIZE + config.VAL_SIZE,
        random_state=config.RANDOM_STATE, stratify=df['binary_label']
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=config.TEST_SIZE / (config.TEST_SIZE + config.VAL_SIZE),
        random_state=config.RANDOM_STATE, stratify=temp_df['binary_label']
    )
    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    # FIX: Create temporary dataset to calculate clinical dimension BEFORE model creation
    clinical_dim = 0
    if config.USE_MULTIMODAL:
        # Create a temporary dataset to preprocess clinical data and get the actual dimension
        temp_dataset = MelanomaDataset(train_df, transform=None, use_multimodal=True)
        # Get clinical features from first sample to determine dimension
        sample = temp_dataset[0]
        if 'clinical' in sample:
            clinical_dim = sample['clinical'].shape[0]
            print(f"Clinical features detected: {clinical_dim} dimensions")
        else:
            print("Warning: No clinical features found, falling back to image-only")
            config.USE_MULTIMODAL = False
    else:
        print("Using image-only model")

    train_dataset = MelanomaDataset(train_df, get_train_transform(config.IMG_SIZE), use_multimodal=config.USE_MULTIMODAL)
    val_dataset = MelanomaDataset(val_df, get_val_transform(config.IMG_SIZE), use_multimodal=config.USE_MULTIMODAL)
    test_dataset = MelanomaDataset(test_df, get_val_transform(config.IMG_SIZE), use_multimodal=config.USE_MULTIMODAL)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE,
                              shuffle=True, num_workers=config.NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE,
                            shuffle=False, num_workers=config.NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE,
                             shuffle=False, num_workers=config.NUM_WORKERS)

    class_counts = train_df['binary_label'].value_counts()
    total = len(train_df)
    class_weights = {
        #changed to be not overly sensitive to melanoma, overfitting in minority
        0: 1.0,  # or sqrt(total/class_counts[0])
        1: min(2.0, total / (2 * class_counts[1])) 
    }
    weights = torch.FloatTensor([class_weights[0], class_weights[1]]).to(config.DEVICE)

    print(f"\nClass weights: {class_weights}")

    print(f"\nCreating enhanced metadata {config.MODEL_NAME} model...")
    # Calculate actual clinical dimension from dataset
    model = MelanomaClassifier(config.MODEL_NAME, config.NUM_CLASSES, config.PRETRAINED, use_multimodal=config.USE_MULTIMODAL, clinical_dim=clinical_dim, fusion_dim=config.FUSION_HIDDEN_DIM, dropout=config.DROPOUT)
    model = model.to(config.DEVICE)

    criterion: nn.Module = FocalLoss(alpha=weights, gamma=1.8, label_smoothing=config.LABEL_SMOOTHING)
    #gamma changed from 2 to 1.8

        # ENHANCED TRAINING PIPELINE (FROM RESNET50)
    print("\nStarting enhanced training with progressive unfreezing...")
    print("=" * 70)

    # Warmup phase
    freeze_backbone(model)
    optimizer = build_optimizer_warmup(model, config)

    best_val_f1 = 0.0
    patience_counter = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}

    print(f"\n--- WARMUP PHASE (Epochs 1-{config.WARMUP_EPOCHS}) ---")
    for epoch in range(config.WARMUP_EPOCHS):
        lr = warmup_lr(epoch, config.WARMUP_EPOCHS, config.HEAD_LR)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, config.DEVICE, 
            config.USE_MULTIMODAL, config.GRADIENT_CLIP
        )
        val_loss, val_acc, val_f1, val_report = validate(model, val_loader, criterion, config.DEVICE, config.USE_MULTIMODAL)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)

        print(f"Ep {epoch+1}: TrLoss={train_loss:.4f} TrAcc={train_acc:.4f} VlLoss={val_loss:.4f} VlAcc={val_acc:.4f} VlF1={val_f1:.4f} LR={lr:.6f}")

    # Fine-tuning phase
    print(f"\n--- FINE-TUNING PHASE (Progressive unfreezing) ---")
    unfreeze_backbone(model)
    optimizer = build_optimizer_finetune(model, config)

    for epoch in range(config.WARMUP_EPOCHS, config.MAX_EPOCHS):
        # Cosine annealing for each param group
        for param_group in optimizer.param_groups:
            base_lr = param_group.get('original_lr', param_group['lr'])
            param_group['lr'] = cosine_annealing(
                epoch - config.WARMUP_EPOCHS,
                config.MAX_EPOCHS - config.WARMUP_EPOCHS,
                base_lr
            )

        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, config.DEVICE, 
            config.USE_MULTIMODAL, config.GRADIENT_CLIP
        )
        val_loss, val_acc, val_f1, val_report = validate(model, val_loader, criterion, config.DEVICE, config.USE_MULTIMODAL)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)

        if epoch % 5 == 0:
            print(f"Ep {epoch+1}: TrLoss={train_loss:.4f} TrAcc={train_acc:.4f} VlLoss={val_loss:.4f} VlAcc={val_acc:.4f} VlF1={val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
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

        if patience_counter >= config.EARLY_STOPPING_PATIENCE:
            print(f"\n✓ Early stopping at epoch {epoch+1}")
            break

    plot_training_history(history, os.path.join(config.RESULTS_DIR, 'training_history.png'))

    print("\n" + "=" * 70)
    print("CALIBRATING AND EVALUATING")
    print("=" * 70)

    checkpoint = torch.load(os.path.join(config.CHECKPOINT_DIR, 'best_model.pth'), map_location=config.DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(config.DEVICE)

    scaler = fit_temperature(model, val_loader, config.DEVICE, verbose=True, use_multimodal=config.USE_MULTIMODAL)

    print("\nSelecting threshold on VALIDATION (calibrated)...")
    val_metrics_cal = evaluate_model(model, val_loader, config.DEVICE, scaler=scaler, threshold=None, use_multimodal=config.USE_MULTIMODAL)
    t_f1, val_f1_at_t = best_threshold_for_f1(val_metrics_cal['labels'], val_metrics_cal['probabilities'])
    print(f"Chosen threshold for F1: t={t_f1:.4f} (val F1 @ t = {val_f1_at_t:.4f})")

    print("\nEvaluating UNCALIBRATED on test...")
    metrics_raw = evaluate_model(model, test_loader, config.DEVICE, scaler=None, threshold=None, use_multimodal=config.USE_MULTIMODAL)

    print("\nEvaluating CALIBRATED (default argmax) on test...")
    metrics_cal = evaluate_model(model, test_loader, config.DEVICE, scaler=scaler, threshold=None, use_multimodal=config.USE_MULTIMODAL)

    print("\nEvaluating CALIBRATED + THRESHOLD-TUNED on test...")
    metrics_cal_t = evaluate_model(model, test_loader, config.DEVICE, scaler=scaler, threshold=t_f1, use_multimodal=config.USE_MULTIMODAL)

    def _pretty(m: Dict[str, Any]) -> str:
        return (f"Acc {m['accuracy']:.4f} | Prec {m['precision']:.4f} | "
                f"Rec {m['recall']:.4f} | F1 {m['f1']:.4f} | "
                f"AUC {m['roc_auc']:.4f} | ECE {m['ece']:.4f} | "
                f"Brier {m['brier']:.4f} | NLL {m['nll']:.4f}")

    print("\nTest (uncalibrated, argmax):")
    print(_pretty(metrics_raw))
    print("\nTest (calibrated, argmax):")
    print(_pretty(metrics_cal))
    print("\nTest (calibrated, tuned t):")
    print(_pretty(metrics_cal_t))

    plot_reliability_diagram(metrics_raw['labels'], metrics_raw['probabilities'],
                             os.path.join(config.RESULTS_DIR, 'reliability_uncalibrated.png'))
    plot_reliability_diagram(metrics_cal['labels'], metrics_cal['probabilities'],
                             os.path.join(config.RESULTS_DIR, 'reliability_calibrated.png'))

    plot_confusion_matrix(metrics_cal['confusion_matrix_meta'],
                          os.path.join(config.RESULTS_DIR, 'confusion_matrix_calibrated.png'))
    plot_roc_curve(metrics_cal['labels'], metrics_cal['probabilities'],
                   os.path.join(config.RESULTS_DIR, 'roc_curve_calibrated.png'))

    print(f"\nResults saved to: {config.RESULTS_DIR}")
    print(f"Temperature T: {scaler.T.item():.4f}")
    print("Training + calibration complete!")
        # COMPREHENSIVE ANALYSIS (FROM RESNET50)
    print("\n" + "=" * 70)
    print("COMPREHENSIVE MODEL ANALYSIS")
    print("=" * 70)

    # Per-class metrics
    class_metrics = per_class_metrics(
        metrics_cal['predictions'], 
        metrics_cal['labels'], 
        metrics_cal['probabilities']
    )

    print(f"\nPER-CLASS PERFORMANCE:")
    for class_name, metrics in class_metrics.items():
        print(f"\n{class_name}:")
        print(f"  Samples: {metrics['count']}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1 Score: {metrics['f1']:.4f}")
        if not np.isnan(metrics['auc']):
            print(f"  AUC: {metrics['auc']:.4f}")

    # Clinical validation
    validation_status = clinical_validation(class_metrics, config.MIN_MELANOMA_F1)

    # Error analysis
    error_report = error_analysis(
        metrics_cal['predictions'],
        metrics_cal['labels'], 
        metrics_cal['probabilities']
    )

    print(f"\nERROR ANALYSIS:")
    print(f"Total Errors: {error_report['total_errors']}")
    print(f"Error Rate: {error_report['error_rate']:.4f}")
    print(f"False Positives: {error_report['false_positives']} (avg confidence: {error_report['avg_fp_confidence']:.4f})")
    print(f"False Negatives: {error_report['false_negatives']} (avg confidence: {error_report['avg_fn_confidence']:.4f})")

    print(f"\n✓ Enhanced fine-tuning completed successfully!")

    return model, metrics_cal, history


if __name__ == "__main__":
    # Run training
    try:
        # Run training with multimodal disabled first to test
        print("Starting training with multimodal features...")
        model, metrics, history = main(use_ham10000=True, use_isic=False, use_multimodal=True)
        print("Training completed successfully!")
    except Exception as e:
        print(f"Error during training: {e}")
        print("Falling back to image-only model...")
        try:
            model, metrics, history = main(use_ham10000=True, use_isic=False, use_multimodal=False)
            print("Image-only training completed successfully!")
        except Exception as e2:
            print(f"Image-only training also failed: {e2}")
