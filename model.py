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

# Warnings
warnings.filterwarnings('ignore')


class Config:
    """Container for training, model, and path configuration.

    Attributes
    ----------
    HAM10000_BASE : str
        Base directory for the HAM10000 dataset.
    ISIC_BASE : str
        Base directory for the ISIC dataset.
    OUTPUT_DIR : str
        Base output directory.
    CHECKPOINT_DIR : str
        Directory for checkpoints.
    RESULTS_DIR : str
        Directory for results and plots.
    MODEL_NAME : str
        Backbone model identifier. One of {'resnet50', 'efficientnet', 'vgg16'}.
    IMG_SIZE : int
        Input image size (square).
    NUM_CLASSES : int
        Number of output classes.
    PRETRAINED : bool
        Whether to use ImageNet pretrained weights.
    BATCH_SIZE : int
        Training batch size.
    NUM_EPOCHS : int
        Maximum number of epochs.
    LEARNING_RATE : float
        Base learning rate (not used directly when discriminative LRs are applied).
    WEIGHT_DECAY : float
        Weight decay for optimizers.
    EARLY_STOPPING_PATIENCE : int
        Patience for early stopping.
    FREEZE_EPOCHS : int
        Number of warmup epochs training the head only.
    HEAD_LR_WARMUP : float
        Learning rate for head during warmup.
    HEAD_LR_FINETUNE : float
        Learning rate for head during fine-tuning.
    BACKBONE_LR_LOW : float
        LR for earliest backbone layers.
    BACKBONE_LR_MID : float
        LR for mid backbone layers.
    BACKBONE_LR_HIGH : float
        LR for deepest backbone layers.
    TEST_SIZE : float
        Proportion for test split.
    VAL_SIZE : float
        Proportion for validation split (from the non-train part).
    RANDOM_STATE : int
        Random seed for splitting.
    USE_MIXUP : bool
        Placeholder flag for mixup usage.
    USE_CUTMIX : bool
        Placeholder flag for cutmix usage.
    DEVICE : str
        'cuda' if available else 'cpu'.
    NUM_WORKERS : int
        DataLoader workers.
    """

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
    BATCH_SIZE = 32
    NUM_EPOCHS = 250
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4
    EARLY_STOPPING_PATIENCE = 30

    # Warmup / fine-tune
    FREEZE_EPOCHS = 3
    HEAD_LR_WARMUP = 1e-3
    HEAD_LR_FINETUNE = 1e-4

    # Discriminative LRs for backbone
    BACKBONE_LR_LOW = 1e-5
    BACKBONE_LR_MID = 2e-5
    BACKBONE_LR_HIGH = 3e-5

    # Dataset split ratio and seeding
    TEST_SIZE = 0.30
    VAL_SIZE = 0.20
    RANDOM_STATE = 42

    # Augmentation flags
    USE_MIXUP = False
    USE_CUTMIX = False

    # Device
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    NUM_WORKERS = 2


def load_ham10000_data(base_path: Union[str, os.PathLike]) -> pd.DataFrame:
    """Load HAM10000 metadata and resolve image paths.

    Parameters
    ----------
    base_path : str or os.PathLike
        Directory containing `HAM10000_metadata.csv` and image folders.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns including:
        - 'image_id'
        - 'dx' (diagnosis)
        - 'image_path' (resolved path to JPEG)
        - 'binary_label' (1 for melanoma, 0 otherwise)
    """
    print('Loading HAM10000 dataset....')
    base_path = str(base_path)

    metadata_path = os.path.join(base_path, 'HAM10000_metadata.csv')
    df = pd.read_csv(metadata_path)

    def get_image_path(image_id: str) -> Optional[str]:
        """Return path to image file if present in either image folder."""
        part1 = os.path.join(base_path, 'HAM10000_images_part_1', f'{image_id}.jpg')
        part2 = os.path.join(base_path, 'HAM10000_images_part_2', f'{image_id}.jpg')
        if os.path.exists(part1):
            return part1
        if os.path.exists(part2):
            return part2
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
    """PyTorch Dataset for melanoma classification with Albumentations transforms.

    Parameters
    ----------
    dataframe : pd.DataFrame
        DataFrame with at least columns 'image_path' and 'binary_label'.
    transform : Optional[A.Compose]
        Albumentations transform to apply to the loaded image.

    Returns
    -------
    dict
        A sample dict containing:
        - 'image': torch.Tensor of shape [3, H, W]
        - 'label': torch.LongTensor scalar {0, 1}
    """

    def __init__(self, dataframe: pd.DataFrame, transform: Optional[A.Compose] = None) -> None:
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        """Number of samples."""
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Load one sample by index.

        Parameters
        ----------
        idx : int
            Row index.

        Returns
        -------
        dict
            Dict with 'image' and 'label' tensors.
        """
        img_path: str = self.df.loc[idx, 'image_path']
        label: int = int(self.df.loc[idx, 'binary_label'])

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        return {
            'image': image,
            'label': torch.tensor(label, dtype=torch.long)
        }


def get_train_transform(img_size: int = 224) -> A.Compose:
    """Build the training augmentation pipeline.

    Parameters
    ----------
    img_size : int, default=224
        Target square image size.

    Returns
    -------
    A.Compose
        Albumentations composition.
    """
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
    """Build the validation/test preprocessing pipeline.

    Parameters
    ----------
    img_size : int, default=224
        Target square image size.

    Returns
    -------
    A.Compose
        Albumentations composition for eval.
    """
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


class MelanomaClassifier(nn.Module):
    """CNN classifier using a torchvision backbone with a custom head.

    Parameters
    ----------
    model_name : str, default='resnet50'
        One of {'resnet50', 'efficientnet'} currently supported.
    num_classes : int, default=2
        Number of output classes.
    pretrained : bool, default=True
        If True, initialize backbone with pretrained weights.

    Notes
    -----
    Forward returns both logits and the pooled backbone features.
    """

    def __init__(self, model_name: str = 'resnet50', num_classes: int = 2, pretrained: bool = True) -> None:
        super().__init__()
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

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input batch of shape [N, 3, H, W].

        Returns
        -------
        (torch.Tensor, torch.Tensor)
            - logits: [N, C]
            - features: pooled features from the backbone, shape [N, F]
        """
        features = self.backbone(x)
        output = self.classifier(features)
        return output, features


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    """Enable or disable gradient updates for backbone parameters.

    Parameters
    ----------
    model : nn.Module
        Model with attribute `backbone`.
    trainable : bool
        If True, unfreezes the backbone; if False, freezes it.
    """
    for p in model.backbone.parameters():
        p.requires_grad = trainable


def get_param_groups_discriminative(model: nn.Module, config: Config) -> List[Dict[str, Any]]:
    """Create optimizer parameter groups with discriminative learning rates.

    Parameters
    ----------
    model : nn.Module
        Model with `backbone` and `classifier`.
    config : Config
        Hyperparameters containing LR tiers and weight decay.

    Returns
    -------
    list of dict
        Parameter groups consumable by torch optimizer.
    """
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


def build_optimizer_warmup(model: nn.Module, config: Config) -> optim.Optimizer:
    """Create AdamW optimizer for the head-only warmup phase.

    Parameters
    ----------
    model : nn.Module
        Model with a `classifier` module.
    config : Config
        Hyperparameters.

    Returns
    -------
    torch.optim.Optimizer
        AdamW optimizer over head parameters.
    """
    head_params = [p for p in model.classifier.parameters() if p.requires_grad]
    return optim.AdamW(head_params, lr=config.HEAD_LR_WARMUP, weight_decay=config.WEIGHT_DECAY)


def build_optimizer_finetune(model: nn.Module, config: Config) -> optim.Optimizer:
    """Create AdamW optimizer with discriminative LRs for fine-tuning.

    Parameters
    ----------
    model : nn.Module
        Model with `backbone` and `classifier`.
    config : Config
        Hyperparameters.

    Returns
    -------
    torch.optim.Optimizer
        AdamW optimizer over grouped parameters.
    """
    groups = get_param_groups_discriminative(model, config)
    return optim.AdamW(groups)


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance.

    Parameters
    ----------
    alpha : Optional[torch.Tensor or float], default=None
        Class weighting factor. Either scalar or tensor of shape [num_classes].
    gamma : float, default=2.0
        Focusing parameter to down-weight easy examples.
    reduction : {'none', 'mean', 'sum'}, default='mean'
        Reduction mode.

    Returns
    -------
    torch.Tensor
        Reduced loss according to `reduction`.
    """

    def __init__(self, alpha: Optional[Union[torch.Tensor, float]] = None,
                 gamma: float = 2.0, reduction: str = 'mean') -> None:
        super().__init__()
        self.alpha = alpha  # tensor of shape [num_classes] or scalar
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.

        Parameters
        ----------
        inputs : torch.Tensor
            Logits of shape [N, C].
        targets : torch.Tensor
            Integer class labels of shape [N].

        Returns
        -------
        torch.Tensor
            Loss tensor reduced by `reduction`.
        """
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        if self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class TemperatureScaler(nn.Module):
    """Post-hoc calibration via temperature scaling."""

    def __init__(self) -> None:
        super().__init__()
        self.log_temperature = nn.Parameter(torch.zeros(1))

    @property
    def T(self) -> torch.Tensor:
        """Return positive temperature parameter."""
        return self.log_temperature.exp()

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Scale logits by temperature.

        Parameters
        ----------
        logits : torch.Tensor
            Unnormalized model outputs of shape [N, C].

        Returns
        -------
        torch.Tensor
            Scaled logits of shape [N, C].
        """
        return logits / self.T


def _nll_criterion(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Negative log-likelihood (cross-entropy) for calibration."""
    return F.cross_entropy(logits, targets)


@torch.no_grad()
def collect_logits_and_labels(
    model: nn.Module,
    loader: DataLoader,
    device: Union[str, torch.device]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run model over a DataLoader and collect logits and labels.

    Parameters
    ----------
    model : nn.Module
        Trained classifier.
    loader : DataLoader
        DataLoader yielding dicts with 'image' and 'label'.
    device : str or torch.device
        Computation device.

    Returns
    -------
    (torch.Tensor, torch.Tensor)
        - logits: [N, C]
        - labels: [N]
    """
    model.eval()
    all_logits: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []
    for batch in tqdm(loader, desc='Collecting logits for calibration'):
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        outputs, _ = model(images)
        all_logits.append(outputs.detach().cpu())
        all_labels.append(labels.detach().cpu())
    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    return logits, labels


def fit_temperature(
    model: nn.Module,
    val_loader: DataLoader,
    device: Union[str, torch.device],
    max_iter: int = 200,
    lr: float = 0.01,
    verbose: bool = True
) -> TemperatureScaler:
    """Fit a TemperatureScaler on validation data by minimizing NLL.

    Parameters
    ----------
    model : nn.Module
        Classifier to calibrate.
    val_loader : DataLoader
        Validation loader.
    device : str or torch.device
        Device for computation.
    max_iter : int, default=200
        Max iterations for the fallback Adam optimizer.
    lr : float, default=0.01
        LR for fallback Adam.
    verbose : bool, default=True
        Print calibration summary.

    Returns
    -------
    TemperatureScaler
        Fitted temperature scaler.
    """
    logits, labels = collect_logits_and_labels(model, val_loader, device)
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


def apply_temperature(
    logits: torch.Tensor,
    scaler: Optional[TemperatureScaler],
    device: Union[str, torch.device]
) -> torch.Tensor:
    """Apply a fitted temperature scaler to logits if provided.

    Parameters
    ----------
    logits : torch.Tensor
        Raw logits [N, C].
    scaler : Optional[TemperatureScaler]
        Fitted scaler or None.
    device : str or torch.device
        Device.

    Returns
    -------
    torch.Tensor
        Possibly scaled logits [N, C].
    """
    if scaler is None:
        return logits
    return scaler(logits.to(device))


def compute_ece(probs: ArrayLike, labels: ArrayLike, n_bins: int = 15) -> float:
    """Compute Expected Calibration Error (ECE) for binary classification.

    Parameters
    ----------
    probs : ArrayLike
        Predicted probabilities for the positive class, shape [N].
    labels : ArrayLike
        Binary labels {0, 1}, shape [N].
    n_bins : int, default=15
        Number of confidence bins.

    Returns
    -------
    float
        ECE value in [0, 1].
    """
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
    """Plot and save a reliability diagram.

    Parameters
    ----------
    labels : ArrayLike
        Binary ground-truth labels [N].
    probs : ArrayLike
        Predicted positive-class probabilities [N].
    save_path : str or os.PathLike
        Output image path.
    n_bins : int, default=15
        Number of bins.
    """
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
    device: Union[str, torch.device]
) -> Tuple[float, float]:
    """Train the model for one epoch.

    Parameters
    ----------
    model : nn.Module
        Model to train.
    loader : DataLoader
        Training DataLoader.
    criterion : nn.Module
        Loss function.
    optimizer : torch.optim.Optimizer
        Optimizer instance.
    device : str or torch.device
        Device.

    Returns
    -------
    (float, float)
        Tuple of (average_loss, accuracy) where accuracy is in [0, 1].
    """
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
            'loss': running_loss / max(1, total),
            'acc': 100 * correct / max(1, total)
        })

    return running_loss / max(1, total), correct / max(1, total)


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: Union[str, torch.device]
) -> Tuple[float, float, float]:
    """Validate the model and compute loss, accuracy, and F1.

    Parameters
    ----------
    model : nn.Module
        Model to evaluate.
    loader : DataLoader
        Validation DataLoader.
    criterion : nn.Module
        Loss function.
    device : str or torch.device
        Device.

    Returns
    -------
    (float, float, float)
        Tuple (avg_loss, accuracy, f1), accuracy and f1 in [0, 1].
    """
    model.eval()
    running_loss = 0.0
    all_labels: List[int] = []
    all_preds: List[int] = []

    with torch.no_grad():
        for batch in tqdm(loader, desc='Validation'):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            outputs, _ = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)

            _, predicted = outputs.max(1)
            all_labels.extend(labels.cpu().numpy().tolist())
            all_preds.extend(predicted.cpu().numpy().tolist())

    avg_loss = running_loss / len(loader.dataset)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    accuracy = (np.array(all_preds) == np.array(all_labels)).mean().item()
    return avg_loss, accuracy, f1


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: Union[str, torch.device],
    scaler: Optional[TemperatureScaler] = None,
    threshold: Optional[float] = None
) -> Dict[str, Any]:
    """Evaluate model with optional calibration and decision threshold.

    Parameters
    ----------
    model : nn.Module
        Model to evaluate.
    loader : DataLoader
        DataLoader.
    device : str or torch.device
        Device.
    scaler : Optional[TemperatureScaler], default=None
        Temperature scaler for logits.
    threshold : Optional[float], default=None
        If provided, use this threshold on positive-class probability.
        If None, uses argmax over classes.

    Returns
    -------
    dict
        Metrics and raw outputs:
        - 'accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'ece'
        - 'brier', 'nll'
        - 'confusion_matrix' (np.ndarray shape [2,2])
        - 'predictions', 'labels', 'probabilities' (lists)
    """
    model.eval()
    all_preds: List[int] = []
    all_labels: List[int] = []
    all_probs: List[float] = []

    with torch.no_grad():
        for batch in tqdm(loader, desc='Evaluating'):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)

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
        'confusion_matrix': cm,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs
    }


def best_threshold_for_f1(labels: ArrayLike, probs: ArrayLike) -> Tuple[float, float]:
    """Compute the probability threshold that maximizes F1 on validation data.

    Parameters
    ----------
    labels : ArrayLike
        Binary labels {0,1}.
    probs : ArrayLike
        Predicted positive-class probabilities.

    Returns
    -------
    (float, float)
        Tuple (threshold, best_f1).
    """
    labels = np.asarray(labels)
    probs = np.asarray(probs)
    prec, rec, thr = precision_recall_curve(labels, probs)
    f1 = 2 * prec[1:] * rec[1:] / (prec[1:] + rec[1:] + 1e-12)
    idx = int(np.argmax(f1))
    return float(thr[idx]), float(f1[idx])


def youden_j_threshold(labels: ArrayLike, probs: ArrayLike) -> float:
    """Compute Youden's J statistic threshold from the ROC curve.

    Parameters
    ----------
    labels : ArrayLike
        Binary labels {0,1}.
    probs : ArrayLike
        Predicted positive-class probabilities.

    Returns
    -------
    float
        Threshold that maximizes TPR - FPR.
    """
    fpr, tpr, thr = roc_curve(labels, probs)
    j = tpr - fpr
    idx = int(np.argmax(j))
    return float(thr[idx])


def plot_training_history(history: Dict[str, List[float]], save_path: Union[str, os.PathLike]) -> None:
    """Plot training/validation loss and accuracy curves.

    Parameters
    ----------
    history : dict
        History dict with keys: 'train_loss', 'val_loss', 'train_acc', 'val_acc'.
    save_path : str or os.PathLike
        Output path for the image file.
    """
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
    """Plot and save a confusion matrix heatmap.

    Parameters
    ----------
    cm : np.ndarray
        Confusion matrix of shape [2, 2].
    save_path : str or os.PathLike
        Output image path.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Benign', 'Melanoma'],
                yticklabels=['Benign', 'Melanoma'])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_roc_curve(
    labels: ArrayLike,
    probs: ArrayLike,
    save_path: Union[str, os.PathLike]
) -> None:
    """Plot and save the ROC curve.

    Parameters
    ----------
    labels : ArrayLike
        Ground truth binary labels {0, 1}.
    probs : ArrayLike
        Predicted probabilities for the positive class.
    save_path : str or os.PathLike
        Path to save the ROC curve image.
    """
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


def main(use_ham10000: bool = True, use_isic: bool = False) -> Tuple[nn.Module, Dict[str, Any], Dict[str, List[float]]]:
    """Main training, calibration, and evaluation pipeline.

    Parameters
    ----------
    use_ham10000 : bool, default=True
        If True, load HAM10000 dataset.
    use_isic : bool, default=False
        If True, load ISIC dataset (requires `load_isic_data`, not provided here).

    Returns
    -------
    (nn.Module, dict, dict)
        - Trained model
        - Metrics dict for calibrated evaluation on test set
        - Training history dict
    """
    config = Config()
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

    train_dataset = MelanomaDataset(train_df, get_train_transform(config.IMG_SIZE))
    val_dataset = MelanomaDataset(val_df, get_val_transform(config.IMG_SIZE))
    test_dataset = MelanomaDataset(test_df, get_val_transform(config.IMG_SIZE))

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE,
                              shuffle=True, num_workers=config.NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE,
                            shuffle=False, num_workers=config.NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE,
                             shuffle=False, num_workers=config.NUM_WORKERS)

    class_counts = train_df['binary_label'].value_counts()
    total = len(train_df)
    class_weights = {
        0: total / (2 * class_counts[0]),
        1: total / (2 * class_counts[1]) * 1.0
    }
    weights = torch.FloatTensor([class_weights[0], class_weights[1]]).to(config.DEVICE)

    print(f"\nClass weights: {class_weights}")

    print(f"\nCreating {config.MODEL_NAME} model...")
    model = MelanomaClassifier(config.MODEL_NAME, config.NUM_CLASSES, config.PRETRAINED)
    model = model.to(config.DEVICE)

    criterion: nn.Module = FocalLoss(alpha=weights, gamma=2.0)

    set_backbone_trainable(model, trainable=False)
    optimizer: optim.Optimizer = build_optimizer_warmup(model, config)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    current_phase = "warmup"

    print("\nStarting training...")
    print("=" * 70)

    history: Dict[str, List[float]] = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    best_val_f1 = 0.0
    patience_counter = 0

    for epoch in range(config.NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{config.NUM_EPOCHS}")
        print("-" * 70)

        if current_phase == "warmup" and epoch == config.FREEZE_EPOCHS:
            print("\nUnfreezing backbone and switching to discriminative LRs...")
            set_backbone_trainable(model, trainable=True)
            optimizer = build_optimizer_finetune(model, config)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
            current_phase = "finetune"

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, config.DEVICE)
        val_loss, val_acc, val_f1 = validate(model, val_loader, criterion, config.DEVICE)

        scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        print(f"LR: {optimizer.param_groups[0]['lr']:.6f}")

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
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= config.EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping at epoch {epoch + 1}")
            break

    plot_training_history(history, os.path.join(config.RESULTS_DIR, 'training_history.png'))

    print("\n" + "=" * 70)
    print("CALIBRATING AND EVALUATING")
    print("=" * 70)

    checkpoint = torch.load(os.path.join(config.CHECKPOINT_DIR, 'best_model.pth'), map_location=config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(config.DEVICE)

    scaler = fit_temperature(model, val_loader, config.DEVICE, verbose=True)

    print("\nSelecting threshold on VALIDATION (calibrated)...")
    val_metrics_cal = evaluate_model(model, val_loader, config.DEVICE, scaler=scaler, threshold=None)
    t_f1, val_f1_at_t = best_threshold_for_f1(val_metrics_cal['labels'], val_metrics_cal['probabilities'])
    print(f"Chosen threshold for F1: t={t_f1:.4f} (val F1 @ t = {val_f1_at_t:.4f})")

    print("\nEvaluating UNCALIBRATED on test...")
    metrics_raw = evaluate_model(model, test_loader, config.DEVICE, scaler=None, threshold=None)

    print("\nEvaluating CALIBRATED (default argmax) on test...")
    metrics_cal = evaluate_model(model, test_loader, config.DEVICE, scaler=scaler, threshold=None)

    print("\nEvaluating CALIBRATED + THRESHOLD-TUNED on test...")
    metrics_cal_t = evaluate_model(model, test_loader, config.DEVICE, scaler=scaler, threshold=t_f1)

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

    plot_confusion_matrix(metrics_cal['confusion_matrix'],
                          os.path.join(config.RESULTS_DIR, 'confusion_matrix_calibrated.png'))
    plot_roc_curve(metrics_cal['labels'], metrics_cal['probabilities'],
                   os.path.join(config.RESULTS_DIR, 'roc_curve_calibrated.png'))

    print(f"\nResults saved to: {config.RESULTS_DIR}")
    print(f"Temperature T: {scaler.T.item():.4f}")
    print("Training + calibration complete!")

    return model, metrics_cal, history


if __name__ == "__main__":
    # Run training
    model, metrics, history = main(use_ham10000=True, use_isic=True)

