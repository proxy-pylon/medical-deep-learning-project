import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
from typing import Dict, Any, Tuple
from tqdm import tqdm


def evaluate_model(model: nn.Module, dataloader: DataLoader, device: str, 
                   threshold: float = 0.5) -> Dict[str, Any]:
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Evaluating')
        for batch in pbar:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
    
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_preds = (all_probs >= threshold).astype(int)
    
    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, zero_division=0),
        'recall': recall_score(all_labels, all_preds, zero_division=0),
        'f1': f1_score(all_labels, all_preds, zero_division=0),
        'roc_auc': roc_auc_score(all_labels, all_probs),
        'confusion_matrix': confusion_matrix(all_labels, all_preds),
        'labels': all_labels,
        'predictions': all_preds,
        'probabilities': all_probs
    }
    
    return metrics


def best_threshold_for_f1(labels: np.ndarray, probs: np.ndarray, 
                          num_thresholds: int = 100) -> Tuple[float, float]:
    thresholds = np.linspace(0, 1, num_thresholds)
    best_f1 = 0.0
    best_t = 0.5
    
    for t in thresholds:
        preds = (probs >= t).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    
    return best_t, best_f1