import os
import sys
import torch
import argparse
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.config import Config
from data import MelanomaDataset, load_ham10000_data
from models import MelanomaClassifier
from training import get_val_transform
from evaluation import (
    evaluate_model, best_threshold_for_f1,
    plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
)


def main():
    parser = argparse.ArgumentParser(description='Evaluate melanoma classifier')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--model', type=str, default='resnet50',
                       choices=['resnet50', 'senet', 'efficientnet', 'convnext', 'vgg16'],
                       help='Model architecture')
    parser.add_argument('--threshold', type=float, default=None,
                       help='Classification threshold (if None, finds optimal)')
    args = parser.parse_args()
    
    config = Config()
    config.MODEL_NAME = args.model
    
    print('=' * 70)
    print("MELANOMA CLASSIFICATION - EVALUATION")
    print('=' * 70)
    print(f"Model: {config.MODEL_NAME}")
    print(f"Checkpoint: {args.checkpoint}")
    print("=" * 70)
    
    df = load_ham10000_data(config.HAM10000_BASE)
    
    train_df, temp_df = train_test_split(
        df, test_size=config.TEST_SIZE + config.VAL_SIZE,
        random_state=config.RANDOM_STATE, stratify=df['binary_label']
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=config.TEST_SIZE / (config.TEST_SIZE + config.VAL_SIZE),
        random_state=config.RANDOM_STATE, stratify=temp_df['binary_label']
    )
    
    val_dataset = MelanomaDataset(val_df, get_val_transform(config.IMG_SIZE))
    test_dataset = MelanomaDataset(test_df, get_val_transform(config.IMG_SIZE))
    
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE,
                           shuffle=False, num_workers=config.NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE,
                            shuffle=False, num_workers=config.NUM_WORKERS)
    
    print(f"\nLoading model...")
    model = MelanomaClassifier(config.MODEL_NAME, config.NUM_CLASSES, pretrained=False)
    checkpoint = torch.load(args.checkpoint, map_location=config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(config.DEVICE)
    
    if args.threshold is None:
        print("\nFinding optimal threshold on validation set...")
        val_metrics = evaluate_model(model, val_loader, config.DEVICE, threshold=0.5)
        threshold, val_f1 = best_threshold_for_f1(
            val_metrics['labels'], val_metrics['probabilities']
        )
        print(f"Optimal threshold: {threshold:.4f} (Val F1: {val_f1:.4f})")
    else:
        threshold = args.threshold
        print(f"\nUsing provided threshold: {threshold:.4f}")
    
    print("\nEvaluating on test set...")
    metrics = evaluate_model(model, test_loader, config.DEVICE, threshold=threshold)
    
    print("\nTest Results:")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")
    print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(f"\nConfusion Matrix:")
    print(metrics['confusion_matrix'])
    
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    plot_confusion_matrix(metrics['confusion_matrix'],
                         os.path.join(config.RESULTS_DIR, 'eval_confusion_matrix.png'))
    plot_roc_curve(metrics['labels'], metrics['probabilities'],
                  os.path.join(config.RESULTS_DIR, 'eval_roc_curve.png'))
    plot_precision_recall_curve(metrics['labels'], metrics['probabilities'],
                               os.path.join(config.RESULTS_DIR, 'eval_pr_curve.png'))
    
    print(f"\nPlots saved to: {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()