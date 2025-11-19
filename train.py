import os
import sys
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.config import Config
from data import MelanomaDataset, load_ham10000_data
from models import MelanomaClassifier, FocalLoss
from training import train_model, get_train_transform, get_val_transform
from evaluation import (
    evaluate_model, best_threshold_for_f1,
    plot_training_history, plot_confusion_matrix,
    plot_roc_curve, plot_precision_recall_curve
)


def main():
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
    
    df = load_ham10000_data(config.HAM10000_BASE)
    
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
    
    criterion = FocalLoss(alpha=weights, gamma=2.0)
    
    print("\nStarting training...")
    print("=" * 70)
    
    save_path = os.path.join(config.CHECKPOINT_DIR, 'best_model.pth')
    history = train_model(model, train_loader, val_loader, criterion, config, save_path)
    
    plot_training_history(history, os.path.join(config.RESULTS_DIR, 'training_history.png'))
    
    print("\n" + "=" * 70)
    print("EVALUATING ON TEST SET")
    print("=" * 70)
    
    checkpoint = torch.load(save_path, map_location=config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(config.DEVICE)
    
    print("\nFinding optimal threshold on validation set...")
    val_metrics = evaluate_model(model, val_loader, config.DEVICE, threshold=0.5)
    optimal_threshold, val_f1_at_t = best_threshold_for_f1(
        val_metrics['labels'], val_metrics['probabilities']
    )
    print(f"Optimal threshold: {optimal_threshold:.4f} (Val F1: {val_f1_at_t:.4f})")
    
    print("\nEvaluating with default threshold (0.5)...")
    metrics_default = evaluate_model(model, test_loader, config.DEVICE, threshold=0.5)
    
    print("\nEvaluating with optimal threshold...")
    metrics_optimal = evaluate_model(model, test_loader, config.DEVICE, 
                                    threshold=optimal_threshold)
    
    def print_metrics(m, title):
        print(f"\n{title}:")
        print(f"Accuracy:  {m['accuracy']:.4f}")
        print(f"Precision: {m['precision']:.4f}")
        print(f"Recall:    {m['recall']:.4f}")
        print(f"F1 Score:  {m['f1']:.4f}")
        print(f"ROC-AUC:   {m['roc_auc']:.4f}")
    
    print_metrics(metrics_default, "Test Results (threshold=0.5)")
    print_metrics(metrics_optimal, f"Test Results (threshold={optimal_threshold:.4f})")
    
    plot_confusion_matrix(metrics_optimal['confusion_matrix'],
                         os.path.join(config.RESULTS_DIR, 'confusion_matrix.png'))
    plot_roc_curve(metrics_optimal['labels'], metrics_optimal['probabilities'],
                  os.path.join(config.RESULTS_DIR, 'roc_curve.png'))
    plot_precision_recall_curve(metrics_optimal['labels'], metrics_optimal['probabilities'],
                               os.path.join(config.RESULTS_DIR, 'pr_curve.png'))
    
    print(f"\nResults saved to: {config.RESULTS_DIR}")
    print("Training complete!")


if __name__ == "__main__":
    main()