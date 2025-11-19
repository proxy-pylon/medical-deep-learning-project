import os
import sys
import torch
import argparse
import matplotlib.pyplot as plt
import cv2
import numpy as np
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.config import Config
from data import MelanomaDataset, load_ham10000_data
from models import MelanomaClassifier
from training import get_val_transform
from interpretability import GradCAM, get_target_layer, overlay_heatmap, denormalize_image
from sklearn.model_selection import train_test_split


def visualize_gradcam(model, dataset, indices, save_dir, config, num_samples=10):
    os.makedirs(save_dir, exist_ok=True)
    
    target_layer = get_target_layer(model, config.MODEL_NAME)
    gradcam = GradCAM(model, target_layer)
    
    model.eval()
    
    for i, idx in enumerate(indices[:num_samples]):
        sample = dataset[idx]
        image_tensor = sample['image'].unsqueeze(0).to(config.DEVICE)
        true_label = sample['label'].item()
        
        with torch.no_grad():
            output = model(image_tensor)
            probs = torch.softmax(output, dim=1)
            pred_label = output.argmax(dim=1).item()
            confidence = probs[0, pred_label].item()
        
        cam = gradcam.generate_cam(image_tensor, target_class=pred_label)
        
        original_img = denormalize_image(sample['image'])
        heatmap_overlay = overlay_heatmap(original_img, cam, alpha=0.4)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(original_img)
        axes[0].set_title(f'Original\nTrue: {"Melanoma" if true_label == 1 else "Benign"}')
        axes[0].axis('off')
        
        axes[1].imshow(cam, cmap='jet')
        axes[1].set_title('GradCAM Heatmap')
        axes[1].axis('off')
        
        axes[2].imshow(heatmap_overlay)
        axes[2].set_title(f'Overlay\nPred: {"Melanoma" if pred_label == 1 else "Benign"}\nConf: {confidence:.3f}')
        axes[2].axis('off')
        
        plt.tight_layout()
        save_path = os.path.join(save_dir, f'gradcam_{i:03d}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved: {save_path}")
    
    gradcam.remove_hooks()


def main():
    parser = argparse.ArgumentParser(description='Visualize GradCAM for melanoma classifier')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--model', type=str, default='resnet50',
                       choices=['resnet50', 'senet', 'efficientnet', 'convnext', 'vgg16'],
                       help='Model architecture')
    parser.add_argument('--num_samples', type=int, default=10,
                       help='Number of samples to visualize')
    parser.add_argument('--output_dir', type=str, default='./output/gradcam',
                       help='Directory to save GradCAM visualizations')
    parser.add_argument('--split', type=str, default='test',
                       choices=['train', 'val', 'test'],
                       help='Which split to visualize')
    args = parser.parse_args()
    
    config = Config()
    config.MODEL_NAME = args.model
    
    print('=' * 70)
    print("GRADCAM VISUALIZATION")
    print('=' * 70)
    print(f"Model: {config.MODEL_NAME}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Split: {args.split}")
    print(f"Samples: {args.num_samples}")
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
    
    if args.split == 'train':
        selected_df = train_df
    elif args.split == 'val':
        selected_df = val_df
    else:
        selected_df = test_df
    
    dataset = MelanomaDataset(selected_df, get_val_transform(config.IMG_SIZE))
    
    print(f"\nLoading model...")
    model = MelanomaClassifier(config.MODEL_NAME, config.NUM_CLASSES, pretrained=False)
    checkpoint = torch.load(args.checkpoint, map_location=config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(config.DEVICE)
    
    melanoma_indices = selected_df[selected_df['binary_label'] == 1].index.tolist()
    benign_indices = selected_df[selected_df['binary_label'] == 0].index.tolist()
    
    num_melanoma = min(args.num_samples // 2, len(melanoma_indices))
    num_benign = min(args.num_samples // 2, len(benign_indices))
    
    np.random.seed(config.RANDOM_STATE)
    selected_melanoma = np.random.choice(melanoma_indices, num_melanoma, replace=False)
    selected_benign = np.random.choice(benign_indices, num_benign, replace=False)
    
    selected_indices = list(selected_melanoma) + list(selected_benign)
    
    print(f"\nGenerating GradCAM visualizations...")
    print(f"Melanoma samples: {num_melanoma}")
    print(f"Benign samples: {num_benign}")
    
    visualize_gradcam(model, dataset, selected_indices, args.output_dir, config, 
                     num_samples=len(selected_indices))
    
    print(f"\nVisualizations saved to: {args.output_dir}")


if __name__ == "__main__":
    main()