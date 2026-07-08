[🇫🇷 Français](./README.fr.md) | [🇬🇧 English](./README.md)

# Melanoma Classifier 

## What Was Done

### 1. Removed Components
- ✅ Temperature Scaling (TemperatureScaler class)
- ✅ Calibration functions (fit_temperature)
- ✅ ECE (Expected Calibration Error)
- ✅ Brier Score
- ✅ Reliability diagrams
- ✅ All calibration-related code

### 2. Added Components
- ✅ ConvNeXt model support (integrated into base_model.py)
- ✅ GradCAM implementation for interpretability
- ✅ GradCAM visualization script

### 3. Unified Structure
- ✅ Separated concerns into modules: config, data, models, training, evaluation, interpretability
- ✅ Removed all notebook code
- ✅ Created clean Python scripts
- ✅ Removed all comments from code

## File Structure

```
melanoma-classifier/
│
├── config/
│   └── config.py                 # All configuration settings
│
├── data/
│   ├── __init__.py
│   ├── dataset.py                # MelanomaDataset class
│   └── data_loader.py            # load_ham10000_data function
│
├── models/
│   ├── __init__.py
│   ├── base_model.py             # MelanomaClassifier (supports all models)
│   ├── senet.py                  # SENet implementation
│   └── losses.py                 # FocalLoss
│
├── training/
│   ├── __init__.py
│   ├── trainer.py                # Training loop, optimizers
│   └── augmentation.py           # Albumentations transforms
│
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py                # Evaluation metrics (no calibration)
│   └── visualization.py          # Plotting functions
│
├── interpretability/
│   ├── __init__.py
│   └── gradcam.py                # GradCAM implementation
│
├── utils/
│   └── __init__.py
│
├── train.py                      # Main training script
├── evaluate.py                   # Evaluation script
├── visualize_gradcam.py          # GradCAM visualization script
├── requirements.txt              # Dependencies
└── README.md                     # Documentation
```

## Supported Models

1. **ResNet50** - Set `MODEL_NAME = 'resnet50'` in config
2. **SENet** - Set `MODEL_NAME = 'senet'` in config
3. **EfficientNet** - Set `MODEL_NAME = 'efficientnet'` in config
4. **ConvNeXt** - Set `MODEL_NAME = 'convnext'` in config (NEW)
5. **VGG16** - Set `MODEL_NAME = 'vgg16'` in config

## Key Features

### Training Features
- Focal Loss for class imbalance
- Discriminative learning rates (different LRs for different layers)
- Warmup phase (head-only training first)
- Fine-tuning phase (full model training)
- Early stopping
- Learning rate scheduling

### Evaluation Metrics (Simplified)
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion Matrix
- Precision-Recall Curve

### Interpretability
- GradCAM heatmaps
- Visual explanations of model predictions
- Works with all model architectures

## Usage Examples

### 1. Train a Model

```bash
# Default (ResNet50)
python train.py

# For other models, edit config/config.py:
# MODEL_NAME = 'senet'  # or 'convnext', 'efficientnet', 'vgg16'
```

### 2. Evaluate a Model

```bash
# With automatic threshold finding
python evaluate.py --checkpoint output/checkpoints/best_model.pth --model resnet50

# With custom threshold
python evaluate.py --checkpoint output/checkpoints/best_model.pth --model resnet50 --threshold 0.6
```

### 3. Generate GradCAM Visualizations

```bash
# Generate 20 visualizations from test set
python visualize_gradcam.py \
    --checkpoint output/checkpoints/best_model.pth \
    --model resnet50 \
    --num_samples 20 \
    --split test

# Visualize from validation set
python visualize_gradcam.py \
    --checkpoint output/checkpoints/best_model.pth \
    --model convnext \
    --num_samples 10 \
    --split val \
    --output_dir ./output/gradcam_val
```

## What Changed from Original Code

### From ResNet Notebook
- Removed all notebook-specific code
- Removed calibration/temperature scaling
- Simplified metrics
- Kept core training logic
- Moved to modular structure

### From SENet Notebook
- Extracted SENet architecture to models/senet.py
- Integrated into unified training pipeline
- Removed calibration code
- Standardized with other models

### From model.py
- Split into multiple modules
- Removed temperature scaling
- Removed calibration metrics (ECE, Brier, reliability)
- Kept Focal Loss
- Kept discriminative learning rates
- Kept augmentation pipeline

## New Additions

### ConvNeXt Support
- Added to models/base_model.py
- Uses torchvision's ConvNeXt implementation
- Supports pretrained weights
- Compatible with all training features

### GradCAM
- New interpretability module
- Generates attention heatmaps
- Shows what the model "looks at"
- Helps debug and build trust
- Works with all architectures

## Next Steps

1. Place HAM10000 data in `./data/` directory
2. Edit `config/config.py` to choose model and hyperparameters
3. Run `python train.py` to train
4. Run `python evaluate.py` to evaluate
5. Run `python visualize_gradcam.py` to interpret

## Dependencies

All dependencies are in requirements.txt:
- numpy
- pandas
- torch
- torchvision
- Pillow
- opencv-python
- tqdm
- matplotlib
- seaborn
- scikit-learn
- albumentations
