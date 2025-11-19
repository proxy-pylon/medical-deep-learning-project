import torch
import os


class Config:
    HAM10000_BASE = './data/archive'
    ISIC_BASE = 'not defined'
    
    OUTPUT_DIR = './output/'
    CHECKPOINT_DIR = OUTPUT_DIR + 'checkpoints'
    RESULTS_DIR = OUTPUT_DIR + 'results'
    
    MODEL_NAME = 'resnet50'
    IMG_SIZE = 224
    NUM_CLASSES = 2
    PRETRAINED = True
    
    BATCH_SIZE = 32
    NUM_EPOCHS = 250
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4
    EARLY_STOPPING_PATIENCE = 30
    
    FREEZE_EPOCHS = 3
    HEAD_LR_WARMUP = 1e-3
    HEAD_LR_FINETUNE = 1e-4
    
    BACKBONE_LR_LOW = 1e-5
    BACKBONE_LR_MID = 2e-5
    BACKBONE_LR_HIGH = 3e-5
    
    TEST_SIZE = 0.30
    VAL_SIZE = 0.20
    RANDOM_STATE = 42
    
    USE_MIXUP = False
    USE_CUTMIX = False
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    NUM_WORKERS = 2