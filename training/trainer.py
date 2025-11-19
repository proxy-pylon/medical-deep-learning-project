import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Tuple, Dict, List
import os


def set_backbone_trainable(model: nn.Module, trainable: bool):
    for param in model.backbone.parameters():
        param.requires_grad = trainable


def build_optimizer_warmup(model: nn.Module, config):
    head_params = list(model.classifier.parameters())
    optimizer = optim.Adam(head_params, lr=config.HEAD_LR_WARMUP, 
                          weight_decay=config.WEIGHT_DECAY)
    return optimizer


def build_optimizer_finetune(model: nn.Module, config):
    if config.MODEL_NAME == 'resnet50':
        early_layers = list(model.backbone.layer1.parameters()) + \
                      list(model.backbone.layer2.parameters())
        mid_layers = list(model.backbone.layer3.parameters())
        late_layers = list(model.backbone.layer4.parameters())
    elif config.MODEL_NAME == 'senet':
        early_layers = list(model.backbone.layer1.parameters()) + \
                      list(model.backbone.layer2.parameters())
        mid_layers = list(model.backbone.layer3.parameters())
        late_layers = list(model.backbone.layer4.parameters())
    elif config.MODEL_NAME == 'efficientnet':
        all_backbone = list(model.backbone.parameters())
        n = len(all_backbone)
        early_layers = all_backbone[:n//3]
        mid_layers = all_backbone[n//3:2*n//3]
        late_layers = all_backbone[2*n//3:]
    elif config.MODEL_NAME == 'convnext':
        all_backbone = list(model.backbone.parameters())
        n = len(all_backbone)
        early_layers = all_backbone[:n//3]
        mid_layers = all_backbone[n//3:2*n//3]
        late_layers = all_backbone[2*n//3:]
    else:
        all_backbone = list(model.backbone.parameters())
        n = len(all_backbone)
        early_layers = all_backbone[:n//3]
        mid_layers = all_backbone[n//3:2*n//3]
        late_layers = all_backbone[2*n//3:]
    
    head_params = list(model.classifier.parameters())
    
    optimizer = optim.Adam([
        {'params': early_layers, 'lr': config.BACKBONE_LR_LOW},
        {'params': mid_layers, 'lr': config.BACKBONE_LR_MID},
        {'params': late_layers, 'lr': config.BACKBONE_LR_HIGH},
        {'params': head_params, 'lr': config.HEAD_LR_FINETUNE},
    ], weight_decay=config.WEIGHT_DECAY)
    
    return optimizer


def train_epoch(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, 
                optimizer: optim.Optimizer, device: str) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc='Training')
    for batch in pbar:
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        pbar.set_postfix({'loss': loss.item(), 'acc': 100. * correct / total})
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc


def validate(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, 
             device: str) -> Tuple[float, float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation')
        for batch in pbar:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    
    from sklearn.metrics import f1_score
    f1 = f1_score(all_labels, all_preds, average='binary')
    
    return epoch_loss, epoch_acc, f1


def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                criterion: nn.Module, config, save_path: str) -> Dict[str, List[float]]:
    
    set_backbone_trainable(model, trainable=False)
    optimizer = build_optimizer_warmup(model, config)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                      factor=0.5, patience=5)
    
    current_phase = "warmup"
    
    history = {
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
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                             factor=0.5, patience=5)
            current_phase = "finetune"
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, 
                                           optimizer, config.DEVICE)
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
            }, save_path)
            print(f"✓ Best model saved! Val F1: {val_f1:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= config.EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping at epoch {epoch + 1}")
            break
    
    return history