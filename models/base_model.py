import torch.nn as nn
from torchvision import models
from torchvision.models import (
    ResNet50_Weights, EfficientNet_B0_Weights, 
    VGG16_BN_Weights, ConvNeXt_Tiny_Weights
)


class MelanomaClassifier(nn.Module):
    def __init__(self, model_name: str = 'resnet50', num_classes: int = 2, 
                 pretrained: bool = True):
        super(MelanomaClassifier, self).__init__()
        self.model_name = model_name.lower()
        
        if self.model_name == 'resnet50':
            weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            self.backbone = models.resnet50(weights=weights)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            
        elif self.model_name == 'efficientnet':
            weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.efficientnet_b0(weights=weights)
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
            
        elif self.model_name == 'vgg16':
            weights = VGG16_BN_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.vgg16_bn(weights=weights)
            num_features = self.backbone.classifier[6].in_features
            self.backbone.classifier = nn.Identity()
            
        elif self.model_name == 'convnext':
            weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.convnext_tiny(weights=weights)
            num_features = self.backbone.classifier[2].in_features
            self.backbone.classifier = nn.Identity()
            
        elif self.model_name == 'senet':
            from .senet import se_resnet50
            self.backbone = se_resnet50(pretrained=pretrained)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)
    
    def get_features(self, x):
        return self.backbone(x)