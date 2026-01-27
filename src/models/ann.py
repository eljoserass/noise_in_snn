"""
SSD (Single Shot MultiBox Detector) using VGG-11 backbone from scratch.
Designed for object detection on TUMTraf Event Dataset.
ANN (Artificial Neural Network) version for RGB images.
"""

import torch
import torch.nn as nn
from typing import Tuple


class VGG11_SSD_ANN(nn.Module):
    """
    VGG11 backbone with SSD heads for object detection on RGB images.
    Multi-scale feature maps are used for detection at different resolutions.
    
    Args:
        num_classes: Number of object classes (excluding background)
    """
    
    def __init__(self, num_classes: int = 21):
        super().__init__()
        self.num_classes = num_classes
        
        # VGG11 Backbone
        # Block 1: 1 conv layer
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        # Block 2: 1 conv layer
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        # Block 3: 2 conv layers
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )  # Feature map 1: [B, 256, 60, 80] for 480×640 input
        
        # Block 4: 2 conv layers
        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )  # Feature map 2: [B, 512, 30, 40] for 480×640 input
        
        # Block 5: 2 conv layers
        self.block5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )  # Feature map 3: [B, 512, 15, 20] for 480×640 input
        
        # Additional SSD layers for more feature maps
        self.extra1 = nn.Sequential(
            nn.Conv2d(512, 1024, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, 1),
            nn.ReLU(inplace=True)
        )  # Feature map 4: [B, 1024, 15, 20] for 480×640 input (no pooling)
        
        self.extra2 = nn.Sequential(
            nn.Conv2d(1024, 256, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )  # Feature map 5: [B, 512, 8, 10] for 480×640 input
        
        # Number of anchors per feature map
        self.num_anchors = [4, 6, 6, 6, 4]
        
        # Classification heads for each feature map
        self.classifiers = nn.ModuleList([
            nn.Conv2d(256, self.num_anchors[0] * num_classes, 3, padding=1),
            nn.Conv2d(512, self.num_anchors[1] * num_classes, 3, padding=1),
            nn.Conv2d(512, self.num_anchors[2] * num_classes, 3, padding=1),
            nn.Conv2d(1024, self.num_anchors[3] * num_classes, 3, padding=1),
            nn.Conv2d(512, self.num_anchors[4] * num_classes, 3, padding=1),
        ])
        
        # Regression heads for each feature map (4 = dx, dy, dw, dh)
        self.regressors = nn.ModuleList([
            nn.Conv2d(256, self.num_anchors[0] * 4, 3, padding=1),
            nn.Conv2d(512, self.num_anchors[1] * 4, 3, padding=1),
            nn.Conv2d(512, self.num_anchors[2] * 4, 3, padding=1),
            nn.Conv2d(1024, self.num_anchors[3] * 4, 3, padding=1),
            nn.Conv2d(512, self.num_anchors[4] * 4, 3, padding=1),
        ])
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, 3, height, width)
            
        Returns:
            classifications: Class predictions (batch, num_anchors, num_classes)
            regressions: Location predictions (batch, num_anchors, 4)
        """
        # x: [B, 3, 480, 640] - single RGB frame (TUMTraf RGB)
        
        # Backbone
        x = self.block1(x)  # [B, 64, 240, 320]
        x = self.block2(x)  # [B, 128, 120, 160]
        
        feat1 = self.block3(x)  # [B, 256, 60, 80]
        feat2 = self.block4(feat1)  # [B, 512, 30, 40]
        feat3 = self.block5(feat2)  # [B, 512, 15, 20]
        feat4 = self.extra1(feat3)  # [B, 1024, 15, 20]
        feat5 = self.extra2(feat4)  # [B, 512, 8, 10]
        
        features = [feat1, feat2, feat3, feat4, feat5]
        
        # Detection heads
        classifications = []
        regressions = []
        
        for feat, clf, reg in zip(features, self.classifiers, self.regressors):
            # Classification: [B, num_anchors*num_classes, H, W]
            cls = clf(feat)
            B, _, H, W = cls.shape
            cls = cls.permute(0, 2, 3, 1).contiguous()
            cls = cls.view(B, -1, self.num_classes)
            classifications.append(cls)
            
            # Regression: [B, num_anchors*4, H, W]
            reg_out = reg(feat)
            reg_out = reg_out.permute(0, 2, 3, 1).contiguous()
            reg_out = reg_out.view(B, -1, 4)
            regressions.append(reg_out)
        
        # Concatenate all predictions
        classifications = torch.cat(classifications, dim=1)  # [B, total_anchors, num_classes]
        regressions = torch.cat(regressions, dim=1)  # [B, total_anchors, 4]
        
        return classifications, regressions
