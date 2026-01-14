"""
SSD (Single Shot MultiBox Detector) using VGG-9 backbone from scratch.
Designed for object detection on TUMTraf Event Dataset.
ANN (Artificial Neural Network) version.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from ..utils.anchors import generate_anchors, get_num_anchors_per_cell, DEFAULT_ANCHOR_CONFIG


class VGG9Backbone(nn.Module):
    """
    VGG-9 backbone for feature extraction.
    Architecture: 2 conv blocks with 2 layers each, then 2 blocks with 1 layer, ending with FC layers
    Modified to output multi-scale feature maps for SSD.
    """
    
    def __init__(self, in_channels: int = 3):
        super(VGG9Backbone, self).__init__()
        
        # Block 1: 2 conv layers, output: 64 channels
        self.conv1_1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.bn1_1 = nn.BatchNorm2d(64)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn1_2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block 2: 2 conv layers, output: 128 channels
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(128)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block 3: 2 conv layers, output: 256 channels
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(256)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn3_2 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block 4: 2 conv layers, output: 512 channels (Feature map 1 for SSD)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4_1 = nn.BatchNorm2d(512)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn4_2 = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block 5: Additional conv for more features (Feature map 2 for SSD)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn5_1 = nn.BatchNorm2d(512)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn5_2 = nn.BatchNorm2d(512)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        # Block 1
        x = F.relu(self.bn1_1(self.conv1_1(x)))
        x = F.relu(self.bn1_2(self.conv1_2(x)))
        x = self.pool1(x)
        
        # Block 2
        x = F.relu(self.bn2_1(self.conv2_1(x)))
        x = F.relu(self.bn2_2(self.conv2_2(x)))
        x = self.pool2(x)
        
        # Block 3
        x = F.relu(self.bn3_1(self.conv3_1(x)))
        x = F.relu(self.bn3_2(self.conv3_2(x)))
        x = self.pool3(x)
        
        # Block 4 - Feature map 1 (before pooling for higher resolution)
        x = F.relu(self.bn4_1(self.conv4_1(x)))
        x = F.relu(self.bn4_2(self.conv4_2(x)))
        feat1 = x  # 40x30 for 640x480 input
        x = self.pool4(x)
        
        # Block 5 - Feature map 2
        x = F.relu(self.bn5_1(self.conv5_1(x)))
        x = F.relu(self.bn5_2(self.conv5_2(x)))
        feat2 = x  # 20x15 for 640x480 input
        x = self.pool5(x)
        
        feat3 = x  # 10x7 for 640x480 input
        
        return feat1, feat2, feat3


class SSDExtraLayers(nn.Module):
    """
    Additional convolutional layers for SSD to generate more feature maps at different scales.
    """
    
    def __init__(self):
        super(SSDExtraLayers, self).__init__()
        
        # Extra layer 1: 512 -> 256
        self.conv6_1 = nn.Conv2d(512, 256, kernel_size=1)
        self.conv6_2 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        
        # Extra layer 2: 512 -> 128
        self.conv7_1 = nn.Conv2d(512, 128, kernel_size=1)
        self.conv7_2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        
        # Extra layer 3: 256 -> 128
        self.conv8_1 = nn.Conv2d(256, 128, kernel_size=1)
        self.conv8_2 = nn.Conv2d(128, 256, kernel_size=3)
        
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        # Extra layer 1
        x = F.relu(self.conv6_1(x))
        feat4 = F.relu(self.conv6_2(x))
        
        # Extra layer 2
        x = F.relu(self.conv7_1(feat4))
        feat5 = F.relu(self.conv7_2(x))
        
        # Extra layer 3
        x = F.relu(self.conv8_1(feat5))
        feat6 = F.relu(self.conv8_2(x))
        
        return feat4, feat5, feat6


class PredictionHead(nn.Module):
    """
    Prediction head for SSD that outputs class scores and bounding box offsets.
    """
    
    def __init__(self, in_channels: int, num_anchors: int, num_classes: int):
        super(PredictionHead, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        
        # Classification head
        self.cls_conv = nn.Conv2d(in_channels, num_anchors * num_classes, kernel_size=3, padding=1)
        
        # Localization head (4 values: dx, dy, dw, dh)
        self.loc_conv = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=3, padding=1)
        
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.size(0)
        
        # Class predictions: (batch, num_anchors * num_classes, H, W) -> (batch, H*W*num_anchors, num_classes)
        cls_pred = self.cls_conv(x)
        cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous()
        cls_pred = cls_pred.view(batch_size, -1, self.num_classes)
        
        # Location predictions: (batch, num_anchors * 4, H, W) -> (batch, H*W*num_anchors, 4)
        loc_pred = self.loc_conv(x)
        loc_pred = loc_pred.permute(0, 2, 3, 1).contiguous()
        loc_pred = loc_pred.view(batch_size, -1, 4)
        
        return cls_pred, loc_pred


class SSD_VGG9(nn.Module):
    """
    SSD (Single Shot MultiBox Detector) with VGG-9 backbone.
    
    Args:
        num_classes: Number of object classes (excluding background)
        input_size: Tuple of (height, width) for input images
        in_channels: Number of input channels (3 for RGB)
    """
    
    def __init__(self, num_classes: int, input_size: Tuple[int, int] = (480, 640), in_channels: int = 3):
        super(SSD_VGG9, self).__init__()
        
        self.num_classes = num_classes + 1  # Add background class
        self.input_size = input_size
        
        # Backbone
        self.backbone = VGG9Backbone(in_channels=in_channels)
        
        # Extra layers
        self.extras = SSDExtraLayers()
        
        # Feature map channels
        feat_channels = [512, 512, 512, 512, 256, 256]
        
        # Calculate number of anchors per feature map (using shared utility)
        self.num_anchors = get_num_anchors_per_cell()
        
        # Prediction heads for each feature map
        self.pred_heads = nn.ModuleList([
            PredictionHead(feat_channels[i], self.num_anchors[i], self.num_classes)
            for i in range(6)
        ])
        
        # Generate default anchor boxes (using shared utility)
        self.anchors = generate_anchors()
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, channels, height, width)
            
        Returns:
            cls_preds: Class predictions (batch, num_anchors, num_classes)
            loc_preds: Location predictions (batch, num_anchors, 4)
        """
        # Get backbone features
        feat1, feat2, feat3 = self.backbone(x)
        
        # Get extra features
        feat4, feat5, feat6 = self.extras(feat3)
        
        features = [feat1, feat2, feat3, feat4, feat5, feat6]
        
        # Get predictions from each feature map
        cls_preds = []
        loc_preds = []
        
        for feat, head in zip(features, self.pred_heads):
            cls_pred, loc_pred = head(feat)
            cls_preds.append(cls_pred)
            loc_preds.append(loc_pred)
        
        # Concatenate predictions from all feature maps
        cls_preds = torch.cat(cls_preds, dim=1)
        loc_preds = torch.cat(loc_preds, dim=1)
        
        return cls_preds, loc_preds
    
    def get_anchors(self, device: torch.device) -> torch.Tensor:
        """Get anchor boxes on the specified device."""
        return self.anchors.to(device)
