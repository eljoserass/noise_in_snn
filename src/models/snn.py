"""
SSD (Single Shot MultiBox Detector) using VGG-11 backbone with Spiking Neural Networks.
Designed for object detection on TUMTraf Event Dataset.
SNN (Spiking Neural Network) version for event-based images.
"""

import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
from typing import Tuple


class VGG11_SSD_SNN(nn.Module):
    """
    VGG11 backbone with SSD heads using Spiking Neural Networks for event-based images.
    Leaky Integrate-and-Fire (LIF) neurons enable temporal integration across timesteps.
    
    Args:
        num_classes: Number of object classes (excluding background)
        beta: Membrane potential decay rate (higher = more memory, 0.9 ≈ 10 timesteps)
        threshold: Spike threshold
        spike_grad: Surrogate gradient function for backpropagation
    """
    
    def __init__(self, num_classes: int = 21, beta: float = 0.9, threshold: float = 1.0, 
                 spike_grad=surrogate.fast_sigmoid(slope=25)):
        super().__init__()
        self.num_classes = num_classes
        
        # VGG11 Backbone with Leaky neurons
        # Block 1: 1 conv layer
        self.conv1 = nn.Conv2d(2, 64, 3, padding=1)  # 2 channels for event polarity
        self.lif1 = snn.Leaky(beta=beta, threshold=threshold, 
                              spike_grad=spike_grad, init_hidden=True)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # Block 2: 1 conv layer
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.lif2 = snn.Leaky(beta=beta, threshold=threshold,
                              spike_grad=spike_grad, init_hidden=True)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Block 3: 2 conv layers
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.lif3_1 = snn.Leaky(beta=beta, threshold=threshold,
                                spike_grad=spike_grad, init_hidden=True)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.lif3_2 = snn.Leaky(beta=beta, threshold=threshold,
                                spike_grad=spike_grad, init_hidden=True)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # Block 4: 2 conv layers
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.lif4_1 = snn.Leaky(beta=beta, threshold=threshold,
                                spike_grad=spike_grad, init_hidden=True)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.lif4_2 = snn.Leaky(beta=beta, threshold=threshold,
                                spike_grad=spike_grad, init_hidden=True)
        self.pool4 = nn.MaxPool2d(2, 2)
        
        # Block 5: 2 conv layers
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.lif5_1 = snn.Leaky(beta=beta, threshold=threshold,
                                spike_grad=spike_grad, init_hidden=True)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.lif5_2 = snn.Leaky(beta=beta, threshold=threshold,
                                spike_grad=spike_grad, init_hidden=True)
        self.pool5 = nn.MaxPool2d(2, 2)
        
        # Additional SSD layers
        self.conv6 = nn.Conv2d(512, 1024, 3, padding=1)
        self.lif6 = snn.Leaky(beta=beta, threshold=threshold,
                              spike_grad=spike_grad, init_hidden=True)
        self.conv7 = nn.Conv2d(1024, 1024, 1)
        self.lif7 = snn.Leaky(beta=beta, threshold=threshold,
                              spike_grad=spike_grad, init_hidden=True)
        
        self.conv8_1 = nn.Conv2d(1024, 256, 1)
        self.lif8_1 = snn.Leaky(beta=beta, threshold=threshold,
                                spike_grad=spike_grad, init_hidden=True)
        self.conv8_2 = nn.Conv2d(256, 512, 3, stride=2, padding=1)
        self.lif8_2 = snn.Leaky(beta=beta, threshold=threshold,
                                spike_grad=spike_grad, init_hidden=True)
        
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
            x: Input tensor of shape (batch, 2, height, width) - event frame at time t
               Membrane states are automatically maintained by snnTorch!
            
        Returns:
            classifications: Class predictions (batch, num_anchors, num_classes)
            regressions: Location predictions (batch, num_anchors, 4)
        """
        # x: [B, 2, 442, 482] - event frame at time t (TUMTraf events)
        
        # Block 1
        cur = self.conv1(x)
        spk = self.lif1(cur)  # snnTorch Leaky only returns spikes
        spk = self.pool1(spk)
        
        # Block 2
        cur = self.conv2(spk)
        spk = self.lif2(cur)
        spk = self.pool2(spk)
        
        # Block 3
        cur = self.conv3_1(spk)
        spk = self.lif3_1(cur)
        cur = self.conv3_2(spk)
        spk = self.lif3_2(cur)
        feat1 = self.pool3(spk)  # [B, 256, 55, 60] for 442×482 input
        
        # Block 4
        cur = self.conv4_1(feat1)
        spk = self.lif4_1(cur)
        cur = self.conv4_2(spk)
        spk = self.lif4_2(cur)
        feat2 = self.pool4(spk)  # [B, 512, 27, 30] for 442×482 input
        
        # Block 5
        cur = self.conv5_1(feat2)
        spk = self.lif5_1(cur)
        cur = self.conv5_2(spk)
        spk = self.lif5_2(cur)
        feat3 = self.pool5(spk)  # [B, 512, 13, 15] for 442×482 input
        
        # Extra layers
        cur = self.conv6(feat3)
        spk = self.lif6(cur)
        cur = self.conv7(spk)
        feat4 = self.lif7(cur)  # [B, 1024, 13, 15] for 442×482 input
        
        cur = self.conv8_1(feat4)
        spk = self.lif8_1(cur)
        cur = self.conv8_2(spk)
        feat5 = self.lif8_2(cur)  # [B, 512, 7, 8] for 442×482 input
        
        features = [feat1, feat2, feat3, feat4, feat5]
        
        # Detection heads (operating on spike rates)
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
    
    def reset_states(self):
        """
        Call this at the start of each video sequence.
        snnTorch automatically manages states with init_hidden=True.
        This ensures a clean slate for new video sequences.
        """
        for module in self.modules():
            if isinstance(module, snn.Leaky):
                module.reset_mem()
