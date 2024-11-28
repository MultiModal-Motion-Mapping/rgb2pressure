"""
Make residual block
"""
import torch
import torch.nn as nn
import numpy as np

class residual(nn.Module):
    def __init__(self):
        super(residual, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.Dropout(p=0.1),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(8),
            nn.Dropout(p=0.1),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(8),
            nn.Dropout(p=0.1),
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        b, c, h, w = x.shape
        y1 = self.conv1(x)
        y2 = self.conv2(x)
        y3 = self.conv3(x)
        
        y = self.sigmoid(y1+y2+y3)
        
        return y    