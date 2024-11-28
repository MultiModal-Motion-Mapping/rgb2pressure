"""
Build Unsample Networks
"""
import torch
import torch.nn as nn

class upsample(nn.Module):
    def __init__(self):
        super(upsample, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Upsample(scale_factor=(2,2), mode='nearest'),
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1),
        )
        self.conv2 = nn.Sequential(
            nn.Upsample(scale_factor=(2,2), mode='nearest'),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1),
        )
        self.conv3 = nn.Sequential(
            nn.Upsample(scale_factor=(2,2), mode='nearest'),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
        )
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x