"""
    预测左右脚跟位置向量
"""
import torch
import torch.nn as nn
import numpy as np
import os
import torch.nn.functional as F

from lib.Networks.Attention import SelfAttention


class FVecNet(nn.Module):
    def __init__(self, input_size=2, input_channel=8):
        super(FVecNet, self).__init__()
        
        self.input_size = input_size
        self.input_channel = input_channel
        
        self.Encoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.LayerNorm(32),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25),
            nn.Linear(32, 64),
            nn.LayerNorm(64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25),
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25),
        )
        
        self.TimeNN = SelfAttention(input_size=128, hidden_size=128)

        self.Decoder = nn.Sequential(
            nn.Linear(128, 64),
            nn.Sigmoid(),
            nn.Linear(64, 32),
            nn.Sigmoid(),
            nn.Linear(32, 16),
            nn.Sigmoid(),
            nn.Linear(16, 8),
            nn.Sigmoid(),
            nn.Linear(8, 4),
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        b, seqlen, h, w = x.shape
        x = x.reshape(-1, h, w)
        x = x.reshape(b*seqlen, h*w)
        
        feature = self.Encoder(x)
        feature = feature.squeeze(-1)
        feature = feature.reshape(b, seqlen, -1)
        
        feature = self.TimeNN(feature)
        
        y = self.Decoder(feature)
        
        return y
        