import torch.nn as nn
import torch.nn.functional as F
import torch

from lib.Networks.Attention import Attention
from lib.Networks.Residual import residual
from lib.Networks.Upsample import upsample


class ContNetwork(nn.Module):
    def __init__(
            self, args,
            n_layers=1,
            hidden_size=2048,
            add_linear=False,
            bidirectional=False,
            use_residual=True,
    ):
        super(ContNetwork, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(1536, 1536),
            nn.BatchNorm1d(1536),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(1536, 3072),
            nn.BatchNorm1d(3072),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
        )
          
        self.attention = Attention()
        
        self.contact_decoder = nn.Sequential(
            nn.Linear(3072, 3072),
            nn.Dropout(p=0.1),
            nn.Linear(3072, 2304*2),
            nn.Dropout(p=0.1),
            nn.Linear(2304*2, 2304*2),
        )
        self.contact_output = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=1, padding=1),
            nn.Conv2d(in_channels=2, out_channels=2, kernel_size=2, stride=1, padding=0),
            nn.Sigmoid(),
        )
        self.press_decoder = nn.Sequential(
            nn.Linear(2304*2, 2304*2),
            nn.Dropout(p=0.1),
            nn.Linear(2304*2, 2304*2),
            nn.Dropout(p=0.1),
            nn.Linear(2304*2, 2304*2),
        )
        self.press_output = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=1, padding=1),
            nn.Conv2d(in_channels=2, out_channels=2, kernel_size=2, stride=1, padding=0),
            nn.Sigmoid(),
        )
        
        self.sigmoid = nn.Sigmoid()
        
        
        
        
    def forward(self, keypoints):
        batch_size, seqlen, l = keypoints.shape
        keypoints = keypoints.reshape(batch_size*seqlen, l)
        
        feat = self.encoder(keypoints).reshape(batch_size, seqlen, -1)

        
        feat = self.attention(feat)
        
        # Output
        hidden = self.contact_decoder(feat)
        contact = self.contact_output(hidden.reshape(batch_size*seqlen, 2, 48, 48))
        
        # hidden = self.press_decoder(hidden)
        # press = self.sigmoid(self.press_output(hidden.reshape(batch_size*seqlen, 2, 48, 48))+contact)*contact
        
        contact = contact.reshape(batch_size, seqlen, 2, 48, 48)
        # press = press.reshape(batch_size, seqlen, 2, 48, 48)
        
        return contact # , press
    



