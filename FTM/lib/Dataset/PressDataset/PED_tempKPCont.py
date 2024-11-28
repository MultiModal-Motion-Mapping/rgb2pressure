import json
import cv2
import os
import os.path as osp
import numpy as np
import torch
import random
import pickle, glob
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from icecream import ic
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time

from lib.Dataset.ImageModule import ImageModule
from lib.Dataset.InsoleModule import InsoleModule


class ContDataset(Dataset):
    def __init__(self, opt, phase):
        self.phase = phase
        self.basedir = opt.datadir
        self.seq_name = opt.seq_name
        self.vectordir = opt.vector
        
        self.insole_module = InsoleModule(self.basedir)
        self.image_module = ImageModule(opt)
        self.normalize_img = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225])

        self.w_sc = opt.w_sc
        self.w_nc = opt.w_nc

        self.img_res = opt.img_res
        self.is_aug = opt.aug.is_aug
        self.scale_factor = opt.aug.scale_factor  # rescale bounding boxes by a factor of [1-options.scale_factor,1+options.scale_factor]
        self.noise_factor = opt.aug.noise_factor
        self.rot_factor = opt.aug.rot_factor  # Random rotation in the range [-rot_factor, rot_factor]

        self.is_sam = opt.aug.is_sam
        self.img_W = 1280
        self.img_H = 720

        self.images_fn = np.load(opt.tv_fn, allow_pickle=True).item()
        self.images_fn = self.images_fn[self.phase]
        if self.phase == 'train':
            random.shuffle(self.images_fn)
        
        self.insole = np.load(os.path.join('./mini_data', 'pressure.npz'), allow_pickle=True)['pressure']
        self.insole = torch.from_numpy(self.insole)/255
        self.insole = F.pad(self.insole, (24, 24, 24, 24), mode="constant", value=0)
        self.contact = (self.insole != 0).float()
        
        self.feats = torch.load(os.path.join('./mini_data', 'feature.pth'))
        
        self.vector = np.load(os.path.join('./mini_data', 'fvectors.npy'), allow_pickle=True)
        self.vector = torch.from_numpy(self.vector)
        self.vector[:,:,0] = self.vector[:,:,0]/1.5
        self.vector[:,:,1] = -self.vector[:,:,1]/2
        
        
        
        self.max_len = 40

    def augm_params(self, flip, sc):
        """Get augmentation parameters."""
        # We flip with probability 1/2
        if np.random.uniform() <= 0.5:
            flip = 1

        # The scale is multiplied with a number
        # in the area [1-scaleFactor,1+scaleFactor]
        sc = min(1 + self.scale_factor,
                 max(1 - self.scale_factor, np.random.randn() * self.scale_factor + 1))
        if sc > 1:
            sc = 2 - sc
        # but it is zero with probability 3/5
        # rebg = np.random.uniform()
        return flip, sc

    def __len__(self):
        return len(self.images_fn)

    def __getitem__(self, index):
        """
        Offer data
        """
        # Get path
        base, num = os.path.split(self.images_fn[index])
        num = int(num)
        
        # kps, kpf
        # kps = np.load(os.path.join(base, '2dkps.npy'), allow_pickle=True).item()['kps']
        # kps = kps[num*self.max_len: self.max_len*(num+1)]
        # kpsf = torch.cat([kps[:,15:17], kps[:, 20:26]], dim=1)
        
        # insole and contact
        insole = self.insole[num*self.max_len: self.max_len*(num+1)]
        contact = self.contact[num*self.max_len: self.max_len*(num+1)]
        
        # vector
        vector_l, vector_r = self.vector[0, num*self.max_len: self.max_len*(num+1), :2], self.vector[1, num*self.max_len: self.max_len*(num+1), :2]
        
        # feature
        feats = self.feats[num*self.max_len: self.max_len*(num+1), :]
        
        
        res = {
            # 'keypoints': None,
            'insole': insole,
            # 'keypoints_f': None,
            'vector_l': vector_l,
            'vector_r': vector_r,
            'case_name': self.images_fn[index],
            'features' : feats,
            'contact': contact,
        }
        return res