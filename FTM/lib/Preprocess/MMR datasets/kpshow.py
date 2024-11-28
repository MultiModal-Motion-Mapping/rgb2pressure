'''
    这个代码用于可视化2D关节点信息
'''
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from lib.Dataset import make_dataset

connect = [[0,1], [0,2], [0,18], [0,17],
            [1,2], [1,3],
            [2,4],
            [3,5],
            [4,6],
            [5,18], [5,7],
            [6,18], [6,8],
            [7,9],
            [8,10],
            [11,19], [11,13],
            [12,19], [12,14],
            [13,15],
            [14,16],
            [15,20], [15,22], [15,24],
            [16,21], [16,23], [16,25],
            [18,19]
            ]

def kpconnect(kps, ax):
    for cont in connect:
        ax.plot([kps[cont[0]][0], kps[cont[1]][0]],[kps[cont[0]][1], kps[cont[1]][1]], color='r')
    return ax

def kpshow(kps, seq_name, frame_ids):
    '''
    input: torch.Tensor (b, 5, kps, 3)
    output: image
    '''
    
    kps = kps.numpy()[...,:2] if type(kps)==torch.Tensor else kps
    batchsize, seqlen, kpsnum, dim = kps.shape
    kps = kps.reshape(-1, kpsnum, dim)

    for index in [2]:
        kp = kps[index]
        
        fig, ax = plt.subplots(1,1)
        for _ in range(kpsnum):
            ax.scatter(x=kp[_][0], y=kp[_][1])
        
        ax = kpconnect(kp, ax)
        ax.invert_yaxis()
        ax.set_title('2Dkeypoints')
        ax.axis('off')
        
        plt.savefig(f'./keypoints/{frame_ids}.jpg')
        plt.close() 
    