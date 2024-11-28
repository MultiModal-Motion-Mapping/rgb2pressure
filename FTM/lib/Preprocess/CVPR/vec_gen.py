"""
本代码用于提取vec
"""
import numpy as np
import os
import json
import torch
import smplx

def vec_gen(opt):
    base = opt['base']
    smpl_model = smplx.create('./essentials/SMPL_NEUTRAL.pkl', 'smpl')
    
    for root, dirs, files in os.walk(base, topdown=False):
        if 'smpl.npy' in files:
            print('Now we process ', root)
            
            data = np.load(os.path.join(root, 'smpl.npy'), allow_pickle=True).item()
            betas, global_orient, body_pose, transl = data['betas'], data['global_orient'], \
                                                      data['body_pose'], data['transl']
            
            l = len(global_orient)
            
            betas, global_orient, body_pose, transl = torch.stack([torch.from_numpy(betas) for _ in range(l)], dim=0).float(), \
                                                      torch.from_numpy(global_orient).float(), \
                                                      torch.from_numpy(body_pose).float(), \
                                                      torch.from_numpy(transl).float()
            result = smpl_model(betas=betas, # shape parameters
                    body_pose=body_pose, 
                    global_orient=global_orient, 
                    transl=transl 
                    )
            
            vector_l = result.joints[:,7,:]
            vector_r = result.joints[:,8,:]
            
            vectors = torch.stack([vector_l, vector_r], dim=0).numpy()
            np.save(os.path.join(root, 'fvectors.npy'), vectors)
            
            

def check(opt):
    data = np.load(opt['save'], allow_pickle=True).item()
    print('hi')
    
    
    
if __name__=='__main__':
    opt = {
        'base': '/nasdata/shenghao/Pressure_release',
        'save': '/home/jiayi/cvpr/fpp-test/FPP-test/essentials/kps.npy'
    }
    vec_gen(opt)
    # check(opt)