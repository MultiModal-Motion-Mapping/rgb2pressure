'''
这个代码来自MI 1353140938@qq.com
主要用于进行smpl_vector的预处理，将其转化为npy文件
'''

import numpy as np
import os
import sys
import smplx
import torch
import trimesh

class smpl_vector_gen():
    def __init__(self, opt):
        self.opt = opt
        self.smpl_model = smplx.create(opt['pkl'], 'smpl')
        
    def infer(self):
        base_path = self.opt['base_path']
        save_path = self.opt['save_path']
        for root,dirs,files in os.walk(base_path):
            if dirs==[]:# 到达叶节点
                print(f'Begin to save {root}!')
                for file in files:
                    # 保存路径
                    path = os.path.join(root, file)
                    # 提取数据
                    data = np.load(path, allow_pickle=True)
                    try:
                        body_pose, global_rot, transl = data['body_pose'].reshape(23,3), \
                                                        data['global_rot'], \
                                                        data['transl']
                    except:
                        print('Here is an error!')
                        print('data\'s files are ', data.files)
                        print('root is ', path)
                        exit()
                    
                    body_pose, global_rot, transl = np.stack([body_pose],axis=0),\
                                                    np.stack([global_rot], axis=0),\
                                                    np.stack(transl, axis=0)
                    assert body_pose.shape==(1, 23, 3) 
                    assert global_rot.shape==(1, 1, 3)
                    assert transl.shape==(1, 3)
                                     
                    # 创建smpl模型
                    result = self.smpl_model(betas=torch.zeros(1, 10), # shape parameters
                                            body_pose=torch.Tensor(body_pose), # pose parameters
                                            global_orient=torch.Tensor(global_rot), # global orientation
                                            transl=torch.Tensor(transl)) # global translation

                    # 获取两脚踝之间的相对坐标
                    smpl_vector = (result.joints[:, 7, :] - result.joints[:, 8, :]).detach().numpy() # 左脚-右脚
                    l_ankle = result.joints[:, 7, :].detach().numpy()
                    
                    smpl = {
                        'smpl_vector': smpl_vector,
                        'l_ankle': l_ankle,
                    }
                    
                    # 保存
                    date, seq, monment = '20230422', root.split('/')[1:][6], root.split('/')[1:][7].replace("_",'')
                    filename = date+'_'+seq+'_'+monment+'_'+file[:-4].replace('_','')+'.npy'
                    np.save(os.path.join(save_path, filename), smpl)
                          
                
if __name__=='__main__':
    opt = {
        'base_path': '/nasdata/jiayi/MMVP/annotations/20230422/smpl_pose',
        'pkl': './essentials/SMPL_NEUTRAL.pkl',
        'save_path': '/nasdata/jiayi/MMVP/vectors/'
    }
    try:
        os.mkdir(opt['save_path'], mode=0o777)
    except FileExistsError:
        print('dir has existed!')
        
    
    G = smpl_vector_gen(opt)
    G.infer()
        
    
    
