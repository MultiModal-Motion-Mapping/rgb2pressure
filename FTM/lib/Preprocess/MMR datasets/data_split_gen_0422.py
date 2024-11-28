'''
这个代码由MI 1353140938@qq.com编写
本代码主要用于生成data_split_0422temporal5.npy文件
'''

import numpy as np
import os
import sys    
    
def data_split_gen(opt):
    # 初始化S01到S12空字典
    keys = [f'S{i:02d}' for i in range(1, 13)]
    dicts = {key: {} for key in keys}
    
    for root,dirs,files in os.walk(opt['base_path']):
        if dirs==[]:# 到达最终节点
            seq, Mocap = root.split('/')[-2], root.split('/')[-1]
            # 对files中的文件名称进行修改
            files_new = [file[-7:-4]+'.npy' for file in files[2:-2]]
            # 创建Mocap字典
            dict = {
                f'{Mocap}': files_new,
            }
            # 将Mocap字典保存在SXX字典里
            dicts[f'{seq}'].update(dict)
    train_dicts = {
        '20230422': {key: dicts[key] for key in opt['train_set']},
    }
    test_dicts = {
        '20230422': {key: dicts[key] for key in opt['test_set']},
    }
    new_split = {
        'train': train_dicts,
        'test': test_dicts,
    }
    
    # 保存
    np.save(opt['save_path'], new_split)
    print('Now you have created a new dataset split!')
    
def check(opt):
    data = np.load(opt['save_path'], allow_pickle=True).item()
    print('good')
    

    
    
if __name__=='__main__':
    opt = {
        'base_path': '/nasdata/jiayi/MMVP/annotations/20230422/smpl_pose',
        'train_set': [f'S{i:02d}' for i in range(1, 10)],
        'test_set': [f'S{i:02d}' for i in range(10, 13)],
        'save_path': './essentials/dataset_split_0422temporal5.npy',
    }
    data_split_gen(opt)
    # check(opt)
    
    

    
    
    
    
