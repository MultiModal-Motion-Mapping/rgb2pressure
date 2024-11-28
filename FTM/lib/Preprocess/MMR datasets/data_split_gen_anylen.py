'''
本代码用于生成自由时间长度的data_split文件，用于测试自由长度的模型效果
'''
import numpy as np
import torch
import math
import os
import re


def extract_s_and_mocap(input_string):
    # 匹配 S 后面跟两个数字的部分，以及 'MoCap' 后面的所有内容
    mocap_match = re.search(r'MoCap.*', input_string)
    
    mocap_value = mocap_match.group(0) if mocap_match else None  # 提取 'MoCap...'
    
    return mocap_value


def get_first_level_subfolders(folder_path):
    # 使用os.listdir获取当前路径下的所有项
    subfolders = [f.path for f in os.scandir(folder_path) if f.is_dir()]
    return subfolders

def check_insole_folder(folder_path):
    # 构建完整的 insole 文件夹路径
    insole_path = os.path.join(folder_path, 'insole')
    
    # 检查该路径是否存在并且是一个目录
    if os.path.isdir(insole_path):
        return True
    else:
        return False

def data_split_gen(opt):
    train, test = {}, {}
    train_seqs = ['S{:02d}'.format(index) for index in range(1,10)]
    test_seqs = ['S{:02d}'.format(index) for index in range(10, 13)]
    
    # 初始化
    train.update(
        {'20230422': {
            'S01' : [],
            'S02' : [],
            'S03' : [],
            'S04' : [],
            'S05' : [],
            'S06' : [],
            'S07' : [],
            'S08' : [],
            'S09' : [],
        }}
        )
    test.update(
        {'20230422': {
            'S10' : [],
            'S11' : [],
            'S12' : [],
        }}
        )
    
    
    for month in ['20230422']:
        for seq in ['S{:02d}'.format(index) for index in range(1,13)]:
            dirpath = os.path.join(opt['base_path'], month, seq)
            subfolders = get_first_level_subfolders(dirpath)
        
            for seq_path in subfolders:
                mocap = extract_s_and_mocap(seq_path)
                if not check_insole_folder(seq_path):
                    continue
                
                filepath = os.path.join(seq_path, 'keypoints')
                for root, dirs, files in os.walk(filepath):
                    files_len = len(files)
                # 判断
                if files_len <=30: # 太短的不要
                    continue
                
                filepath = os.path.join(seq_path, 'insole')
                for root, dirs, files in os.walk(filepath):
                    files_len = len(files)
                # 判断
                if files_len <=30:
                    continue
                
                if seq in train_seqs:
                    train[f'{month}'][f'{seq}'].append(mocap)
                else:
                    test[f'{month}'][f'{seq}'].append(mocap)
                    
    
    new_split = {
        'train': train,
        'test': test,
    }
    
    # 保存
    np.save(opt['save_path'], new_split)
    print('Now you have created a new dataset split!')

def check(opt):
    data = np.load(opt['save_path'], allow_pickle=True).item()
    print('good')


opt = {
    'base_path': '/nasdata/jiayi/MMVP/images/images',
    'save_path': './essentials/dataset_split_attention_anylen.npy',
}
# data_split_gen(opt)
check(opt)