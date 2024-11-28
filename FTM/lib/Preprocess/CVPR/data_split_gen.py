"""
本代码用于生成自家数据集的data split
"""
import os
import numpy as np
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
    """
    Generate Data_split.npy
        0729, 0730, 0731, 0801, 0802, 0804, 0805, 0806, 0807, 0809, 0812 ---- train
        0813, 0814, 0815 ---- test
    """
    base = opt['base']
    max_len = opt['max_len']
    save = opt['save']
    
    train_split_ls, test_split_ls = [], []
    train_set = ['0729', '0730', '0731', '0801', '0802', '0804', '0805', '0806', '0807', '0809', '0812']
    test_set = ['0813', '0814', '0815']
    
    for root, dirs, files in os.walk(base):
        if 'keypoints.npy' in files: # Arrive 
            print(root)
            data = np.load(os.path.join(root, 'pressure.npz'), allow_pickle=True)['pressure']
            l = len(data)
            
            if os.path.basename(os.path.dirname(os.path.dirname(root))) in train_set:
                for index in range(2, int(l/max_len)):
                    item = root+f'/{index}'
                    train_split_ls.append(item)
            elif os.path.basename(os.path.dirname(os.path.dirname(root))) in test_set:
                for index in range(2, int(l/max_len)):
                    item = root+f'/{index}'
                    test_split_ls.append(item)                
            else:
                print(root)
                assert 0
      
    results = {
        'train': train_split_ls,
        'test': test_split_ls,
    }        
    np.save(save, results)
    print('Now you have created a data split file!')
        
def mini_data_split(opt):
    """
    Generate mini data split file to tune
    """      
    base = './mini_data/'
    max_len = opt['max_len']
    
    l = []
    for index in range(2, 10):
        path = base+f'{index}'
        l.append(path)
    results = {
        'train': l,
        'test': l,
    }
    np.save('./essentials/promini_data_split.npy', results)
    print('Now you have created mini data split!')
          

if __name__=='__main__':              
    opt = {
        'base': '/nasdata/shenghao/Pressure_release',
        'max_len': 40,
        'save': '/home/jiayi/cvpr/FTM/FTM/essentials/data_split.npy',
    }
    mini_data_split(opt)