import sys
sys.path.append("/home/jiayi/MMVP/MMVP_MI/FPP-Net-new/")
from lib.config.config import config_cont as config
import argparse
import torch
import os,cv2
from tqdm import tqdm,trange
from torch.utils.data import DataLoader
from torch.nn import DataParallel
from icecream import ic
import numpy as np
import matplotlib.pyplot as plt

from lib.Dataset import make_dataset
from lib.Networks import make_network
from lib.Dataset.InsoleModule import InsoleModule
import torch.nn.functional as F
from prettytable import PrettyTable
from lib.tools.tools import L1, L2, vec_show, press_show, contact_show



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--num_threads', type=int)
    parser.add_argument('--batch_size', default=1,type=int)
    parser.add_argument('--gpus', type=str, default='cpu', help='gpu ids: e.g. 0  0,1,2, 0,2, -1 for CPU mode')
    parser.add_argument('--model', type=str, default='Vector', help='Model to choose: Vector--F2; FP--F1')
    arg = parser.parse_args()

    cfg = config()
    cfg.load(arg.config)
    cfg = cfg.get_cfg()
    cfg.defrost()
    cfg.num_threads = arg.num_threads
    cfg.gpus = arg.gpus
    cfg.batch_size = arg.batch_size
    cfg.networks.model = arg.model
    cfg.trainer.model = arg.model
    cfg.freeze()
    ic(cfg)

   
    
    os.makedirs(cfg.result_path, exist_ok=True)
    os.makedirs('%s/%s' % (cfg.result_path, cfg.name), exist_ok=True)

    dataset_bsc = make_dataset.make_dataset(cfg.dataset, phase='test')
    dataloader = DataLoader(dataset_bsc, batch_size=cfg.batch_size, shuffle=False,
               num_workers=cfg.num_threads, pin_memory=cfg.dataset.pin_memory)
    

    gc_net = make_network.make_network(cfg.networks)

    if cfg.gpus != 'cpu':
        gpu_ids = [int(i) for i in cfg.gpus.split(',')]
        device = torch.device('cuda:%d' % gpu_ids[0])
        gc_net = gc_net.to(device)
        gc_net = DataParallel(gc_net, gpu_ids)
    else:
        gpu_ids = None
        device = torch.device('cpu')

    load_net_checkpoint = cfg.load_net_checkpoint[0] if arg.model=='Vector' else cfg.load_net_checkpoint[1]
    if os.path.exists(load_net_checkpoint):
        print('gc_net : loading from %s' % load_net_checkpoint)
        gc_net.load_state_dict(torch.load(load_net_checkpoint))
        gc_net.eval()
    else:
        print('can not find checkpoint %s'% load_net_checkpoint)

    # input('You have download network!')
    
    m_insole = InsoleModule('/data/PressureDataset')
    isDebug = True if sys.gettrace() else False
    gc_net.eval()
    for data in tqdm(dataloader):    
        # 获取图片路径
        image_path = data['case_name'][0]
        # data_id,sub_ids,seq_name = (image_path).split('/')
        
        for data_item in ['features']:
            data[data_item] = data[data_item].to(device=device)
        
        if arg.model == 'Vector':
            pred_vec = gc_net(x=data['keypoints_f'])
            # loss = L2(data, pred_vec)
            vec_show(data, pred_vec)
        

        elif arg.model == 'FP':
            # pred_contact, pred_press = gc_net(keypoints=data['features'])
            # pred_contact, pred_press = pred_contact.detach().cpu(), pred_press.detach().cpu()
            pred_contact = gc_net(keypoints=data['features'])
            pred_contact = pred_contact.detach().cpu()
            # loss = L1(data, pred)
            contact_show(data, pred_contact)
            # press_show(data, pred_press)
    
            
        else:
            print('Your model is wrong!')
            assert 0
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
       