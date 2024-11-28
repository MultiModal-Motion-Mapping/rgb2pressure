from lib.config.config import config_cont as config
import argparse
import torch
from torch.nn import DataParallel
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
from torch.utils.data import DataLoader
from icecream import ic

from lib.Dataset import make_dataset
from lib.Networks import make_network
from lib.Trainer import make_trainer
from lib.Record.record import ContRecorder

'''
    我们将对网络层做一些修剪
'''




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--num_threads', type=int)
    parser.add_argument('--batch_size', type=int)
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
    
    print(f'Now you are training {arg.model}!')
    
    os.makedirs(cfg.checkpoint_path, exist_ok=True)
    for name in cfg.name:
        os.makedirs('%s/%s' % (cfg.checkpoint_path, name), exist_ok=True)
        os.makedirs('%s/%s' % (cfg.result_path, name), exist_ok=True)

    device = None

    dataset_bsc = make_dataset.make_dataset(cfg.dataset, phase='train')
    
    dataloader = DataLoader(dataset_bsc, batch_size=cfg.batch_size, shuffle=not cfg.dataset.serial_batches,
                            num_workers=cfg.num_threads, pin_memory=cfg.dataset.pin_memory)
    
    
    gc_net = make_network.make_network(cfg.networks)
    
    if cfg.gpus != 'cpu':
        gpu_ids = [int(i) for i in cfg.gpus.split(',')]
        device = torch.device("cuda")
        gc_net = DataParallel(gc_net)
        gc_net = gc_net.to(device)
    else:
        gpu_ids = None

    
    optimizer = torch.optim.Adam(gc_net.parameters(), lr=cfg.trainer.lr)
    recorder = ContRecorder(cfg)
   
    trainer = make_trainer.make_trainer(dataloader, gc_net, optimizer, recorder, gpu_ids, cfg.trainer)
    
    trainer.train()
    print("=========================Training completed!!!!!=========================")

