from tqdm import tqdm
import torch
import torch.nn.functional as F
import numpy as np
import time
import os

from lib.tools.tools import L1, L2, L3


class ContTrainer:
    def __init__(self, dataloader, model, optimizer, recorder, gpu_ids,opts=None):
        self.dataloader = dataloader
        self.model = model
        self.optimizer = optimizer
        self.recorder = recorder
        if gpu_ids is None:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda:%d' % gpu_ids[0])
        
        self.to_cuda = ['insole', 'vector_r', 'vector_l', 'features', 'contact']

        self.init_lr = opts.lr
        self.num_train_epochs = opts.num_train_epochs
        self.epochs = opts.epochs

        self.w_press = opts.w_press
        self.w_cont = opts.w_cont

        self.type = opts.model


        
        
    def adjust_learning_rate(self, optimizer, epoch):
        """
        Sets the learning rate to the initial LR decayed by x every y epochs
        x = 0.1, y = args.num_train_epochs = 100
        """
        lr = self.init_lr * (0.1 ** (epoch // self.num_train_epochs))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def train(self):
        self.model.train()
        self.recorder.init()
        for epoch in range(self.epochs):
            iter = 0
            loss_ls, loss_contact_ls, loss_press_ls, loss_vec_ls= [],[],[],[]
            self.adjust_learning_rate(self.optimizer, epoch)
            
            for data in tqdm(self.dataloader):
                self.optimizer.zero_grad()
                log = {}
                loss, loss_contact, loss_press, loss_vec = torch.tensor(0).float(), torch.tensor(0).float(), \
                                                           torch.tensor(0).float(), torch.tensor(0).float()
                
                for data_item in self.to_cuda:
                    data[data_item] = data[data_item].to(device=self.device)
                
                # 针对不同的模型
                if self.type=='Vector':
                    pred_vec = self.model(x=data['keypoints_f'])
                    loss_vec = L2(data, pred_vec)
                    loss = loss_vec

                elif self.type=='FP':
                    
                    # pred_contact, pred_press = self.model(keypoints=data['features'])
                    pred_contact = self.model(keypoints=data['features'])
                    pred_press = None
                    
                    loss_contact = L1(data, pred_press, pred_contact)
                    loss = loss_contact*self.w_cont 
                
                else:
                    print('You have inputed wrong Model name!!')
                    assert 0

                loss.backward()
                
                self.optimizer.step()

                # 记录
                log.update({
                    'data': data,
                    'net' : self.model,
                    'optim' : self.optimizer, 
                    'loss':loss,
                    'loss_contact':loss_contact,
                    'loss_press': loss_press,
                    'loss_vec': loss_vec,
                    'epoch' : epoch,
                    'iter':iter,
                    'img_path': data['case_name'],
                    'type': self.type,
                })
                loss_ls.append(loss), loss_press_ls.append(loss_press)
                loss_contact_ls.append(loss_contact), loss_vec_ls.append(loss_vec)
                self.recorder.logPressNetTensorBoard(log)
                iter+=1
                
            loss_mean = torch.mean(torch.stack(loss_ls, dim=0))
            loss_press_mean = torch.mean(torch.stack(loss_press_ls, dim=0))
            loss_contact_mean = torch.mean(torch.stack(loss_contact_ls, dim=0))
            loss_vec_mean = torch.mean(torch.stack(loss_vec_ls, dim=0))
            
            print('Epoch[%d]: total loss[%f], Press loss:[%f], Contact loss:[%f], Vec loss:[%f]'%(
                epoch,loss_mean, loss_press_mean, loss_contact_mean, loss_vec_mean))
            self.recorder.log(log)
            
            

