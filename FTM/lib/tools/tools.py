import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import time
    
def Get_xy(vector):
    vector = vector.reshape(-1,2)
    x = (120*(1-vector[:,0])).to(dtype=torch.int)
    y = (160*(1-vector[:,1])).to(dtype=torch.int)
    
    x = torch.clamp(x, min=0, max=120)
    y = torch.clamp(y, min=0, max=159)
    return x, y

def L1(data, pred_press, pred_contact):
    """
    Get L1 loss for FPCNet
    """
    device = pred_contact.device
    
    b, seqlen, l, h, w = pred_contact.shape
    
    # pred_press_rs, pred_press_ls = pred_press[:,:,0,:,:], pred_press[:,:,1,:,:]
    pred_cont_rs, pred_cont_ls = pred_contact[:,:,0,:,:], pred_contact[:,:,1,:,:]

    # pred_press_rs, pred_press_ls = pred_press_ls.reshape(-1, 48, 48), pred_press_rs.reshape(-1, 48, 48)
    pred_cont_rs, pred_cont_ls = pred_cont_ls.reshape(-1, 48, 48), pred_cont_rs.reshape(-1, 48, 48)
    
    # gt_press_imgs = data['insole'].reshape(-1, 208, 168)
    gt_cont_imgs = data['contact'].reshape(-1, 208, 168)
    
    rxs, rys = Get_xy(data['vector_r'])
    lxs, lys = Get_xy(data['vector_l'])
    
    loss_press, loss_cont = [], []
    # for pred_press_r, pred_press_l, pred_cont_r, pred_cont_l, \
    #     rx, ry, lx, ly, gt_press_img, gt_cont_img in \
    #     zip(pred_press_rs, pred_press_ls, pred_cont_rs, pred_cont_ls, rxs, rys, lxs, lys, gt_press_imgs, gt_cont_imgs):
    for pred_cont_r, pred_cont_l, \
        rx, ry, lx, ly, gt_cont_img in \
        zip(pred_cont_rs, pred_cont_ls, rxs, rys, lxs, lys, gt_cont_imgs):

        # Record Nonzero and Zero Positions
        # nonzero = torch.nonzero(gt_cont_img)
        # zero = torch.nonzero(gt_cont_img == 0)
        
        # Get Pred image
        # pred_press_img = torch.zeros_like(gt_press_img).to(device=device)
        # pred_press_img[ry: ry+48, rx: rx+48] += pred_press_r
        # pred_press_img[ly: ly+48, lx: lx+48] += pred_press_l
        
        pred_cont_img = torch.zeros_like(gt_cont_img).to(device=device)
        pred_cont_img[ry: ry+48, rx: rx+48] += pred_cont_r
        pred_cont_img[ly: ly+48, lx: lx+48] += pred_cont_l
        
        # Mask
        mask_img = torch.zeros_like(gt_cont_img).to(device=device)+1e-10
        mask_r = torch.ones_like(pred_cont_r)
        mask_l = torch.ones_like(pred_cont_l)
        mask_img[ry: ry+48, rx: rx+48] += mask_r
        mask_img[ly: ly+48, lx: lx+48] += mask_l
        # pred_press_img = pred_press_img/mask_img
        pred_cont_img = pred_cont_img/mask_img
        
        ## Press loss
        # loss1
        # loss1_weight = 1
        # loss1 = F.mse_loss(gt_press_img[zero[:,0], zero[:,1]], pred_press_img[zero[:,0], zero[:,1]])
        # # loss2
        # loss2_weight = 1.2
        # loss2 = F.mse_loss(gt_press_img[nonzero[:,0], nonzero[:,1]], pred_press_img[nonzero[:,0], nonzero[:,1]]) if nonzero.numel()!=0 else 0
        
        # loss_press.append(loss1*loss1_weight+ loss2*loss2_weight)
        
        ## Contact loss
        loss = F.binary_cross_entropy(target=gt_cont_img, input=pred_cont_img)
        loss_cont.append(loss)
        
        
    # loss_press = torch.stack(loss_press, dim=0)
    loss_cont = torch.stack(loss_cont, dim=0)
    # return loss_press.mean(), loss_cont.mean()
    return loss_cont.mean()

def L2(data, pred_vec):
    """
    Get L2 loss for FVecNet
    """
    gt_vec = torch.cat([data['vector_r'], data['vector_l']], dim=-1)
    loss = F.mse_loss(gt_vec, pred_vec)
    return loss    

def L3(data, pred_press, pred_contact):
    """
    Improve L1 Calculate speed
    """
    device = pred_press.device
    b, seqlen, l, h, w = pred_contact.shape
    
    # Reshape predicted pressure and contact maps
    pred_press_rs = pred_press[:, :, 0, :, :].reshape(-1, 48, 48)
    pred_press_ls = pred_press[:, :, 1, :, :].reshape(-1, 48, 48)
    pred_cont_rs = pred_contact[:, :, 0, :, :].reshape(-1, 48, 48)
    pred_cont_ls = pred_contact[:, :, 1, :, :].reshape(-1, 48, 48)

    gt_press_imgs = data['insole'].reshape(-1, 208, 168).to(device)
    gt_cont_imgs = data['contact'].reshape(-1, 208, 168).to(device)

    # Get x, y coordinates for placing predictions
    rxs, rys = Get_xy(data['vector_r'])
    lxs, lys = Get_xy(data['vector_l'])

    # Initialize pred images
    pred_press_imgs = torch.zeros_like(gt_press_imgs, device=device) 
    pred_cont_imgs = torch.zeros_like(gt_cont_imgs, device=device)

    # Loop over each element in batch to place predicted values
    for i in range(pred_press_rs.size(0)):
        # Ensure coordinates are within bounds
        ry, rx = rys[i].item(), rxs[i].item()
        ly, lx = lys[i].item(), lxs[i].item()
        
        # Place predicted pressure and contact
        pred_press_imgs[i, ry: ry + 48, rx: rx + 48] += pred_press_rs[i]
        pred_press_imgs[i, ly: ly + 48, lx: lx + 48] += pred_press_ls[i]
        pred_cont_imgs[i, ry: ry + 48, rx: rx + 48] += pred_cont_rs[i]
        pred_cont_imgs[i, ly: ly + 48, lx: lx + 48] += pred_cont_ls[i]

    # Divide by mask
    mask_img = torch.zeros_like(gt_press_imgs, device=device).detach() + 1e-10
    for i in range(pred_press_rs.size(0)):
        ry, rx = rys[i].item(), rxs[i].item()
        ly, lx = lys[i].item(), lxs[i].item()
        
        mask_img[i, ry: ry + 48, rx: rx + 48] += 1
        mask_img[i, ly: ly + 48, lx: lx + 48] += 1

    pred_press_imgs /= mask_img
    pred_cont_imgs /= mask_img

    # Compute Press loss
    zero_mask = (gt_cont_imgs == 0).detach()
    nonzero_mask = ~zero_mask

    loss1 = F.mse_loss(gt_press_imgs[zero_mask], pred_press_imgs[zero_mask])
    loss2 = F.mse_loss(gt_press_imgs[nonzero_mask], pred_press_imgs[nonzero_mask]) if nonzero_mask.sum() > 0 else 0

    loss_press = loss1 + 1.2 * loss2

    # Compute Contact loss
    loss_cont = F.binary_cross_entropy(input=pred_cont_imgs, target=gt_cont_imgs)

    return loss_press.mean(), loss_cont.mean()

    


def vec_show(data, pred_vec):
    """
    Show an image generated by FVecNet
    """
    # Get Name
    names = data['case_name']
    base, num = os.path.split(names[0]) 
    
    # Get pred x,y
    pred_vec_r, pred_vec_l = pred_vec.squeeze()[:, :2], pred_vec.squeeze()[:, 2:]
    pred_rxs, pred_rys = Get_xy(pred_vec_r)
    pred_lxs, pred_lys = Get_xy(pred_vec_l)
    
    # Get GT x,y
    gt_vec_r, gt_vec_l = data['vector_r'].squeeze(), data['vector_l'].squeeze()
    gt_rxs, gt_rys = Get_xy(gt_vec_r)
    gt_lxs, gt_lys = Get_xy(gt_vec_l)
    
    # Draw an image
    for index in range(len(gt_rxs)):
        pred_rx, pred_ry = pred_rxs[index], pred_rys[index]
        pred_lx, pred_ly = pred_lxs[index], pred_lys[index]
        
        gt_rx, gt_ry = gt_rxs[index], gt_rys[index]
        gt_lx, gt_ly = gt_lxs[index], gt_lys[index]
        
        # print('===========================')
        # print(f"GT:({gt_rx},{gt_ry})  Pred:({pred_rx},{pred_ry})")
        # print(f"GT:({gt_lx},{gt_ly})  Pred:({pred_lx},{pred_ly})")
        
        img = np.zeros((160, 120))  # RGB  
        img[pred_ry, pred_rx] = 255  
        img[pred_ly, pred_lx] = 255    
        img[gt_ry, gt_rx] = 100      
        img[gt_ly, gt_lx] = 100      

        # 确保保存的目录存在
        os.makedirs('./visual', exist_ok=True)

        # 保存图像，确保保存为 RGB 格式
        num = int(num)
        plt.imsave(os.path.join('./visual', f'{base}_{num * 40 + index}.jpg'), img, cmap='gray')
        # time.sleep(1)
    
def contact_show(data, pred_cont):
    # Get Name
    names = data['case_name']
    base, num = os.path.split(names[0]) 
    num = int(num)
    
    pred_cont  = pred_cont.numpy()
    pred_rs, pred_ls = pred_cont[:,:,0,:,:], pred_cont[:,:,1,:,:]
    
    b, seqlen, h, w = pred_ls.shape
    
    pred_ls, pred_rs = pred_ls.reshape(-1, 48, 48), pred_rs.reshape(-1, 48, 48)
    
    gt_imgs = data['contact'].reshape(-1, 208, 168)
    rxs, rys = Get_xy(data['vector_r'])
    lxs, lys = Get_xy(data['vector_l'])
    
    for index in range(b*seqlen):
        # Get pred image
        pred_r, pred_l = pred_rs[index], pred_ls[index]
        
        # Get position
        rx, ry = int(rxs[index]), int(rys[index])
        lx, ly = int(lxs[index]), int(lys[index])
        
        # Get GT image
        gt_img = gt_imgs[index]
        
        # Get Pred image
        pred_img = np.zeros_like(gt_img)
        
        # Cast image
        pred_img[ry: ry+48, rx: rx+48] += pred_r
        pred_img[ly: ly+48, lx: lx+48] += pred_l
        
        # Mask
        mask_img = np.zeros_like(gt_img)+1e-10
        mask_r = np.ones_like(pred_r)
        mask_l = np.ones_like(pred_l)
        mask_img[ry: ry+48, rx: rx+48] += mask_r
        mask_img[ly: ly+48, lx: lx+48] += mask_l
        # Get Pred Probabilities
        pred_img = pred_img/mask_img
        pred_img = (pred_img >= 0.50).astype(np.float32)
        
        # Concatenate images
        line = np.ones((len(pred_img), 10))
        img = np.concatenate([gt_img, line, pred_img], axis=-1)
        
        # Save
        os.makedirs('./visual/contact/', exist_ok=True)
        plt.imsave(os.path.join('./visual/contact/', f'{base}_' + '{:05d}.jpg'.format(num*40+index)), img, cmap='gray')

def press_show(data, pred_press):
    # Get Name
    names = data['case_name']
    base, num = os.path.split(names[0]) 
    num = int(num)
    
    pred_press  = pred_press.numpy()
    pred_rs, pred_ls = pred_press[:,:,0,:,:], pred_press[:,:,1,:,:]
    
    b, seqlen, h, w = pred_ls.shape
    
    pred_ls, pred_rs = pred_ls.reshape(-1, 48, 48), pred_rs.reshape(-1, 48, 48)
    
    gt_imgs = data['insole'].reshape(-1, 208, 168)
    rxs, rys = Get_xy(data['vector_r'])
    lxs, lys = Get_xy(data['vector_l'])
    
    for index in range(b*seqlen):
        # Get pred image
        pred_r, pred_l = pred_rs[index], pred_ls[index]
        
        # Get position
        rx, ry = int(rxs[index]), int(rys[index])
        lx, ly = int(lxs[index]), int(lys[index])
        
        # Get GT image
        gt_img = gt_imgs[index]
        
        # Get Pred image
        pred_img = np.zeros_like(gt_img)
        
        # Cast image
        pred_img[ry: ry+48, rx: rx+48] += pred_r
        pred_img[ly: ly+48, lx: lx+48] += pred_l

        # Mask
        mask_img = np.zeros_like(gt_img)+1e-10
        mask_r = np.ones_like(pred_r)
        mask_l = np.ones_like(pred_l)
        mask_img[ry: ry+48, rx: rx+48] += mask_r
        mask_img[ly: ly+48, lx: lx+48] += mask_l
        # Get Pred Probabilities
        pred_img = pred_img/mask_img
        
        # Concatenate images
        line = np.ones((len(pred_img), 10))
        img = np.concatenate([gt_img, line, pred_img], axis=-1)
        
        # Save
        os.makedirs('./visual/FP', exist_ok=True)
        plt.imsave(os.path.join('./visual/FP', f'{base}_' + '{:05d}.jpg'.format(num*40+index)), img, cmap='gray')
