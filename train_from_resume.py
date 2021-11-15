#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   train_from_resume.py
@Time    :   2021/09/03 11:18:23
@Author  :   Tang Yujin 
@Version :   1.0
@Contact :   tangyujin0275@gmail.com
'''
#from dataset.transforms import RandomFlip,RandomContrast,RandomRotate
from functools import partial
from nibabel.filename_parser import _endswith
from numpy.lib.function_base import append

import torch.nn as nn
import numpy as np
#from torch import optim
from torch.autograd import Variable
from torch.optim import optimizer

from dataset.fracnet_dataset import FracNetTrainDataset
from dataset import transforms as tsfm
from utils.metrics import dice, recall, precision, fbeta_score
from model.unet import UNet
from model.losses import MixLoss, DiceLoss
import torch, os

from torch.utils.tensorboard import SummaryWriter



# path
train_image_dir = '/mntntfs/med_data1/tangyujin/001RibFrac_dataset/ribfrac-train-images/Part1'
train_label_dir = '/mntntfs/med_data1/tangyujin/001RibFrac_dataset/ribfrac-train-labels/Part1'
val_image_dir = '/mntntfs/med_data1/tangyujin/001RibFrac_dataset/ribfrac-val-images'
val_label_dir = '/mntntfs/med_data1/tangyujin/001RibFrac_dataset/ribfrac-val-labels'
model_dir = './ckpt_200_train_from_resume/' 

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# paras
batch_size = 4
num_workers = 4
num_epoch = 200
start_epoch = 0
use_val = False
RESUME = True
random_state = np.random.RandomState(47)

# optimize
#optimizer = optim.SGD

criterion = MixLoss(nn.BCEWithLogitsLoss(), 0.5, DiceLoss(), 1)

# evaluation
def compute_dsc(A, B):
    A = A.reshape((-1))
    B = B.reshape((-1))
    inter = np.sum(A*B)
    union = (np.sum(A)+np.sum(B))
    return (2*inter)/(union+0.00001)

def compute_sen(A, B):
    A = A.reshape((-1))
    B = B.reshape((-1))
    inter = np.sum(A*B)
   
    return inter/(np.sum(B)+0.000001)

def compute_ppv(A, B):
    A = A.reshape((-1))
    B = B.reshape((-1))
    inter = np.sum(A*B)
   
    return inter/(np.sum(A)+0.000001)

# model setting
model = UNet(1, 1, first_out_channels=16)
model = nn.DataParallel(model.cuda())

optimizer = torch.optim.Adam(model.parameters(), lr= 1e-3, betas=(0.5, 0.999))

# data cd
train_transforms = [
    tsfm.Window(-200, 1000),
    tsfm.MinMaxNorm(-200, 1000),
    #tsfm.RandomFlip(random_state, axis_prob=0.5,axis = 0)
    tsfm.RandomRotate(random_state, angle_spectrum=15, axes=None, mode='reflect', order=0),
    #tsfm.RandomContrast(random_state, alpha=(0.5, 1.5), mean=0.0, execution_probability=0.1)   
    ]
val_transforms = [
tsfm.Window(-200, 1000),
tsfm.MinMaxNorm(-200, 1000)
#tsfm.RandomFlip(random_state, axis_prob=0.5),
#tsfm.RandomRotate(random_state, angle_spectrum=30, axes=None, mode='reflect', order=0),
#tsfm.RandomContrast(random_state, alpha=(0.5, 1.5), mean=0.0, execution_probability=0.1)   
]
# ds_train = FracNetTrainDataset(train_image_dir, train_label_dir,transforms=train_transforms)
# dl_train = FracNetTrainDataset.get_dataloader(ds_train, batch_size, False,num_workers)
# ds_val = FracNetTrainDataset(val_image_dir, val_label_dir,transforms=val_transforms)
# dl_val = FracNetTrainDataset.get_dataloader(ds_val, batch_size, False,num_workers)

# writer 都放在同一个文件夹下 这样同一个tensorboard可以显示四张图
train_writer_dir = os.path.join(model_dir,'log')
if not os.path.exists(train_writer_dir ):
    os.makedirs(train_writer_dir )
val_writer_dir = os.path.join(model_dir,'log')
if not os.path.exists(val_writer_dir):
    os.makedirs(val_writer_dir)
train_dice_writer_dir = os.path.join(model_dir,'log')
if not os.path.exists(train_dice_writer_dir):
    os.makedirs(train_dice_writer_dir)
val_dice_writer_dir = os.path.join(model_dir,'log')
if not os.path.exists(val_dice_writer_dir):
    os.makedirs(val_dice_writer_dir)

train_writer = SummaryWriter(train_writer_dir)
val_writer = SummaryWriter(val_writer_dir)
train_dice_writer = SummaryWriter(train_dice_writer_dir)
val_dice_writer = SummaryWriter(val_dice_writer_dir)

# 定义从断点开始加载
if RESUME:
    path_checkpoint = "/mntntfs/med_data1/tangyujin/UNet3d_exp2/ckpt_200_more_metrics_withoutaug/199_UNet3d_param.pth"  # 断点路径
    checkpoint = torch.load(path_checkpoint)  # 加载断点
 
    model.module.load_state_dict(checkpoint)  # 加载模型可学习参数
    start_epoch = 199  # 设置开始的epoch
# train
# print("START TRAINING...")
for epoch in range(start_epoch,start_epoch+num_epoch):
    # loss_record = torch.tensor(0)
    tarin_loss_list = []
    val_loss_list = []
    train_dice_list = []
    val_dice_list = []
    
    ds_train = FracNetTrainDataset(train_image_dir, train_label_dir,transforms=train_transforms)
    dl_train = FracNetTrainDataset.get_dataloader(ds_train, batch_size, False,num_workers)
    ds_val = FracNetTrainDataset(val_image_dir, val_label_dir,transforms=val_transforms)
    dl_val = FracNetTrainDataset.get_dataloader(ds_val, batch_size, False,num_workers)

    train_loss = []
    val_loss = []
    train_dice = []
    val_dice = []
    # train
    print("START TRAINING...")
    for batch_idx, (image_tuple, label_tuple) in enumerate(dl_train):

        #print("START TRAINING...[%d/%d]"%(batch_idx,len(dl_train)))

        
        
        image_tuple = Variable(image_tuple.cuda())
        label_tuple = Variable(label_tuple.cuda())

        pred_tuple = model(image_tuple)

        train_loss = criterion(pred_tuple,label_tuple)
        tarin_dice = compute_dsc(pred_tuple.detach().cpu().numpy(),label_tuple.cpu().numpy())

        tarin_loss_list.append(train_loss.item())
        train_dice_list.append(tarin_dice)
        # loss_list = Variable(loss_list.cuda())

        # Back propagation
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
    # calculate metrics on validation dataset
    print("START VALIDATION...")
    for batch_idx, (image_tuple, label_tuple) in enumerate(dl_val):
    
        image_tuple = Variable(image_tuple.cuda())
        label_tuple = Variable(label_tuple.cuda())
        

        pred_tuple = model(image_tuple)

        val_loss = criterion(pred_tuple,label_tuple)
        val_dice = compute_dsc(pred_tuple.detach().cpu().numpy(),label_tuple.cpu().numpy())

        val_loss_list.append(val_loss.item())
        val_dice_list.append(val_dice)

    
    train_loss_epoch_mean = torch.mean(torch.FloatTensor(tarin_loss_list))
    val_loss_epoch_mean = torch.mean(torch.FloatTensor(val_loss_list))
    train_dice_epoch_mean = np.mean(train_dice_list)
    val_dice_epoch_mean = np.mean(val_dice_list)

    train_writer.add_scalar('train_loss', train_loss_epoch_mean, epoch+1)
    print("Epoch [%d/%d],  train_loss: %.4f" % 
    (epoch, start_epoch+num_epoch, train_loss_epoch_mean))

    val_writer.add_scalar('val_loss', val_loss_epoch_mean, epoch+1)
    print("Epoch [%d/%d],  val_loss: %.4f" % 
    (epoch, start_epoch+num_epoch, val_loss_epoch_mean))
    # 代码改到了这里！！！
    train_dice_writer.add_scalar('tarin_dice', train_dice_epoch_mean, epoch+1)
    print("Epoch [%d/%d],  train_dice: %.4f" % 
    (epoch, start_epoch+num_epoch, train_dice_epoch_mean))

    val_dice_writer.add_scalar('val_dice', val_dice_epoch_mean, epoch+1)
    print("Epoch [%d/%d],  val_dice: %.4f" % 
    (epoch, start_epoch+num_epoch, val_dice_epoch_mean))
    
    torch.save(model.module.state_dict(), model_dir + repr(epoch) + '_UNet3d_param.pth')

