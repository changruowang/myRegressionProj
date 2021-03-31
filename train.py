#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, sys
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torch
import utils
# from multiprocessing import freeze_support
import cv2
from visual import imshow
from utils import MetricLogger, multiple_regression_eval
from tensorboardX import SummaryWriter
import argparse
from model_config import MultiOutputModel
from dataset import HDF5_DataSet, noise_class2idx, noise_class2weight, NoiseLevelRegDataset
from my_transform import get_transform
import csv

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

def get_args():
    parser = argparse.ArgumentParser(description='My classification code training based on pytorch!')
    parser.add_argument('--work_dir', default='/home/arc-crw5713/myRegresionProj/test1', help='save out put dir')
    parser.add_argument('--data_dir', type=str, default='/home/arc-crw5713/data/balance_crop_gray', help='')
    parser.add_argument('--model_name', default='unet', help='se_resnet18')
    parser.add_argument('--milestones', default='50,200,300', help='')
    parser.add_argument('--epochs', default=500, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--use_loss_weights',type=str2bool,nargs='?',const=False,help='Turn on or turn off loss weights')
    parser.add_argument('--output_sel',type=str,default='1,0,0',help='sel ynoise_loss, strength,cnoise')
    parser.add_argument('--crop_sel',type=str,default='random',help='sel ynoise_loss, strength,cnoise')
    parser.add_argument('--patch_size',type=int,default='64',help='')


    parser.add_argument('--device', default='cuda:0', help='device')
    parser.add_argument('--b', '--batch_size', default=2, type=int)
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 8)')
    parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', default=0.0005, type=float,dest='wd',
                         help='weight decay (default: 0.0005)')
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
  
    parser.add_argument('--resume', default=None, help='resume from checkpoint')
    parser.add_argument('--test_only', action='store_true', dest='test_only', help='test only', default=False)
    parser.add_argument('--log_dir', default='./log_dir', help='path where to save')


    args = parser.parse_args()

    return args

def classification_train():
    
    args = get_args()
    work_dir = args.work_dir
    utils.mkdir(work_dir)
    model_dir = os.path.join(work_dir, "checkpoints")
    log_dir = os.path.join(work_dir, "train_loss")
    resum_model_path = os.path.join(work_dir, "checkpoints/epoch_{epoch:}_model.pth".format(epoch=str(args.resume)))
    milestones = [int(num) for num in args.milestones.split("_")]
    out_sel = [float(num) for num in args.output_sel.split("_")]
    # print(milestones)

    utils.mkdir(log_dir)
    utils.mkdir(model_dir)
    
    cuda_device = torch.device("cuda:0")

    tf_logger = SummaryWriter(log_dir)

####数据集准备   '/home/arc-crw5713/data/noise_level'
    train_data = HDF5_DataSet(db_path=args.data_dir, phase='train', 
                            transform=get_transform(True, args.patch_size, params=args.crop_sel))
    valid_data = HDF5_DataSet(db_path=args.data_dir, phase='val', 
                            transform=get_transform(False, args.patch_size, params=args.crop_sel))

    train_dateloader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
    valid_dateloader = torch.utils.data.DataLoader(valid_data, batch_size=64, shuffle=False, num_workers=4, drop_last=True, pin_memory=False)


####模型准备
    model = MultiOutputModel(2, model_name=args.model_name, weights=args.use_loss_weights, loss_sel=out_sel)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    
####Resum training
    resum_model_path = os.path.join('./checkpoints/epoch_{epoch:}_model.pth'.format(epoch=str(args.resume)))
    if os.path.exists(resum_model_path):
        print('training start from args.resume...')
        resum_dict = torch.load(resum_model_path)
        model.load_state_dict(resum_dict['model'])
        optimizer.load_state_dict(resum_dict['optimizer'])
        lr_scheduler.load_state_dict(resum_dict['lr_scheduler'])
    model.to(cuda_device)

#### 记录训练过程
    print('Start regression training!')
    metric_logger = MetricLogger(delimiter='  ', logger=tf_logger)
    metric_logger.add_meter('lr', window_size=1, fmt='{value:.6f}')

    f = open(os.path.join(work_dir, 'eval.txt'),'w')
    min_eval_loss = 100
    # re = multiple_regression_eval(model, valid_dateloader,device=cuda_device, f=f)

    num_epochs = args.epochs
    for epoch in range(num_epochs):
        model.train()
        for batch  in metric_logger.log_every(train_dateloader, args.print_freq, epoch):

            images = batch['img'].to(cuda_device)
            target_labels = batch['labels']
            target_labels = {t: target_labels[t].float().to(cuda_device) for t in target_labels}

            out = model(images)
            loss, losses_dict = model.get_loss(out, target_labels)
           
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            metric_logger.update(**losses_dict)
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        lr_scheduler.step()
#### 计算在验证集上的损失
        # if epoch % 5 == 0:
        eval_loss = multiple_regression_eval(model, valid_dateloader,device=cuda_device, f=f)['loss_all']
        if eval_loss <= min_eval_loss:
            utils.save_checkpoint({'model': model.state_dict(),
                                'optimizer': optimizer.state_dict(),
                                'lr_scheduler': lr_scheduler.state_dict(),
                                'epoch': epoch}, os.path.join(model_dir, 'epoch_best_model.pth'))
            min_eval_loss = eval_loss  
#### 保存模型
        utils.save_checkpoint({'model': model.state_dict(),
                                'optimizer': optimizer.state_dict(),
                                'lr_scheduler': lr_scheduler.state_dict(),
                                'epoch': epoch}, os.path.join(model_dir, 'epoch_now_model.pth'))
    f.close()
    print('End regression training!')



if __name__ == '__main__':
    classification_train()
   

 