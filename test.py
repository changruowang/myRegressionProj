#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, sys, glob
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torch
import numpy as np
import utils
from multiprocessing import freeze_support
import dataset
from dataset import noise_class2idx, NoiseLevelRegDataset, save_csv, noise_class2weight
from model_config import MultiOutputModel
import cv2
from utils import multiple_regression_eval, load_image
from my_transform import get_transform, MySelectedCrop
import argparse
from tqdm import tqdm
import visual

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

device = torch.device('cuda:0')

def get_args():
    parser = argparse.ArgumentParser(description='My detection code training based on pytorch!')
    parser.add_argument('--work_dir', default='/home/arc-crw5713/myRegresionProj/ghostnet_0.001_wt_t', help='save out put dir')
    parser.add_argument('--model_name', default='ghostnet', help='se_resnet18')
    parser.add_argument('--use_tta', action="store_true", help='se_resnet18')
    parser.add_argument('--crop_sel',type=str,default='random',help='sel ynoise_loss, strength,cnoise')
    parser.add_argument('--image_dir',type=str,default='/home/arc-crw5713/data/noise_level/',help='')
    parser.add_argument('--patch_size',type=int,default=224,help='')
    parser.add_argument('--test_mode',type=str,default='loader',help='')
    parser.add_argument('--tta_metric',type=str,default='median',help='')
    parser.add_argument('--visual_cat',type=str,default='Ynoise',help='')
    parser.add_argument('--run_times',type=int,default=1,help='')
    
    args = parser.parse_args()
    return args
    

def load_model_weights(model, opt):
    '''加载模型权重 
    '''
    model_path = os.path.join(opt.work_dir, "checkpoints/epoch_best_model.pth")
    model.load_state_dict(torch.load(model_path)['model'])
    print('Successfully load weights from %s'%model_path)

def load_weights(model, work_dir):
    '''加载模型权重
    Args: 
        work_dir: 工作路径 
    '''
    model_path = os.path.join(work_dir, "checkpoints/epoch_best_model.pth")
    model.load_state_dict(torch.load(model_path)['model'])
    print('Successfully load weights from %s'%model_path)

@torch.no_grad()
class MyClassificationTTA(object):
    '''测试专用的类
        将 tta测试的 逻辑进行了封装，输入图片，在该类完成对输入图片的patch采样，预测，以及综合
        决策输出
    '''
    def __init__(self, model, device, use_tta=False, opt=None, metric='mean', out_crop_results=True):
        '''
        Args： 
            model: 模型字典 {'unet': unet_mode, 'resnet': resnet} （可以同时包含多个需要测试的模型）
                这样可以保证每个模型每次测试的是相同的 patch
            use_tta: 是否使用 tta 测试，即是是直接对整张图预测 还是 crop多个patch分别预测后综合
            opt： 配置结构
            metric: 多个结果融合的方法 mean median 取均值或者取中位数
            out_crop_results：是否输出 每个crop 的位置和预测值（如果有gt也会存储gt）, 存为txt格式
                。在visual.py中有解析和可视化结果文件的函数
        '''
        self.tta_transform = MySelectedCrop(opt.patch_size, mode=opt.crop_sel)
        self.post_transform = torchvision.transforms.Compose([
                            torchvision.transforms.ToTensor(),
                            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        assert(type(model) == type({}))
        self.model = model
        self.device = device 
        self.use_tta = use_tta
        self.metric = metric
        self.out_crop_results = out_crop_results
    def get_crop_mounts(self):
        return self.tta_transform.get_crop_mounts()

    def __call__(self, image_path):

        img = load_image(image_path)

        if self.use_tta:
            croped_imgs, poses = self.tta_transform.tta_out(img, image_path)

            imgs = [self.post_transform(img).unsqueeze(0) for img in croped_imgs]
            # imgs = [self.post_transform(img).unsqueeze(0) for img in self.tta_transform.tta_out(img)]
            imgs = torch.cat(imgs, dim=0).to(self.device)

        else:
            imgs = self.post_transform(self.tta_transform(img)).unsqueeze(0).to(self.device)
        
        out_str = {}
       
        for k, m in self.model.items():
            ans = {k:v.cpu() for k,v in m(imgs).items()}

            if self.metric == 'mean':
                re = {k:(torch.sum(v)-torch.min(v)-torch.max(v))/(len(croped_imgs)-2) for k, v in ans.items()}
            elif self.metric == 'median':
                re = {k:torch.median(v, dim=0)[0] for k, v in ans.items()}

            tmp = ['%.3f'%re['Ynoise'], '%.3f'%re['strength']]

            if self.use_tta and self.out_crop_results: 
                y_scores = ans['Ynoise'].numpy().tolist()
                str_scores = ans['strength'].numpy().tolist()
                tmp += ['%.3f_%.3f_%d_%d'%(sy, ss, pos[0], pos[1]) for sy, ss, pos in 
                                                    zip(y_scores, str_scores, poses)]
       
            out_str.update({k:tmp})

        return re, out_str


@torch.no_grad()
def test_on_loader(models, opt):
    '''使用 dataloader 测试， 用于在.h5格式的验证集上计算平均损失，将结果存在 test_loss.txt文件中
    Args： 
        model: 模型字典 {'unet': unet_mode, 'resnet': resnet} （可以同时包含多个需要测试的模型）
        opt： 参数
    '''
    valid_data = NoiseLevelRegDataset(base_path=opt.image_dir, phase='val', class2weight=noise_class2weight,
                                    class2idx=noise_class2idx, transform=get_transform(False, path_size=opt.patch_size, params=opt.crop_sel))

    valid_dateloader = torch.utils.data.DataLoader(valid_data, batch_size=32, shuffle=False, num_workers=0, drop_last=True)
    # multiple_regression_eval(model, valid_dateloader, device)
    all_results = []
    loss_labels = ['Ynoise','strength','loss_all']
    all_loss = {model_name : {k:0.0 for k in loss_labels} for model_name in models.keys()}

    pbar = tqdm(valid_dateloader, total=len(valid_dateloader))

    for batch in pbar:
        
        images = batch['img'].to(device)
        target_labels = batch['labels']
        target_labels = {t: target_labels[t].float().to(device) for t in target_labels}
        

        for m_name, m in models.items():
            ans = m(images)
            _, loss_ans = m.get_loss(ans, target_labels)
            
            for k in loss_labels:
                all_loss[m_name][k] += loss_ans[k]

            # out = model(images)

        # _, losses_dict = model.get_loss(out, target_labels)
        # for k in loss_labels:
        #     all_loss[k] += losses_dict[k]

        # re = torch.stack((out['Ynoise'], out['strength'], target_labels['Ynoise'], target_labels['strength']), dim=1).cpu().numpy().tolist()
        # all_results += [[name]+ans for name, ans in zip(batch['name'], re)]

    for name, ans in all_loss.items():
        all_loss[name] = {k: v/len(valid_dateloader) for k,v in ans.items()}

        f = open(os.path.join(opt.work_dir[name], 'test_loss.txt'),'w')
        for label in loss_labels:
            f.write((label+':%.5f\t')%(all_loss[name][label])) 

        f.close()

    # all_results = np.asarray(all_results)
    # fieldnames = ['path', 'Ynoise', 'strength', 'gt_Ynoise', 'gt_strength']
    # save_csv(all_results, os.path.join(opt.work_dir, 'loader_pred_results.txt'),fieldnames=fieldnames)


@torch.no_grad()
def test_on_imagedir(models, opt):
    '''测试图片文件夹中所有图片，将测试结果存储为 imagedir_pred_results.txt 可以使用 visual.py
        中的可视化函数解析
    Args： 
        model: 模型字典
        opt.image_dir： 包含路片的路径
    '''
    model_tta = MyClassificationTTA(models, device, use_tta=opt.use_tta, opt=opt, metric=opt.tta_metric)
    all_results = {k:[] for k in models.keys()} 
    imgs_path = glob.glob(os.path.join(opt.image_dir, '*[bmp,jpg]'))

    # ### 每张图测试 多次
    imgs_path = [p for p in imgs_path for i in range(opt.run_times)]

    pbar = tqdm(imgs_path, total=len(imgs_path))
    for img_path in pbar:
        (_,file_name) = os.path.split(img_path)
   
        re, res_str = model_tta(img_path)
        
        for k, v in res_str.items():
            all_results[k].append([file_name] + v)
            
    p_mounts = model_tta.get_crop_mounts()
    fieldnames = ['path', 'Ynoise', 'strength'] + ['p%d'%i for i in range(p_mounts)]

    for model_name, res_str in all_results.items():
        res_str = np.asarray(res_str)
        save_csv(res_str, os.path.join(opt.work_dir[model_name], 'imagedir_pred_results.txt'), 
                                                                    fieldnames=fieldnames)


@torch.no_grad()
def test_on_annotion(models, opt):
    '''测试用 txt 文件标注了的数据集。将测试结果存储为 anno_pred_results.txt 可以使用 visual.py
        中的可视化函数解析
    Args： 
        model: 模型字典
        opt.image_dir： 此时表示 .txt 标注文件的路径
    '''
    anno_path = opt.image_dir
    (base_path, _) = os.path.split(anno_path)

    model_tta = MyClassificationTTA(models, device, use_tta=opt.use_tta, opt=opt, metric=opt.tta_metric)

    all_results = {k:[] for k in models.keys()} 

    images_name, gt_Ynoise, gt_strength, _ = dataset.read_noise_level_annotations(opt.image_dir, name_only=True)

    ### 单张图测试50次
    images_name = [p for p in images_name for i in range(opt.run_times)]
    gt_Ynoise = [p for p in gt_Ynoise for i in range(opt.run_times)]
    gt_strength = [p for p in gt_strength for i in range(opt.run_times)]


    pbar = tqdm(images_name, total=len(images_name))
    for idx, file_name in enumerate(pbar):

        img_path = os.path.join(base_path, 'images', file_name)
       
        # out, crop_results = model_tta(img_path)

        re, res_str = model_tta(img_path)
        
        for k, v in res_str.items():
            all_results[k].append([file_name, gt_Ynoise[idx], gt_strength[idx]] + v)

    p_mounts = model_tta.get_crop_mounts()
    fieldnames = ['path', 'gt_Ynoise', 'gt_strength', 'Ynoise', 'strength'] + ['p%d'%i for i in range(p_mounts)]

    for model_name, res_str in all_results.items():
        res_str = np.asarray(res_str)
        save_csv(res_str, os.path.join(opt.work_dir[model_name], 'anno_pred_results.txt'), 
                                                                    fieldnames=fieldnames)


if __name__ == '__main__':
    args = get_args()
    ### 模型初始化  输入可以有多个模型
    model_names = [str(name) for name in args.model_name.split(",")]

    work_dirs = [args.work_dir%(name)  for name in model_names]

    args.work_dir = {} 
    models = {}
    for model_name, work_dir in zip(model_names, work_dirs): 
        model = MultiOutputModel(2, model_name=model_name, weights=False).to(device)
        load_weights(model, work_dir)
        model.eval()
        models.update({model_name:model})
        args.work_dir.update({model_name:work_dir})

    if args.test_mode == 'dir':
        test_on_imagedir(models, args)
    elif args.test_mode == 'loader':
        test_on_loader(models, args)
    elif args.test_mode == 'txt':
        test_on_annotion(models, args)
    


# @torch.no_grad()
# def test_on_loader(opt):
#     device = torch.device('cuda:0')
#     model = MultiOutputModel(2, model_name=opt.model_name, weights=False).to(device)
#     load_model_weights(model, opt)
#     model.eval()
#     valid_data = NoiseLevelRegDataset(base_path=opt.image_dir, phase='val', class2weight=noise_class2weight,
#                                     class2idx=noise_class2idx, transform=get_transform(False, path_size=64, params=opt.crop_sel))

#     valid_dateloader = torch.utils.data.DataLoader(valid_data, batch_size=64, shuffle=False, num_workers=4, drop_last=True)
#     # multiple_regression_eval(model, valid_dateloader, device)
#     all_results = []
#     loss_labels = ['Ynoise','strength','loss_all']
#     all_loss = {k:0.0 for k in loss_labels}

#     pbar = tqdm(valid_dateloader, total=len(valid_dateloader))

#     for batch in pbar:
        
#         images = batch['img'].to(device)
#         target_labels = batch['labels']
#         target_labels = {t: target_labels[t].float().to(device) for t in target_labels}
        
#         out = model(images)

#         _, losses_dict = model.get_loss(out, target_labels, out_sel=[1,1])
#         for k in loss_labels:
#             all_loss[k] += losses_dict[k]

#         re = torch.stack((out['Ynoise'], out['strength'], target_labels['Ynoise'], target_labels['strength']), dim=1).cpu().numpy().tolist()
#         all_results += [[name]+ans for name, ans in zip(batch['name'], re)]
    
#     all_loss = {k: v/len(valid_dateloader) for k, v in all_loss.items()}
#     all_results = np.asarray(all_results)
#     fieldnames = ['path', 'Ynoise', 'strength', 'gt_Ynoise', 'gt_strength']
#     save_csv(all_results, os.path.join(opt.work_dir, 'pred_results.txt'),fieldnames=fieldnames)

#     f = open(os.path.join(opt.work_dir, 'test_loss.txt'),'w')
#     for label in loss_labels:
#         f.write((label+':%.5f\t')%(all_loss[label])) 
#     f.close()

# @torch.no_grad()
# def test_on_imagedir(opt):
#     device = torch.device('cuda:0')

#     model = MultiOutputModel(2, model_name=opt.model_name, weights=False).to(device)
#     load_model_weights(model, opt)
#     model.eval()
#     model_tta = MyClassificationTTA(model, device, use_tta=opt.use_tta, opt=opt, metric=opt.tta_metric)
#     all_results = []

#     imgs_path = glob.glob(os.path.join(opt.image_dir, '*[bmp,jpg]'))

#     # ### 每张图测试50次
#     # imgs_path = [p for p in imgs_path for i in range(50)]

#     pbar = tqdm(imgs_path, total=len(imgs_path))
#     for img_path in pbar:
#         (_,file_name) = os.path.split(img_path)
   
#         out, crop_results = model_tta(img_path)
        
#         tmp = [file_name, out['Ynoise'].item(), out['strength'].item()] + crop_results
#         all_results.append(tmp)

#     p_mounts = model_tta.get_crop_mounts()
#     fieldnames = ['path', 'Ynoise', 'strength'] + ['p%d'%i for i in range(p_mounts)]
#     all_results = np.asarray(all_results)
#     save_csv(all_results, os.path.join(opt.work_dir, 'imagedir_pred_results.txt'), fieldnames=fieldnames)
#     # all_results = np.asarray(all_results)
#     # fieldnames = ['path', 'Ynoise', 'strength']
#     # save_csv(all_results, os.path.join(opt.work_dir, 'pred_results.txt'), fieldnames=fieldnames)




# class MyClassificationTTA(object):
#     def __init__(self, model, device, use_tta=False, opt=None, metric='mean', out_crop_results=True):
#         self.tta_transform = MySelectedCrop(opt.patch_size, mode=opt.crop_sel)
#         self.post_transform = torchvision.transforms.Compose([
#                             torchvision.transforms.ToTensor(),
#                             torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
#         self.model = model
#         self.device = device 
#         self.use_tta = use_tta
#         self.metric = metric
#         self.out_crop_results = out_crop_results
#     def get_crop_mounts(self):
#         return self.tta_transform.get_crop_mounts()

#     def __call__(self, image_path):

#         img = load_image(image_path)

#         if self.use_tta:
#             croped_imgs, poses = self.tta_transform.tta_out(img)
#             imgs = [self.post_transform(img).unsqueeze(0) for img in croped_imgs]
#             # imgs = [self.post_transform(img).unsqueeze(0) for img in self.tta_transform.tta_out(img)]
#             imgs = torch.cat(imgs, dim=0).to(self.device)

#         else:
#             imgs = self.post_transform(self.tta_transform(img)).unsqueeze(0).to(self.device)
        
#         out = self.model(imgs) 

#         out = {k:v.cpu() for k,v in out.items()}

#         if self.metric == 'mean':
#             re = {k:(torch.sum(v)-torch.min(v)-torch.max(v))/(len(croped_imgs)-2) for k, v in out.items()}
#         elif self.metric == 'median':
#             re = {k:torch.median(v, dim=0)[0] for k, v in out.items()}
#         crop_results = None
#         ### 只能输出一个类别的结果
#         if self.use_tta and self.out_crop_results:
#             y_scores = out['Ynoise'].numpy().tolist()
#             str_scores = out['strength'].numpy().tolist()
#             crop_results = ['%.3f_%.3f_%d_%d'%(sy, ss, pos[0], pos[1]) for sy, ss, pos in zip(y_scores, str_scores, poses)]
        
#         return re, crop_results



# @torch.no_grad()
# def test_on_annotion(opt):
#     anno_path = opt.image_dir
#     (base_path, _) = os.path.split(anno_path)

#     device = torch.device('cuda:0')

#     model = MultiOutputModel(2, model_name=opt.model_name, weights=False).to(device)
#     load_model_weights(model, opt)
#     model.eval()
#     model_tta = MyClassificationTTA(model, device, use_tta=opt.use_tta, opt=opt, metric=opt.tta_metric)
#     all_results = []

#     images_name, gt_Ynoise, gt_strength, _ = dataset.read_noise_level_annotations(opt.image_dir, name_only=True)

#     # ### 单张图测试50次
#     # images_name = [p for p in images_name for i in range(50)]
#     # gt_Ynoise = [p for p in gt_Ynoise for i in range(50)]

#     pbar = tqdm(images_name, total=len(images_name))
#     for idx, file_name in enumerate(pbar):
#         img_path = os.path.join(base_path, 'images', file_name)
       
#         # img = load_image(img_path, transform=trans)
#         out, crop_results = model_tta(img_path)
       
#         tmp = [file_name, out['Ynoise'].item(), out['strength'].item()] 
#         tmp += [gt_Ynoise[idx], gt_strength[idx]] + crop_results
#         all_results.append(tmp)

#     p_mounts = model_tta.get_crop_mounts()
#     fieldnames = ['path', 'Ynoise', 'strength', 'gt_Ynoise', 'gt_strength'] + ['p%d'%i for i in range(p_mounts)]

#     all_results = np.asarray(all_results)
    
#     save_csv(all_results, os.path.join(opt.work_dir, 'anno_pred_results.txt'), fieldnames=fieldnames)


