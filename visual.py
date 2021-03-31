#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import utils
from PIL import Image, ImageDraw, ImageFont
from dataset import read_noise_level_annotations
import matplotlib.pylab as plt
from torchvision.transforms.transforms import ToPILImage
import numpy as np
import cv2
import os
import pandas as pd
import csv

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


def imshow(inp, labels=None):
    title = []
    if labels is not None:
        for i in range(labels.shape[0]):
            print(i)
            title.append(label_map[labels[i].item()])

    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

def paint_chinese_opencv(im,chinese,pos,color,font_scale=0.5):  ## locate *.ttc 查找字体位置
    '''在图上写字符串 为了支持中文 所以单独写了一个函数 
    Args: 
        im: 输入 numpy 格式图片
        chinese： 要显示的字符串
        pos：字符串在图上显示的位置 （x,y）
        color: 字体颜色 (b,g,r)
        font_scale: 字体大小 
    '''
    b,g,r = color

    img_PIL = Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))
    # font = ImageFont.truetype("simhei.ttf", int(font_scale*24), encoding="utf-8")  
    fillColor = (r,g,b) #(255,0,0)
    position = pos #(100,100)
    
    draw = ImageDraw.Draw(img_PIL)
    draw.text(position,chinese,fill=fillColor)
 
    img = cv2.cvtColor(np.asarray(img_PIL),cv2.COLOR_RGB2BGR)
    return img

### 未用
def visual_pred_results(result_file, images_dir):
    with open(result_file) as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            img_path = os.path.join(images_dir, row[ 'path'])
            re_ynoise = float(row['Ynoise'])
            re_strength = float(row['strength'])

            cat_str = 'pred - Ynoise:%.3f, strength:%.3f\n'%(re_ynoise, re_strength)

            if 'gt_Ynoise' in row.keys():
                ynoise = row['gt_Ynoise']
                strength = row['gt_strength']
                cat_str += 'label - Ynoise:%s, strength:%s\n'%(ynoise, strength)
            
            image = cv2.imread(img_path)
            image = paint_chinese_opencv(image, cat_str, (0,0), (0,0,255), font_scale=1)
            cv2.imshow('result', image)
            key = chr(cv2.waitKey(0))
            if key == '\x1b':   ## esc 按钮退出
                break
## 未用
def visual_datasets(annotion_path):
    (base_path,_) = os.path.split(annotion_path)
    images_name, Ynoise, strength, Cnoise = read_noise_level_annotations(annotion_path)
    g_window_name = 'image'

    for idx in range(len(images_name)):
        image = cv2.imread(os.path.join(base_path,'images',images_name[idx]))
        cat_str = Ynoise[idx] + '_' + strength[idx] + '_' + Cnoise[idx]
        image = paint_chinese_opencv(image, cat_str, (0,0), (0,0,255), font_scale=2)
        
        print(images_name[idx], Ynoise[idx], strength[idx], Cnoise[idx])

        cv2.imshow(g_window_name, image)
        key = chr(cv2.waitKey(0))
        if key == '\x1b':   ## esc 按钮退出
            break

def visual_label_distribute(annotion_path):
    '''可视化数据集中各个类别的样本分布
    Args: 
        annotion_path：gt文件的路径
    '''
    # 颜色
    color = sns.color_palette()
    # 数据print精度
    pd.set_option('precision',3) 
    # _, Ynoise, strength, Cnoise = read_noise_level_annotations(annotion_path)
    df = pd.read_csv(annotion_path, delimiter = '\t')
    df.info()
    print(df.describe())

    plt.figure(figsize = (15,3))

    for i, k in enumerate(['Ynoise', 'strength', 'Cnoise']):
        #print(k)
        plt.subplot(1,3,i+1)
        df[k].value_counts().plot(kind = 'bar', color = color[0])
        plt.xticks(rotation=0)
        plt.xlabel(k, fontsize = 12)
        plt.ylabel('Frequency', fontsize = 12)
    plt.tight_layout()
    plt.show()

## 未用 
def visual_board_log():
    from tensorboard.backend.event_processing import event_accumulator
    #加载日志数据
    ea=event_accumulator.EventAccumulator('/home/arc-crw5713/myClassification_torch/efficientnet-b0_0.001_wt_t_attr_1,1,0/train_loss/all loss/Cnoise/events.out.tfevents.1611220310.arc-crw5713-arn7n8-64dcf9c978-rlh2k') 
    ea.Reload()
    print(ea.scalars.Keys())

    val_acc=ea.scalars.Items('all_loss')
    print(len(val_acc))
    print([(i.step,i.value) for i in val_acc])

    import matplotlib.pyplot as plt
    fig=plt.figure(figsize=(6,4))
    ax1=fig.add_subplot(111)
    val_acc=ea.scalars.Items('all_loss')
    ax1.plot([i.step for i in val_acc],[i.value for i in val_acc],label='all_loss')
    ax1.set_xlim(0)
    acc=ea.scalars.Items('all_loss')
    ax1.plot([i.step for i in acc],[i.value for i in acc],label='all_loss')
    ax1.set_xlabel("step")
    ax1.set_ylabel("")

    plt.legend(loc='lower right')
    plt.savefig('result.png')


def visual_crop_results2(result_file, images_dir, save_path):
    ''' 可视化 patch 预测结果，根据结果文件中的块的的位置和得分在 图中标出
    Args:
        result_file：记录图片名字，对应的crop的块的位置，对应的 预测结果 .txt
        images_dir：图片的存放途径
        save_path：输出图片存储位置
    '''
    with open(result_file) as f:
        reader = csv.DictReader(f, delimiter='\t')
        pm = -1
        
        for k in reader.fieldnames:
            if k[0] == 'p':
                pm = pm+1

        for row in reader:    
            img_path = os.path.join(images_dir, row[ 'path'])
            image = cv2.imread(img_path)

            ans = [row['p%d'%i].split('_') for i in range(pm)]

            strength_crops_re = []
            ynoise_crops_re = []

            for crop in ans:
                if crop[0] =='':
                    continue
                # s = str(crop[0]) if visual_cat=='Ynoise' else str(crop[1])
                s = str(crop[0]) + '/' + str(crop[1])
                x = int(crop[2])
                y = int(crop[3])

                ynoise_crops_re.append(float(crop[0]))
                strength_crops_re.append(float(crop[1]))

                cv2.rectangle(image, (x,y), (x+64,y+64), (0,255,0), 1)
                image = paint_chinese_opencv(image, s, (x+4, y-32), (0,0,255), font_scale=1)

            ynoise_crops_re, strength_crops_re = np.array(ynoise_crops_re), np.array(strength_crops_re)
            var_y, var_s = np.std(ynoise_crops_re), np.std(strength_crops_re)

            cat_str = 'Ynoise/strength\n'
            cat_str += 'pre: %.3f/%.3f\n'%(float(row['Ynoise']), float(row['strength']))
            if 'gt_Ynoise' in row.keys():
                cat_str += 'gt: %s/%s\n'%(row['gt_Ynoise'], row['gt_strength'])
            cat_str += 'var: %.3f/%.3f\n'%(var_y, var_s)

            image = paint_chinese_opencv(image, cat_str, (0,0), (0,0,255), font_scale=1)
            cv2.imwrite(os.path.join(save_path,'crops_re_'+row['path']), image)
            # cv2.imshow('result', image)
            # key = chr(cv2.waitKey(0))
            # if key == '\x1b':   ## esc 按钮退出
            #     break

from dataset import noise_class2idx　
def visual_resample_results(result_file, sel='Ynoise'):
    '''统计对单张图测50次的方差和均值
    Args:
        result_file: 结果文件路径
        sel： 选择需要统计的属性  Ynoise / strength
    '''
    re = {}
    gt = {}
    with open(result_file) as f:
        reader = csv.DictReader(f, delimiter='\t')
        
        for row in reader:
            if row['path'] not in re.keys():
                re.update({row['path']:[]})
                gt.update({row['path']:[]})

            re[row['path']].append(float(row[sel]))
            if 'gt_%s'%sel in row.keys():
                gt[row['path']].append(float(noise_class2idx[sel][row['gt_%s'%sel]]))
               
    (base_path, _)= os.path.split(result_file)
    fout = open(os.path.join(base_path, '%s_run50.txt'%sel),'w')
    for k,v in re.items():
        v = np.array(v)

        print_str = '%s,\tmean:%.3f,\tstd:%.3f,\t'%(k, np.mean(v), np.std(v))

        if len(gt[k]) > 0:
            y = gt[k][0]
            erro = np.abs(v - y)
            print_str += 'label:%.3f,\tmean_erro:%.3f,\tstd_erro:%.3f'%(y, np.mean(erro), np.std(erro))
        
        print(print_str)
        print_str+='\n'
        fout.write(print_str) 
            # cv2.imwrite(os.path.join(save_path,'crops_re_'+row['path']), image)
            # # cv2.imshow('result', image)
            # # key = chr(cv2.waitKey(0))
            # # if key == '\x1b':   ## esc 按钮退出
            # #     break
    fout.close()

import argparse
def get_args():
    parser = argparse.ArgumentParser(description='My detection code training based on pytorch!')
    parser.add_argument('--anno_dir', default='', help='save out put dir')
    parser.add_argument('--image_dir', default='', help='save out put dir')

    parser.add_argument('--mode', default=0, type=int, help='save out put dir')
    
    args = parser.parse_args()
    return args
if __name__ == "__main__":
    args = get_args()

    # visual_datasets('D:\\changruowang\\data\\noise_level\\train.txt')
    # visual_label_distribute('/home/arc-crw5713/data/noise_level/train.txt')
    # visual_board_log()  

    if args.mode == 0:
        # 测试图片文件路径  输出结果路径
        base_dir, _ = os.path.split(args.anno_dir)
        save_dir = os.path.join(base_dir, 'out')
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        visual_crop_results2(args.anno_dir, args.image_dir, save_dir)

    elif args.mode == 1:
        visual_resample_results(args.anno_dir, sel='strength')

    