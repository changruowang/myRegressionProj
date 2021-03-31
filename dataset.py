#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import os
import numpy as np
import h5py  
from PIL import Image
from torch.utils.data import Dataset
import glob
from tqdm import tqdm
import torch
import cv2
import time
from utils import select_throughGray, select_throughVar

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

##　离散标签 到 数值的映射  label idx-> num
noise_class2idx = {'Ynoise': {'unknown':0, 'small':0.25, 'middle':0.5, 'large':0.75, 'huge':1},
            'strength': {'low':0.25, 'middle':0.5, 'high':0.75, 'huge':1},
            'Cnoise': {'unknown':0, 'NoColorNoiseOrNormal':1, 'Abnoraml':2}}
##　每类属性汇中标签数目比例  label idx->weight
noise_class2weight = {'Ynoise': {'unknown':0.36, 'small':0.24, 'middle':0.12, 'large':0.1, 'huge':0.18},
            'strength': {'low':0.25, 'middle':0.12, 'high':0.125, 'huge':0.5},
            'Cnoise': {'unknown':0.47, 'NoColorNoiseOrNormal':0.075, 'Abnoraml':0.46}}


def read_noise_level_annotations(annotions, name_only=True):
    '''解析噪声标注文件
    Args:
        输入所有待解析的txt文件列表 [txt_path1, txt_path2, ...]
    Returns:
        图片文件名列表，与文件名对应位置的Ynoise标签， strength列表, Cnoise列表
    '''
    images_name = []
    Ynoise = []
    strength = []
    Cnoise = []

    if type(annotions) == type(''):
        annotions = [annotions]

    for annotion in annotions:
        with open(annotion) as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:                    ###跳过　unknown 类 0  0.25  0.5  0.75  1.0
                if row['Ynoise'] == 'unknown':
                    continue
                if name_only:
                    (_,tempfilename) = os.path.split(row[ 'path'])   
                else:
                    tempfilename = row['path']
                
                images_name.append(tempfilename)
                Ynoise.append(row['Ynoise'])
                strength.append(row['strength'])
                Cnoise.append(row['Cnoise'])   

                #print(tempfilename, ' ', row['Ynoise'], ' ', row['strength'], ' ', row['Cnoise'])

    assert(len(images_name) == len(Ynoise) and len(images_name) == len(strength) and len(images_name) == len(Cnoise))
    
    return images_name, Ynoise, strength, Cnoise
 
def save_as_hd5f(annotion_path, file_mounts=4):
    '''将噪声数据集打包为.h5数据集
    Args:
        annotion_path: 输入标注的噪声水平annotion.txt
        file_mounts: 输出.h5文件数目（在本地台式机上由于内存太小，无法一次性存储整个数据集，就将数据集分为多个.h5文件）
    '''
    (base_path,filename) = os.path.split(annotion_path)
    images_name, Ynoise, strength, Cnoise = read_noise_level_annotations(annotion_path)  ## len(images_name) 
    len_per_file = math.floor(len(images_name)  / file_mounts)
    image_datas = np.zeros((len_per_file, 250, 250, 3), dtype=np.uint8)  ### 不加dtype默认float64 大大增加内存
    
    par = tqdm(range(len_per_file * file_mounts), total=(len_per_file * file_mounts))
    
    lable_datas = [None] * len_per_file 
    img_names = [None] * len_per_file

    dt = h5py.string_dtype(encoding='utf-8')
    label_inform = np.array(['Ynoise'.encode("utf-8"), 'strength'.encode("utf-8"), 'Cnoise'.encode("utf-8")])

    for idx in par:
        i = idx % len_per_file
        name = images_name[idx]

        img_path = os.path.join(base_path, 'images', name)

        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
       
        lable_datas[i] = ([Ynoise[idx].encode("utf-8"), strength[idx].encode("utf-8"), Cnoise[idx].encode("utf-8")])
        img_names[i] = (name.encode("utf-8"))
        image_datas[i,:,:,:] = img

        if i == (len_per_file-1):   ### 
            f = h5py.File(os.path.join(base_path, filename[:-4]+'_%d.h5'%(int(idx/len_per_file))),'w')   # construct h5 file
            f['data'] = image_datas       

            lables_array = np.array(lable_datas)
            names_array = np.array(img_names)
 
            f.create_dataset('labels', lables_array.shape , dtype=dt, data=lables_array)        
            f.create_dataset('inform', label_inform.shape , dtype=dt, data=label_inform)        
            f.create_dataset('img_names', names_array.shape , dtype=dt, data=names_array)       
            f.close()  

    
def save_csv(data, path, fieldnames=['path', 'Ynoise', 'strength', 'Cnoise']):
    '''将列表数据存储为 csv 文件
    Args:
        data： [[a1, c1, d1, ...], [a2, c2, d2, ...]] 待存储数据
        path： 文件保存路径
        fieldnames： 数据列名称
    '''
    with open(path, 'w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames,delimiter='\t')
        writer.writeheader()
        for row in data:
            writer.writerow(dict(zip(fieldnames, row)))

def split_dataset(base_path):
    '''分割训练集和测试集，0.7比例
    Args:
        输入存放有txt格式的gt路径（可以读取多个txt）
    Returns:
        将分割后的数据集，存储为 train.txt  test.txt文件
    '''
    file_lists = glob.glob(os.path.join(base_path, '*.txt'))
    path, Ynoise,strength,Cnoise = read_noise_level_annotations(file_lists)

    all_data = []
  
    pbar = tqdm(range(len(path)), total=len(path))
    for idx in pbar:
        img_path = os.path.join(base_path, 'images', path[idx])
        if os.path.exists(img_path):
            all_data.append([img_path, Ynoise[idx],  strength[idx],  Cnoise[idx]])

    mounts = int(len(all_data)*0.7)
    np.random.seed(42)
    all_data = np.asarray(all_data)
    inds = np.random.choice(len(all_data), len(all_data), replace=False)
    save_csv(all_data[inds][:mounts], os.path.join(base_path, 'train.txt'))
    save_csv(all_data[inds][mounts:], os.path.join(base_path, 'val.txt'))

def crop_patches(img, patch_size=64, mode='gray'):
    '''从大图中crop小图
    Args:
        img: 大图
        patch_size： 输出patch尺寸
        mode: crop 的依据
            gray: 随机选 并根据灰度阈值筛除不合适的patch
            var: 优先选择局部方差较大的patch
    Returns:    
        返回经过筛选后的 patch 的左上角顶点列表  
    '''
    if mode == 'var':
        selected_pos = select_throughVar(img, patch_size)
    elif mode == 'gray':
        selected_pos = select_throughGray(img, patch_size)

    ### 剔除间距小于 45 的点，保证采样的patch位置 不一样
    dist = scipy.spatial.distance.cdist(selected_pos.T, selected_pos.T, metric='euclidean')
    y, x = np.where(dist < 30)    ## 45 时IOU大约0.25
    del_idxs = set(x[x > y])   ## 选取矩阵上三角坐标   del_idxs 上三角区域含有距离小于20的列号
    sel_idxs = set(range(dist.shape[0])) - del_idxs
    final_poses = selected_pos[:, list(sel_idxs)]

    return final_poses

def generate_balanced_crop_images(annotion_path, save_path, BALANCE_CAT='strength', patch_size=64, mode='gray'):
    '''制作具有 类别平衡的数据集
    Args:
        annotion_path: 原始数据集的标注文件路径
        save_path： 输出存储位置
        BALANCE_CAT: 需要保证平衡的属性 只支持 Ynoise  or  strength  （如果是新的数据集 需要修改orig_sample_weights字典）
        patch_size： crop 的样本尺寸
        mode：crop 的依据
            gray: 随机选 并根据灰度阈值筛除不合适的patch
            var: 优先选择局部方差较大的patch
    '''
    orig_sample_weights = {'Ynoise': {'small':60, 'middle':30, 'large':25, 'huge':40}, 
                            'strength': {'low':100, 'middle':48, 'high':48, 'huge':200}}
    patchs_per_image = orig_sample_weights[BALANCE_CAT]

    (base_path, _) = os.path.split(annotion_path)   
    from dataset import read_noise_level_annotations, save_csv
    images_name, Ynoise, strength, Cnoise = read_noise_level_annotations(annotion_path)

    image_save_dir = os.path.join(save_path, 'images')
    if not os.path.exists(image_save_dir):
        os.makedirs(image_save_dir)

    all_data = []

    pbar = tqdm(range(len(images_name)), total=len(images_name))
    for idx in pbar:
        image = cv2.imread(os.path.join(base_path,'images',images_name[idx]))
        h, w, _ = image.shape 

        selected_pos = crop_patches(image, patch_size=patch_size, mode=mode)  ##  selected_pos  (2, N)

        if BALANCE_CAT == 'Ynoise':
            M = patchs_per_image[Ynoise[idx]] 
        else:      
            M = patchs_per_image[strength[idx]] 
        ## 一张图
        for idx_i in range(min(M, selected_pos.shape[1])):
            x, y = selected_pos[0, idx_i], selected_pos[1, idx_i]
            x, y = int(x - patch_size / 2), int(y - patch_size / 2)
            if x < 0 or y < 0 or x + patch_size >= w or y + patch_size >= h:
                continue
            new_name = 'crop_%d_%d_'%(idx, idx_i)+images_name[idx]
            save_dir = os.path.join(image_save_dir, new_name)
            cv2.imwrite(save_dir, image[y:y+patch_size, x:x+patch_size, :])
            all_data.append([new_name, Ynoise[idx], strength[idx], Cnoise[idx]])
               
    all_data = np.asarray(all_data)
    save_csv(all_data, os.path.join(save_path, 'annotions.txt'))

class HDF5_DataSet(Dataset):
    '''加载HD5F格式的数据集  ｛'data': _, 'labels': _ ...｝
    
    :db_path: 含有hd5f格式数据的路径，训练书记用 train*.h5方式命名， 测试数据用 test*.h5 方式命名
    :phase: phase = 'train' / 'test' 表示是训练数据还是测试数据
    :transform： torchvision格式的数据扩增结构。由于torchvision使用PIL实现的速度较慢，因此重新实现了 几个常用的方法
               在my_transform.py文件夹
    '''
    def __init__(self, db_path, phase='train', transform=None):
        super(HDF5_DataSet, self).__init__()  
     
        file_lists = glob.glob(os.path.join(db_path, '%s*.h5'%(phase)))

        self.labels = []
        self.names = []
        print('loading data from h5 files....')
        ### 合并多个 train01.h5  train02.h5 ... 文件
        for idx, f5_file in enumerate(file_lists):
            f = h5py.File(f5_file ,'r', libver='latest', swmr=True)

            if idx == 0:
                h, w, c = f['data'][0,:,:,:].shape
                n = len(f['data'])
                self.data = np.zeros((n*len(file_lists), h, w, c), dtype=np.uint8)
     
            self.data[idx*n:(idx+1)*n, :, :, :] = f['data'][:].astype(np.uint8)
            self.labels.append(f['labels'][:])
            self.names.append(f['img_names'][:])
            self.inform = f['inform'][:]
            f.close()
           
        self.labels = np.concatenate(self.labels, axis = 0)
        self.names = np.concatenate(self.names, axis = 0)
        self.transform = transform
        print('data ok')

    def __getitem__(self, index):
        '''读取一条数据
        Returns:
            返回一个字典，'name'表示图片名字， 'img'为读取的图片数据， 'labels':标签，已经映射成了数字    
        '''
        img = self.data[index,:,:,:].astype(np.uint8)

        labels = self.labels[index,:]
        name = self.names[index]
        if self.transform:
            img = self.transform(img)
        
        labels = [float(noise_class2idx[self.inform[idx]][lb]) for idx,lb in enumerate(labels)]
    

        dict_data = {
            # 'var': local_var
            'name': name,
            'img': img,
            'labels': {
                'Ynoise': labels[0],
                'strength': labels[1],
                'Cnoise': labels[2]
            }
        }

        return dict_data
    
    def __len__(self):
        return self.data.shape[0]   

class DoubleHDF5_DataSet(Dataset):
    '''加载HD5F格式的数据集  与 HDF5_DataSet 不同的是，这个可以同时加载 strenght 和 ynoise两个数据集，读取的时候，
        根据索引index的奇偶选择从哪个数据集读取数据。
    Args：    
        :ynoise_path: ynoise数据集存放路径
        :strength_path: strength数据集存放路径
        :transform： torchvision格式的数据扩增结构。由于torchvision使用PIL实现的速度较慢，因此重新实现了 几个常用的方法
            在my_transform.py文件夹
        :phase: phase = 'train'/'test' 表示是训练数据还是测试数据           
    '''
    def __init__(self, ynoise_path, strength_path, phase='train', transform=None):
        super(DoubleHDF5_DataSet, self).__init__()  
        
        ynoise_files = glob.glob(os.path.join(ynoise_path, '%s*.h5'%(phase)))
        strength_files = glob.glob(os.path.join(strength_path, '%s*.h5'%(phase)))
        print('loading ynoise data...')
        self.ydata, self.ylabels, self.ynames, self.yinform = self.load_hd5_files(ynoise_files)
        print('loading strength data...')
        self.sdata, self.slabels, self.snames, self.sinform = self.load_hd5_files(strength_files)
        
        assert(list(self.yinform) == list(self.sinform))
        
        m = min(len(self.ydata), len(self.sdata))
        self.data_len = m * 2
        self.transform = transform
        print('data ok')

    def load_hd5_files(self, file_lists):
        '''加载合并多个 hd5f 文件
        '''
        labels = []
        names = []
        for idx, f5_file in enumerate(file_lists):
            f = h5py.File(f5_file ,'r', libver='latest', swmr=True)

            if idx == 0:
                h, w, c = f['data'][0,:,:,:].shape
                n = len(f['data'])
                data = np.zeros((n*len(file_lists), h, w, c), dtype=np.uint8)
     
            data[idx*n:(idx+1)*n, :, :, :] = f['data'][:].astype(np.uint8)
            labels.append(f['labels'][:])
            names.append(f['img_names'][:])
            inform = f['inform'][:]
            f.close()
        labels, names = np.concatenate(labels, axis = 0), np.concatenate(names, axis = 0)
        return data, labels, names, inform

    def __getitem__(self, index):
        '''读取一条数据
        Returns:
            返回一个字典，'name'表示图片名字， 'img'为读取的图片数据， 'labels':标签，已经映射成了数字
            根据索引index的奇偶选择从哪个数据集读取数据，将来自对应数据集的标签权重（wYnoise、wstrength）记为1 否则0    
        '''
        weight = index % 2   ## 根据单双树切换数据集  偶数选择 Ynoise
        idx = int(index/2 if weight==0 else (index-1)/2)

        img = (self.ydata[idx, :,:,:] if weight==0 else self.sdata[idx, :,:,:]).astype(np.uint8)
        labels = self.ylabels[idx,:] if weight==0 else self.slabels[idx,:]
        name = self.ynames[idx] if weight==0 else self.snames[idx]

        if self.transform:
            img = self.transform(img)
        
        labels = [float(noise_class2idx[self.yinform[idx]][lb]) for idx,lb in enumerate(labels)]
    
        dict_data = {
            'name': name,
            'img': img,
            'labels': {
                'Ynoise': labels[0],
                'strength': labels[1],
                'Cnoise': labels[2],
                'wYnoise': 1-float(weight),
                'wstrength': float(weight)
            }
        }

        return dict_data
    
    def __len__(self):
        return self.data_len 



### 未使用
class NoiseLevelRegDataset(Dataset):
    def __init__(self, base_path, phase=None, class2idx=None,transform=None, class2weight=None):
        super().__init__()

        self.transform = transform
        
        self.base_path = base_path
        if phase is None:
            file_lists = glob.glob(os.path.join(base_path, '*.txt'))
        else:
            file_lists = os.path.join(base_path, phase+'.txt')
        # 转换成 idx 
        self.data, self.Ynoise, self.strength, self.Cnoise = read_noise_level_annotations(file_lists)

        self.class2weight = class2weight
        if class2weight is not None:
            self.wYnoise = [0] * len(self.data)
            self.wstrength = [0] * len(self.data)
            for idx in range(len(self.data)):
                self.wYnoise[idx] = float(class2weight['Ynoise'][self.Ynoise[idx]])
                self.wstrength[idx] = float(class2weight['strength'][self.strength[idx]])

        for idx in range(len(self.data)):
            self.Ynoise[idx] = float(class2idx['Ynoise'][self.Ynoise[idx]])
            self.strength[idx] = float(class2idx['strength'][self.strength[idx]])
            self.Cnoise[idx] = class2idx['Cnoise'][self.Cnoise[idx]]

            # print(self.Ynoise[idx], ' ', self.strength[idx], ' ', self.Cnoise[idx])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.base_path, 'images', self.data[idx])

        # start_time = time.clock()     #
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        # img = Image.open(img_path)
        # end_time = time.clock()
        # print(end_time-start_time)    #

        if self.transform:
            img = self.transform(img)


        dict_data = {
            'name': self.data[idx],
            'img': img,
            'labels': {
                'Ynoise': self.Ynoise[idx],
                'strength': self.strength[idx],
                'Cnoise': self.Cnoise[idx]
            }
        }

        if self.class2weight is not None:
            dict_data['labels'].update({'wYnoise':self.wYnoise[idx], 'wstrength':self.wstrength[idx]})

        return dict_data 

import argparse
def get_args():
    parser = argparse.ArgumentParser(description='Make noise level eval dataset')
    parser.add_argument('--anno_dir', help='')
    parser.add_argument('--save_dir', help='')
    parser.add_argument('--make_category', default='strength', type=str, help='strength or Ynoise')
    parser.add_argument('--file_mounts', default=4, type=int, help='>0')
    parser.add_argument('--mode', default='gray', type=str, help='gray or var')
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    ### 从大图上crop出patch 并存储
    generate_balanced_crop_images(args.anno_dir, args.save_dir, args.make_category,
                                    args.crop_size, args.mode)
    ### 将crop出来的patches 打包为 .h5文件
    save_as_hd5f(os.path.join(args.save_dir, 'annotions.txt'), file_mounts=args.file_mounts)
