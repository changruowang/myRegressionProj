import cv2
import numbers
import random
import numpy as np
import utils
import math
import torchvision.transforms as transforms
import os 
### 必须关闭opencv的多线程 否则会和dataloader的冲突
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

###整张图 等间隔采样16个patch
def sample_range_crops(h, w, th, tw):
    '''整张图 等间隔采样16个patch
    Args: 
        h,w: 原图的宽长  
        th, tw: 采样patch的宽长
    Returns:
        采样的patch的左上角点坐标  ((x1,y1), (x2,y2),...,())
    '''
    poses = []

    h_step = math.floor((h - th * 4) / 5)
    w_step = math.floor((w - tw * 4) / 5)

    for w_i in range(4):
        for h_i in range(4):
            x = w_i * tw + (w_i + 1) * w_step
            y = h_i * th + (h_i + 1) * h_step
            poses.append((x, y))

    return poses


def is_patch_valid(patch):
    '''判断patch是否有效
    Args: 
        patch： 图像数据numpy格式
    Returns:
        False无效、True有效
    '''
    # print(np.mean(patch), '.....')
    if np.mean(patch) > 220:
        return False
    return True

def sample_area_crops(h, w, p, img):
    '''整张图分为3x3 9 个 大区域  每个区域随机采样一个patch 
    Args: 
        h,w: 原图的宽长
        p: 采样的patch的尺寸
        img： 原图像数据numpy格式（会用来筛选patch是否有效）
    Returns:
        采样的patch的左上角点坐标  ((x1,y1), (x2,y2),...,())
    '''
    poses = []

    h_space, w_space = math.floor(h / 3), math.floor(w / 3)
    assert(h_space >= p and w_space >= p)

    for i in range(3):
        for j in range(3):
            h_st = i * h_space
            w_st = j * w_space

            cnt = 0
            while cnt < 50:
                cnt += 1
                h_s = random.randint(h_st, h_space*(i+1)-p-1)
                w_s = random.randint(w_st, w_space*(j+1)-p-1)
                if is_patch_valid(img[h_s:h_s+p, w_s:w_s+p, :]):
                    break

            poses.append((w_s, h_s))
    return poses

import json
### 输入 OpenCV numpy格式的数据  RGB
class MySelectedCrop(object):
    '''自己实现的 '有选择' 的crop 图像的方法。主要用途有两个：
        一方面作为 对接 torchvision 训练数据增广的接口，在训练过程中可以选择  
        topK_var, top1_var, random三种crop的方法。（使用hd5f预先crop打包训练数据后基本就用不上这个
        功能了）
        另一方面  作为测试的 tta 接口，测试时 调用 tta_out 完成
    Args: 
        size: 输出patch尺寸
        mode: crop的方法 （测试和训练的时候支持的crop方法不同）
    '''

    def __init__(self, size, mode='topK_var'):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.mode = mode
        self.fixed_crops = {}
        ### 读取存储 crop 信息的文件，如果没有 就默认用 area 的方法 crop 并将crop结果存下
        if mode == 'fixed':
            f = open('fixed_crops.json', 'r')
            self.fixed_crops = json.loads(f.read())
            # print(self.fixed_crops)
            if self.fixed_crops is None:
                self.fixed_crops = {}

    @staticmethod
    def cal_var_score(x, patch_size):
        ''' 计算方差得分。首先计算图象的局部方差，将方差低于阈值的位置设置为0，高于的设置为1，
            然后统计一个patch_size区域内的均值，极为最后得分。物理意义就是 统计patch_size中局部方差
            大于阈值的像素比例。   
        Args: 
            x: 原图象numpy格式
            patch_size: 要裁剪的patch尺寸
        Returns:
            返回计算的每个像素位置的得分  大小和输入图像大小一样
        '''
        img = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)

        img = img.astype(np.float32) / 255.0
        # 计算均值图像和均值图像的平方图像
        img_blur = cv2.blur(img, (7, 7))
        reslut_1 = img_blur ** 2
        # 计算图像的平方和平方后的均值
        reslut_2 = cv2.blur(img ** 2, (7, 7))

        local_var = np.sqrt(np.maximum(reslut_2 - reslut_1, 0)) * 100

        h, w = img.shape
        hf_ph, hf_pw = int(patch_size[0]/2), int(patch_size[1]/2)

        socores = np.zeros((h, w))

        local_var[local_var >= 1.0] = 1
        local_var[local_var < 1.0] = 0

        scores = cv2.blur(local_var, patch_size)
        results = np.zeros((h, w))
        
        results[hf_ph+1 : h-hf_ph, hf_pw+1 : w-hf_pw] = scores[hf_ph+1 : h-hf_ph, hf_pw+1 : w-hf_pw]
        return results

    def get_crop_mounts(self):
    '''该函数主要在 tta 测试的时候用，不同的 tta 方式生成的crop的数量不同 训练的时候只输出1个patch
    '''
        if self.mode == 'topK_var':
            return 16
        if self.mode == 'range':
            return 16
        if self.mode == 'area' or self.mode == 'fixed':
            return 9

    def get_params(self, img, output_size, K=None ,OUT_SCORES=False):
        '''训练的时候调用该函数，根据 crop 的模式 选择一个patch输出
        Args: 
            img: 输入图片
            output_size： 输出 crop 后的尺寸
            K： topK_var 模式中的 K 参数，即从前K个大的方差中选择要crop的位置
            OUT_SCORES： 是否输出方差得分，如果为False则从topK中的随机选择一个crop位置输出，如果为
                        true, 则将topK的位置和对应的得分一起返回（在测试tta中使用）
        Returns:
            返回crop的位置 y x h w
        '''
        if K is None:
            K = 0.5
        h, w, _ = img.shape
        th, tw = output_size

        if w == tw and h == th:
            return 0, 0, h, w

        if self.mode == 'random':
            i = random.randint(0, h - th)
            j = random.randint(0, w - tw)
            return i, j, th, tw

        scores = self.cal_var_score(img, output_size)

        if self.mode == 'topK_var' or OUT_SCORES:
            K = int(w * h * K)  ##前百分之70大的点
            x, y = np.arange(w), np.arange(h)
            x, y = np.meshgrid(x, y)
            pos = np.vstack((x.flatten(), y.flatten()))
            scores_flatten = scores.flatten()
            idx = np.argpartition(scores_flatten, -K)[-K:]
            if OUT_SCORES:
                return pos[:, idx], scores_flatten[idx]
            sel_poses = pos[:, idx]
            pos = sel_poses[:, random.randint(0, sel_poses.shape[1])]
        elif self.mode == 'top1_var': 
            sel_poses = np.where(scores==np.max(scores.flatten()))
            # print(len(sel_poses[1]), np.max(scores.flatten()))
            sel_poses = np.vstack((sel_poses[1], sel_poses[0]))
            pos = sel_poses[:, 0]
        else:
            raise NotImplementedError("Not implement crop mode in __call__%s"%self.mode)  
        
        # print(pos[1]-th//2, pos[0]-tw//2, th, tw, h, w)

        return pos[1]-th//2, pos[0]-tw//2, th, tw

    def tta_out(self, img, path=None):
        '''tta测试的时候调用该函数，根据 tta 的 crop 的模式 crop出多个 patch
            支持的tta模式有： fixed range area topK_var
            range：从图上等间隔crop 
            fixed: 从json配置文件中读出每张图固定的crop位置（键为图片路径，值为crop的位置）如果配置文件不存在
                    就默认为 area 方式crop,并在测试结束后将crop的信息存储为json，这个模式是为了实现，不同
                    crop_size的对比实验。测试两次，分别设置不同的crop_size，使用相同的文件
        Args: 
            img: 输入图片
            path: 图片的路径，在fixed 模式下使用
           
        Returns:
            返回crop的位置 y x h w
        '''
        re = []
        crop_poses = []
        th, tw = self.size
        h, w, _ = img.shape
        if self.mode == 'topK_var':
            poses, scores = self.get_params(img, self.size, OUT_SCORES=True, K=0.2)
            sort_idxs = np.argsort(scores, kind='mergesort')  
            sel_poses = poses[:, sort_idxs]
            m = sel_poses.shape[1]
          
            scores = scores[sort_idxs]  ##  -int(m/8), -int(m/4), -int(m/2)    -5, -10, -15
            for k in range(1,int(m),int(m/15)):# (-1, -int(m/8), -int(m/4), -int(m/2), -int(m)):   # , -int(len(sel_poses)/8), -int(len(sel_poses)/4), -int(len(sel_poses)/2)
                
                i, j = sel_poses[:, -k][1]-th//2,  sel_poses[:, -k][0]-tw//2
                # print('pos x y:', j, i, scores[k])
                crop_poses.append((j, i))
               
        elif self.mode == 'range':
            crop_poses = sample_range_crops(h, w, th, tw, img)
        elif self.mode == 'area':  ## 分区域采样
            crop_poses = sample_area_crops(h, w, th, img)
        elif self.mode == 'fixed':
            if path in self.fixed_crops.keys():
                crop_poses =  self.fixed_crops[path]
            else:
                crop_poses = sample_area_crops(h, w, th, img)
                self.fixed_crops.update({path:crop_poses})

        re = [img[y:y+th, x:x+tw, :] for (x, y) in crop_poses]

        return re, crop_poses

    def __del__(self):
        ''' fixed 模式，退出保存之前采样的crop位置信息为json文件
        '''
        if self.mode == 'fixed':        
            b = json.dumps(self.fixed_crops)
            f2 = open('fixed_crops.json', 'w')
            f2.write(b)
            f2.close() 

    def __call__(self, img):
        i, j, h, w = self.get_params(img, self.size)

        return img[i:i+h, j:j+w, :]

# Flipped Horizontally 水平翻转
class MyRandomHorizontalFlip(object):
    ''' 使用opencv 重新实现的 数据扩增方法 水平翻转 p反转的概率
    '''
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, img):
        if random.random() < self.p:
            return cv2.flip(img, 1)
        return img
###　垂直翻转
class MyRandomVerticalFlip(object):
    ''' 使用opencv 重新实现的 数据扩增方法 随机垂直翻转 p反转的概率
    '''
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, img):
        if random.random() < self.p:
            return cv2.flip(img, 0)
        return img


def get_transform(train=False, path_size=64, params='random'):
    ''' 输入网络前的图象预处理流程 包含了 随机垂直水平反转，随机裁剪，转化为tensor 以及归一化
    Args:
        path_size: 裁剪的patch尺寸
        params： 未使用
    '''
    if train:
        return transforms.Compose([
                                    MyRandomHorizontalFlip(),
                                    MyRandomVerticalFlip(),
                                    MySelectedCrop(path_size, mode=params),  
                                    # transforms.RandomCrop(path_size),                
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    else:
        return transforms.Compose([ MySelectedCrop(path_size, mode=params),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
