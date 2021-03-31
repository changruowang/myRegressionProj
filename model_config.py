#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torchvision
import torch.nn as nn
import os
import torch
# import torchvision.models as models
from models.timm.senet import seresnet50, seresnet18
import torch.nn.functional as F
from models.mobilenet_v3 import mobilenetv3
from models.ghostnet import ghostnet
from models.unet import U_Net,U_Net_Att

class MyEfficientNet(nn.Module):
    '''用于 封装 EfficientNet 无实际意义 
    '''
    def __init__(self, name):
        super().__init__()
        from efficientnet_pytorch import EfficientNet
        self.model = EfficientNet.from_pretrained(name)
        self.last_channel = 1280
        # self.features = self
    def __call__(self, x):
        return self.model.extract_features(x)



class MultiOutputModel(nn.Module):
    ''' 用于封装不同的backbone 的结构  backbone + avg_pool + 2xline
    '''
    def __init__(self, n=2, model_name='se_resnet18', weights=True, loss_sel=[1,1]):
        ''' 
        Args： 
            n: 输出需要回归的属性数量， 默认2  即回归 Ynoise strength 两个属性
            model_name: backbone的名字.目前支持的有：
                mobilenet_v2  se_resnet18 resnet50 efficientnet-b0 mobilenet_v3_small unet ghostnet
            weights: 是否使用 样本权重 在训练的时候给不同样本设置权重。只有在联合训练的模式上用得上。
            loss_sel： 需要训练的输出  [1,1] 代表两个输出都训练  [1,0]代表只训练第一个输出
        '''
        super().__init__()
        self.loss_sel = loss_sel
        self.base_model, last_channel = self.get_feats_extract_convs(model_name)
        
        self.weights = weights
    
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # create separate classifiers for our outputs
        self.regression = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=last_channel, out_features=n)
        )
 
        if model_name == 'unet_att':
            kernel = [[1/9.0, 1/9.0, 1/9.0],
                [1/9.0, 1/9.0, 1/9.0],
                [1/9.0, 1/9.0, 1/9.0]]
            kernel = torch.FloatTensor(kernel).expand(1,1,3,3)
            self.weight = nn.Parameter(data=kernel, requires_grad=False)
    ### 不使用
    def cal_local_var(self, img):
        v = torch.max(img, dim=1)[0].unsqueeze(1)
        # 计算均值图像和均值图像的平方图像
        # img_blur = cv2.blur(img, (size, size))
        v_pad = nn.ReflectionPad2d(1)(v)
    
        img_blur = F.conv2d(v_pad, self.weight, padding=0, groups=1)

        reslut_1 = img_blur ** 2
        # 计算图像的平方和平方后的均值
        v2 = nn.ReflectionPad2d(1)(v ** 2)
        reslut_2 = F.conv2d(v2, self.weight, padding=0, groups=1)

        var_ = torch.sqrt(torch.max(reslut_2 - reslut_1, torch.zeros_like(reslut_1)))

        return (var_ - torch.min(var_))/ (torch.max(var_) - torch.min(var_))


    def get_feats_extract_convs(self, model_name):
        ''' 获取 backbone的特征提取层
        Args： 
            model_name： backbone 的名字
        '''
        self.model_name = model_name
        if model_name == 'mobilenet_v2':
            model = torchvision.models.mobilenet_v2(pretrained=True)
        elif model_name == 'se_resnet18':
            model = seresnet18() 
        elif model_name == 'resnet18':
            model = torchvision.models.resnet18() 
        elif model_name == 'resnet50':
            model = torchvision.models.resnet50() 
        elif model_name == 'efficientnet-b0' or model_name == 'efficientnet-b0-sig':
            model = MyEfficientNet('efficientnet-b0')
        elif model_name == 'mobilenet_v3_small':
            model = mobilenetv3(mode='small', pretrained=True)
        elif model_name == 'ghostnet':
            model = ghostnet(width=1.0)
        elif model_name == 'unet':
            model = U_Net()
        elif model_name == 'unet_att':
            model = U_Net_Att()
        try: 
            try:
                base_model = model.features 
            except:
                base_model = model
            last_channel  = model.last_channel 
        except:
            base_model = nn.Sequential(*list(model.children())[:-2])

            with torch.no_grad():
                out = base_model(torch.rand(1,3,224,224))
            last_channel = out.shape[1]
        return base_model, last_channel

    ### 不使用
    def weightsdict2list(self, wts_dict):
        wts_list = {k: None for k,v in wts_dict.items()}
        for k,v in wts_dict.items():
            temp = {}
            for k1, v1 in v.items():
                temp.update({k1: float(v1)}) 
            wts_list[k] = temp
        return wts_list

    def forward(self, x):
        '''
        Returns: 
            返回字典的形式  {'Ynoise'： tensor, 'strength': tensor}
        '''
        if self.model_name == 'unet_att':  ## 这个模型弃用
            var = self.cal_local_var(x)
            x = torch.cat((x, var), dim=1)

        x = self.base_model(x)  ## 特征提取
        
        x = self.pool(x)        ## 池化

        x = torch.flatten(x, 1) ## 全连接输出
        re = self.regression(x)
       
        return {
            'Ynoise': re[:,0],
            'strength': re[:,1]
        }
    ### 未用
    def gt2weights(self, attr, gt):
        weight_list = self.weights[attr].to(gt.device)
        return weight_list
    
    def get_loss(self, net_output, ground_truth):
        '''计算损失 L2损失
        Returns: 
            返回字典的形式
        '''
        if self.weights:  ###　给不同的样本损失乘权重 主要是在联合训练的时候 用于区分来自哪个数据集的样本
            wY = ground_truth['wYnoise']
            wS = ground_truth['wstrength']
            mY = wY[wY>0.0].size(0)
            mS = wS[wS>0.0].size(0)
       
            ynoise_loss = torch.sum(wY * ((net_output['Ynoise'] - ground_truth['Ynoise'])**2)) / mY
            strength_loss = torch.sum(wS * ((net_output['strength'] - ground_truth['strength'])**2)) / mS
            # cnoise_loss = F.cross_entropy(net_output['Cnoise'], gt_C, weight=self.gt2weights('Cnoise',gt_C))
        else:  ### 单输出训练直接计算损失
            
            ynoise_loss = torch.mean((net_output['Ynoise'] - ground_truth['Ynoise'])**2)
            strength_loss = torch.mean((net_output['strength'] - ground_truth['strength'])**2)
            
        loss = ynoise_loss*self.loss_sel[0] + strength_loss*self.loss_sel[1]
        
        return loss, {'Ynoise': ynoise_loss.item(), 
                      'strength': strength_loss.item(),
                      'loss_all': loss.item() }

if __name__ == "__main__":
### 测时间
    model = MultiOutputModel(2,'efficientnet-b0', weights=True)  # se_resnet18
    model.eval()
    x = torch.rand(1,3,64,64)
    import time
    start_time = time.clock() 
    with torch.no_grad():
        y = model(x)
    end_time = time.clock()
    print(end_time - start_time)

### 测模型复杂度
    # for model_name in [
    #                     'efficientnet-b0', 'unet', 'unet_att']:
    # # for model_name in ['mobilenet_v2', 'se_resnet18', 'resnet18', 'efficientnet-b0', 'efficientnet-b1',\
    # #                     'mobilenet_v3_small', 'ghostnet']:
    #     from thop import profile
    #     model = MultiOutputModel(2, model_name)
    #     in_ = torch.rand(1,3,224,224)
    #     flops, params = profile(model, inputs=(in_, ))
    #     print(model_name, ' flops:',flops, ' params:',params)


### se_resnet18         flops: 1819404152.0  params: 11271748.0
### resnet18            flops: 1818560512.0  params: 11182668.0
### efficientnet-b0     flops: 13554752.0    params: 57388.0
### efficientnet-b1     flops: 18760512.0    params: 77420.0
### mobilenet_v2        flops: 312929856.0   params: 2239244.0
### ghostnet            flops: 146280204.0   params: 2673350.0     1.658951
### ghostnetx075        flops: 89816236.0    params: 1528468.0
### mobilenet_v3_small  flops: 62389336.0    params: 1659930.0     1.54
### mobilenet_v3_large  flops: 215871480.0   params: 2818364.0
### unet                flops: 1937596800.0  params: 336802.0      0.52
### efficientnet-b0     flops: 13554752.0    params: 57388.0
