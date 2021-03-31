#! /bin/bash  
### 预先从一张图中扣40张64x64的图，筛选规则为 patch中 灰度阈值抠图，若不够则从剩下的随机选
### 训练 strength 属性

lr='0.001'                      ## 初始学习率
epochs='500'                    ## 训练的总epoch数
milestones='50_200_300'         ## 学习率调整区间，每次学习率/10
model_name='efficientnet-b0'    ## backbone名字  修改为 unet / efficientnet-b0 复现实验a的训练过程
use_loss_weights='f'            ## 不使用样本权重，只有在两个数据集联合输出训练的时候设置为t
crop_sel='random'               ## 无用
patch_size='64'                 ## 输入网络的 patch 尺寸
output_sel='0_1_0'              ## 选择需要训练的属性  0_1_0 即训练 strength属性  1_0_0 代表训练ynoise属性 
                                ## 训练不同的属性要注意选择 对应的 balance crop数据集

## 使用gray阈值 预先crop的数据集路径
data_dir='/home/arc-crw5713/data/crop_strength_gray64'
work_dir='g_'$model_name'_'$lr'_wt_'$use_loss_weights'_o_'$output_sel

python train.py --work_dir $work_dir --model_name $model_name --lr $lr --crop_sel $crop_sel --data_dir $data_dir \
                --milestones $milestones --use_loss_weights $use_loss_weights --patch_size $patch_size

