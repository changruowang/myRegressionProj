#! /bin/bash  
### 用于两个数据集交替迭代训练的数据
### 
lr='0.001'
epochs='500'
milestones='50_200_300'
model_name='mobilenet_v3_small'  ## 可将模型名字更换为 efficientnet-b0 
use_loss_weights='t'

 
output_sel='1_1_0'
crop_sel='random'   
patch_size='64'
ynoise_dir='/home/arc-crw5713/data/crop_ynoise_gray64'
strength_dir='/home/arc-crw5713/data/crop_strength_gray64'

work_dir='d_'$model_name'_'$lr'_wt_'$use_loss_weights'_o_'$output_sel


python union_train.py --work_dir $work_dir --model_name $model_name --lr $lr --crop_sel $crop_sel --ynoise_dir $ynoise_dir --strength_dir $strength_dir\
                --milestones $milestones --use_loss_weights $use_loss_weights --patch_size $patch_size --output_sel $output_sel
