#! /bin/bash  
###  用于 从初始大小为250的大patch 中随机扣 128 输入网络训练的脚本
lr='0.001'
epochs='500'
milestones='50_200_300'
model_name='efficientnet-b0'
use_loss_weights='f'      
crop_sel='random'                   ### 大 patch 中随机 crop 一个128的小patch　 
output_sel='0_1_0'                  ### 训练  strength

## gray crop的数据集
patch_size='128'
data_dir='/home/arc-crw5713/data/crop_strength_gray250'
work_dir='g250_'$model_name'_'$lr'_wt_'$use_loss_weights'_o_'$output_sel


python train.py --work_dir $work_dir --model_name $model_name --lr $lr --crop_sel $crop_sel --data_dir $data_dir \
                --milestones $milestones --use_loss_weights $use_loss_weights --patch_size $patch_size --output_sel $output_sel
