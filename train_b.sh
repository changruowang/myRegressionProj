#! /bin/bash  
### 预先从一张图中扣40张64x64的图，筛选规则为 patch中非平坦区域占比大于百分之30的patch中随机选，
### 若不够则从随机选  因为已经预先扣好了 所以patch_size设置为64即可

lr='0.001'
epochs='500'
milestones='50,200,300'
model_name='efficientnet-b0'
use_loss_weights='f'
crop_sel='random'    
patch_size='64' 
output_sel='1_0_0'

## var crop的数据集
data_dir='/home/arc-crw5713/data/crop_yoise_var64'
work_dir='v_'$model_name'_'$lr'_wt_'$use_loss_weights'_o_'$output_sel


python train.py --work_dir $work_dir --model_name $model_name --lr $lr --crop_sel $crop_sel --data_dir $data_dir \
                --milestones $milestones --use_loss_weights $use_loss_weights --patch_size $patch_size --output_sel $output_sel
