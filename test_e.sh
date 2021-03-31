#! /bin/bash  
###  在测试集上对比 输入 patch 为 128  和  输入为 patch 为 64 的结果
model_name='efficientnet-b0'
output_sel='0_1_0'
use_loss_weights='f'
lr='0.001'

## test dir
image_dir='/home/arc-crw5713/data/noise_level/test_images'   
test_mode='dir'
crop_sel='fixed'
tta_metric='mean' 


patch_size='128' 
work_dir='g250_%s_'$lr'_wt_'$use_loss_weights'_o_'$output_sel
python test.py --work_dir $work_dir --model_name $model_name --crop_sel $crop_sel --image_dir $image_dir \
                --patch_size $patch_size  --test_mode $test_mode --use_tta --tta_metric $tta_metric


anno_dir='g250_efficientnet-b0_0.001_wt_f_o_0_1_0/imagedir_pred_results.txt'
python visual.py --anno_dir $anno_dir --image_dir $image_dir --mode 0


patch_size='64' 
work_dir='g_%s_'$lr'_wt_'$use_loss_weights'_o_'$output_sel
python test.py --work_dir $work_dir --model_name $model_name --crop_sel $crop_sel --image_dir $image_dir \
                --patch_size $patch_size  --test_mode $test_mode --use_tta --tta_metric $tta_metric

anno_dir='g_efficientnet-b0_0.001_wt_f_o_0_1_0/imagedir_pred_results.txt'
python visual.py --anno_dir $anno_dir --image_dir $image_dir --mode 0