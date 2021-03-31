#! /bin/bash 

# ############################### test ################################## 
#### 同时对比 efficient-b0 和 mobilenet-v3的效果
### crop_sel： patch采样方式 topK_var  random  top1_var  range  area  fix
### tta_metric:  tta结果融合方式   median  mean

model_name='efficientnet-b0,mobilenet_v3_small'
work_dir='d_%s_0.001_wt_t_o_1_1_0'
patch_size='64' 


## 在 测试图象上 对比 efficientnet  和  mobilenet_v3_small 的预测性能，这种测试方式是针对没有gt的图象
## 将 run_times 设置为 50 则对每张图分别运行50次
image_dir='/home/arc-crw5713/data/noise_level/test_images'   
test_mode='dir'
crop_sel='fix'
tta_metric='mean' 

## 读取 .h5 文件 的方式 测试平均损失，在验证集/训练集上测试平均损失
# # test using dataloader
# test_mode='loader'
# crop_sel='random'  ## 不能使用tta  
# tta_metric='mean'

## 以 txt 文件的形式测试，可以是训练集的子集或者验证集的子集，图片名和对应的注释以 txt 文件存储 
# ## test using txt
# image_dir='/home/arc-crw5713/data/noise_level/val_sub.txt'  
# test_mode='txt'
# crop_sel='area'   
# tta_metric='mean'


python test.py --work_dir $work_dir --model_name $model_name --crop_sel $crop_sel --image_dir $image_dir \
                --patch_size $patch_size  --test_mode $test_mode --use_tta --tta_metric $tta_metric \
                --run_times 1

# ############################### visual ################################## 

### 将输出结果可视化 anno_dir 为测试输出的 结果文件
## mode 0 表示将anno_dir中的结果 在 图上标注，标注crop的区域和对应的得分
## mode 1 表示统计50次运行的平均误差和方差 （需要将上面的 run_times 设置为 50）
anno_dir='d_mobilenet_v3_small_0.001_wt_t_o_1_1_0/imagedir_pred_results.txt'

python visual.py --anno_dir $anno_dir --image_dir $image_dir \
                --mode 0