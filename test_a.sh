#! /bin/bash 

############################### 参数意义 ################################
# crop_sel： 测试patch采样方式 topK_var  random  range  area  fixed
#       topK_var： 从方差前K个大的位置中随机选择 crop的区域
#       random：全图随机 crop 
#       range:  全图 等间距采样N个patch测试
#       area：  将全图划分为 9 个子区域 每个子区域随机采集一个patch测试
#       fixed:  从规定的配置文件中获取某张图的 crop 位置，若配置文件不存在 默认为area方式crop
# tta_metric:  tta结果融合方式   median  mean
#       median： 取多个 patch 测试结果的中位数为最终预测结果
#       mean: 取多个patch 测试结果的均值
# test_mode： 加载测试数据的方法，支持 3 种
#       dir： 直接从包含图片的文件夹中读取图片列表，因此不能读取标注信息，这种方式主要用来测试
#             少量图片并可视化结果
#       loader： 加载制作好的 .h5 数据集测试，包含标签，这种方式主要用来统计验证集上的平均损失  
#       txt： 加载以 txt 文件标注的数据集，可读标签，用来测试少量图片并可视化测试结果和gt结果
# image_dir：包含待测试图片的文件夹 /  txt 注释的文件路径 / 包含.h5 数据的文件路径
# patch_size：测试时 输入网络测试的 patch 的尺寸
# run_times： 每张图运行测试的次数  loader读取数据的方式不用这个参数
# model_name： 模型名称，可以是多个模型同时测试，就将模型名称用 逗号 隔开
# work_dir： 包含预训练模型的文件夹名称（模型名称位置 用 %s 替换，在程序中会自动替换为模型名称）
######################################################################### 

################ 同时对比 efficient-b0 和 mobilenet-v3的效果 ##############

################################# test ################################## 

model_name='efficientnet-b0,unet'
work_dir='g_%s_0.001_wt_f_o_0_1_0'
patch_size='64' 


## 在 测试图象上 对比 efficientnet  和  mobilenet_v3_small 的预测性能，这种测试方式是针对没有gt的图象
image_dir='/home/arc-crw5713/data/noise_level/test_images'   
test_mode='dir'
crop_sel='fixed'
tta_metric='mean' 

## 读取 .h5 文件 的方式 测试平均损失，在验证集/训练集上测试平均损失
# test_mode='loader'
# crop_sel='random'  ## 不能使用tta  
# tta_metric='mean'

## 以 txt 文件的形式测试，可以是训练集的子集或者验证集的子集，图片名和对应的注释以 txt 文件存储 
# image_dir='/home/arc-crw5713/data/noise_level/val_sub.txt'  
# test_mode='txt'
# crop_sel='area'   
# tta_metric='mean'


python test.py --work_dir $work_dir --model_name $model_name --crop_sel $crop_sel --image_dir $image_dir \
                --patch_size $patch_size  --test_mode $test_mode --use_tta --tta_metric $tta_metric \
                --run_times 1


############################### 参数意义 ################################
# anno_dir: test程序输出存储的txt结果文件路径，在对应的 work_dir下
# image_dir： 包含测试图片的路径 将上面输出的txt文件中的crop结果和预测得分在图片
#             上标注
# mode： 可视化的模式
#       0：表示将anno_dir中的结果 在 图上标注，标注crop的区域和对应的得分
#       1：表示统计50次运行的平均误差和方差 （需要将上面的 run_times 设置为 50）
######################################################################### 

# ############################### visual ################################## 
anno_dir='g_unet_0.001_wt_f_o_0_1_0/imagedir_pred_results.txt'
python visual.py --anno_dir $anno_dir --image_dir $image_dir --mode 0
anno_dir='g_efficientnet-b0_0.001_wt_f_o_0_1_0/imagedir_pred_results.txt'
python visual.py --anno_dir $anno_dir --image_dir $image_dir --mode 0