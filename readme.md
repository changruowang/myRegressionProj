# 制作数据集：
原始数据集 要按照下面格式存放
>   -image \
    ---- image1.jpg \
    ---- image2.jpg \
          ...       \
    -annotions.txt

输入参数含义：
* anno_dir： 原始注释文件路径
* save_dir： 生成的.h5文件输出存储路径
* make_category： 需要制作的类别（如果是新的数据集，则需要修改源代码中各个样本类别每张图采样的数量，否则制作出的数据集不一定平衡 Ynoise，strength）
* crop_size: 训练样本的尺寸
* mode： crop的依据，灰度阈值 or 局部方差阈值
``` 
python dataset.py --anno_dir 原始数据集标注文件路径  --save_dir 输出存储路径 --make_category Ynoise --crop_size 64 --mode gray
```
目前已经制作好的数据集：
* /home/arc-crw5713/data/crop_yoise_var64/      使用方差的原则筛选的训练patch  用于yoise训练
* /home/arc-crw5713/data/crop_yoise_gray64/     使用灰度阈值筛选的训练patch 用于yoise训练
* /home/arc-crw5713/data/crop_strength_gray64/  使用灰度阈值筛选的训练patch 用于stren训练
* /home/arc-crw5713/data/crop_strength_gray250/ 使用灰度阈值筛选的训练250大小的patch 用于stren训练，训练时可以再crop出更小的128patch输入网络

# 测试
直接运行 test_a.sh / ...  test_e.sh 脚本即可运行测试代码，测试参数的含义在 测试脚本中有注释

# 训练
直接运行 train_a.sh / .... train_e.sh 脚本即可重新运行模型，训练参数的含义在 测试脚本中有注释

# 可视化
可视化方式在 测试脚本中有注释 和 使用 详细用法见测试脚本中的 visual 部分

# 模型复杂度计算
较简单，直接查看 model_config.py 文件

# 模型文件夹命名规则
(g,d,v,g250) _ (backbone名字) _ (学习率) _ wt _ (t,f) _ o _ (1_0_0, 0_1_0, 1_1_0)

第一项：表示训练用的数据集类型。 
* g: 最基本的数据，即从大图上随机选择64x64,patch,然后以灰度阈值筛除过曝的patch
* v: 优先选局部方差较大的patch作为训练样本
* g250: 以灰度阈值删选patch，但是打包为.h5文件的patch尺寸是224，然后训练的时候再从224中随机扣128作为网络输入
* d：strength 和 ynoise 两个数据集交替使用，以同时训练两个属性

第二项：表示backbone名字。
* efficientnet-b0
* unet
* mobilenet-b0

第三项：训练用的学习率 前期曾对不同学习率做过测试，现在统一学习率都是用0.001训练的模型，所以该项没有变化
第四项：只训练一路输出时 设置为 f，同时训练两路输出时设置为 t

第五项： 表示是否训练对应的输出
* 1_0_0: 只训练了 ynoise 输出
* 0_1_0: 只训练了 strength 输出
* 1_1_0: 同时训练了 ynoise 和 strength输出 （这时要将 wt 设置为 t 以确保两个数据集是交替训练的）

## 已经训练好的模型
* g_efficientnet-b0_0.001_wt_f_o_0_1_0     effi模型，gray crop64训练集训练的 strength属性
* g_unet_0.001_wt_f_o_0_1_0                unet模型，gray crop64训练集训练的 strength属性
* v_efficientnet-b0_0.001_wt_f_o_0_1_0     effi模型，var crop64训练集训练的 strength属性
* g250_efficientnet-b0_0.001_wt_f_o_0_1_0  effi模型，var crop128训练集训练的 strength属性
* d_efficientnet-b0_0.001_wt_t_o_1_1_0     effi模型，gray crop64训练集训练的 ynoise_strength联合属性 
* d_mobilenet_v3_small_0.001_wt_t_o_1_1_0  mobile模型，gray crop64训练集训练的 ynoise_strength联合属性 