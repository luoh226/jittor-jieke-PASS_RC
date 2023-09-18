| 第三届计图挑战赛开源模板

# Jittor 大规模无监督语义分割赛题 PASS_RC
<img src="D:\DeepLearning\Jittor\img\readme\ILSVRC2012_val_00010974.JPEG" alt="ILSVRC2012_val_00010974" style="zoom:50%;" />                  <img src="D:\DeepLearning\Jittor\img\readme\箭头.png" alt="箭头" style="zoom:67%;" />                     <img src="D:\DeepLearning\Jittor\img\readme\ILSVRC2012_val_00010974_mask.png" alt="ILSVRC2012_val_00010974_mask" style="zoom:50%;" />



![矫正对比图](D:\DeepLearning\Jittor\img\readme\矫正对比图.png)

## 简介
| 简单介绍项目背景、项目特点

本项目包含了第三届计图挑战赛计图 - 大规模无监督语义分割赛题的代码实现。本项目的特点是：采用了RC显著性检测方法对PASS生成的FG Mask进行矫正，生成了更加接近GT Mask的前景掩膜，最终分割效果比PASS提高了2.32%。

## 安装 
本项目可在 2 张 3090 上训练，训练时间约为 40 小时。在 1 张3090上测试大约需要 1 小时。

#### 运行环境
- ubuntu 20.04 LTS
- python >= 3.8
- jittor >= 1.3.7.16

#### 安装依赖
执行以下命令安装 python 依赖
```
pip install -r requirements.txt
```

## 训练

首先运行：

```
bash train_1.sh
```

然后根据输出的信息找到best_mIoU最高的checkpoint以及对应的阈值，将其填入train_2.sh的BEST_CHECKPOINT以及BEST_THRESH参数，再运行train_2.sh。

```
bash train_2.sh
```

## 推理

生成测试集上的结果可以运行以下命令：

```
bash test.sh
```

## 致谢
此项目基于论文*Large-scale Unsupervised Semantic Segmentation* 实现，部分代码参考了 [SaliencyRC]([congve1/SaliencyRC: This the python implementation of the saliency cut method proposed in the paper "Global Contrast based Salient Region Detection" (github.com)](https://github.com/congve1/SaliencyRC))。
