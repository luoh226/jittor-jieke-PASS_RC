| 第三届计图挑战赛

# Jittor 大规模无监督语义分割赛题 PASS_RC
任务目标

<img src=".\img\ILSVRC2012_val_00010974.JPEG" alt="ILSVRC2012_val_00010974" style="zoom:10%;" />                  <img src=".\img\箭头.png" alt="箭头" style="zoom:17%;" />                     <img src=".\img\ILSVRC2012_val_00010974_mask.png" alt="ILSVRC2012_val_00010974_mask" style="zoom:10%;" />

矫正前后对比图（第1、3列）

<img src=".\img\矫正对比图.png" alt="ILSVRC2012_val_00010974_mask" style="zoom:50%;" align="left"/>

## 简介
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
此项目基于论文*Large-scale Unsupervised Semantic Segmentation* 实现，部分代码参考了 [SaliencyRC](https://github.com/congve1/SaliencyRC)。
