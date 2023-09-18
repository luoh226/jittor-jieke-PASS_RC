| 第三届计图挑战赛开源模板

# Jittor 大规模无监督语义分割赛题 PASS_RC
<img src="D:\DeepLearning\Jittor\img\readme\ILSVRC2012_val_00010974.JPEG" alt="ILSVRC2012_val_00010974" style="zoom:50%;" />                  <img src="D:\DeepLearning\Jittor\img\readme\箭头.png" alt="箭头" style="zoom:67%;" />                     <img src="D:\DeepLearning\Jittor\img\readme\ILSVRC2012_val_00010974_mask.png" alt="ILSVRC2012_val_00010974_mask" style="zoom:50%;" />



![矫正对比图](D:\DeepLearning\Jittor\img\readme\矫正对比图.png)

## 简介
| 简单介绍项目背景、项目特点

本项目包含了第三届计图挑战赛计图 - 大规模无监督语义分割赛题的代码实现。本项目的特点是：采用了RC显著性检测方法对PASS生成的FG Mask进行矫正，生成了更加接近GT Mask的前景掩膜，最终分割效果比PASS提高了2.32%。

## 安装 
| 介绍基本的硬件需求、运行环境、依赖安装方法

本项目可在 2 张 3090 上训练，训练时间约为 40 小时。在 1 张3090上测试大约需要 1 小时。

#### 运行环境
- ubuntu 20.04 LTS
- python >= 3.7
- jittor >= 1.3.0

#### 安装依赖
执行以下命令安装 python 依赖
```
pip install -r requirements.txt
```

#### 预训练模型
预训练模型模型下载地址为 https:abc.def.gh，下载后放入目录 `<root>/weights/` 下。

## 数据预处理
| 介绍数据预处理方法，可选

将数据下载解压到 `<root>/data` 下，执行以下命令对数据预处理：
```
bash scripts/prepross.sh
```

## 训练
｜ 介绍模型训练的方法

单卡训练可运行以下命令：
```
bash scripts/train.sh
```

多卡训练可以运行以下命令：
```
bash scripts/train-multigpu.sh
```

## 推理
｜ 介绍模型推理、测试、或者评估的方法

生成测试集上的结果可以运行以下命令：

```
bash scripts/test.sh
```

## 致谢
| 对参考的论文、开源库予以致谢，可选

此项目基于论文 *A Style-Based Generator Architecture for Generative Adversarial Networks* 实现，部分代码参考了 [jittor-gan](https://github.com/Jittor/gan-jittor)。

## 注意事项

点击项目的“设置”，在Description一栏中添加项目描述，需要包含“jittor”字样。同时在Topics中需要添加jittor。

![image-20220419164035639](https://s3.bmp.ovh/imgs/2022/04/19/6a3aa627eab5f159.png)