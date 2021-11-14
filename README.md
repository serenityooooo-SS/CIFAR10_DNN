# CIFAR10_DNN


Homework #1 for AI Security

学习构建并训练CNN网络，来对CIFAR-10数据集进行图像分类

## CIFAR-10数据集

### 简介

[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)是由Hinton的学生Alex Krizhevsky和Ilya Sutskever整理的一个用于普适物体的小型数据集。它一共包含10个类别的RGB彩色图片：飞机、汽车、鸟类、猫、鹿、狗、蛙类、马、船。

CIFAR-10数据集包含10个类的60000张32x32的彩色图像,每个类有6000张图像,有50000张训练图像和10000张测试图像,数据集中的图像大小为32x32x3。

与MNIST手写数字数据集的区别：

| CIFAR-10         | MNIST         | 
| --------         | -----         |
| 3通道彩色RGB图像  | 灰度图像       | 
| 尺寸32x32         | 尺寸28x28     |  
| 比例、特征不同     | 特征较为明显   | 

所以线性模型在CIFAR-10表现很差。

<!-- ### 数据集下载

* [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) 直接下载
* 代码下载：https://github.com/tensorflow/models/tree/master/tutorials/image/cifar10 -->

### 网络结构

定义卷积神经网络的结构

这里，将定义一个CNN的结构。将包括以下内容：

* 卷积层：可以认为是利用图像的多个滤波器（经常被称为卷积操作）进行滤波，得到图像的特征。
* 通常，我们在 PyTorch 中使用 nn.Conv2d 定义卷积层，并指定以下参数：nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)
* 池化层：这里采用的最大池化：对指定大小的窗口里的像素值最大值。 在 2x2 窗口里，取这四个值的最大值。于最大池化更适合发现图像边缘等重要特征，适合图像分类任务。

网络结构如下：

imput images -- converlutional layer -- pooling layer -- converlutional layer -- pooling layer -- full-connected layer


```python
Net(
(conv1): Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
(conv2): Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
(conv3): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
(pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
(fc1): Linear(in_features=1024, out_features=500, bias=True)
(fc2): Linear(in_features=500, out_features=10, bias=True)
(dropout): Dropout(p=0.3, inplace=False)
)
```

### 选择损失函数与优化函数

```python
import torch.optim as optim
# 使用交叉熵损失函数
criterion = nn.CrossEntropyLoss()
# 使用随机梯度下降，学习率lr=0.01
optimizer = optim.SGD(model.parameters(), lr=0.01)
```

### 训练卷积神经网络模型

训练集和验证集的损失是如何随着时间的推移而减少的；如果验证损失不断增加，则表明可能过拟合现象。

（实际上，如果n_epochs设置为40，可以发现存在过拟合现象！所以代码中设为30）



