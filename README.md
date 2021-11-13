# CIFAR10_DNN


Homework #1 for AI Security

学习构建并训练CNN网络，来对CIFAR-10数据集进行图像分类

## CIFAR-10数据集

### 简介

[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)是由Hinton的学生Alex Krizhevsky和Ilya Sutskever整理的一个用于普适物体的小型数据集。它一共包含10个类别的RGB彩色图片：飞机、汽车、鸟类、猫、鹿、狗、蛙类、马、船。

数据集包含50000张训练图片和10000张测试图片，数据集中的图像大小为32x32x3。

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


