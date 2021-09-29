# _*_ coding:utf-8_*_
# 编写人员：王桢罡
# 编写时间：2021/1/5 22:23
# 文件名称：data
# 开发工具：pycharm

##导入MNIST数据集
from torchvision.datasets import MNIST ##torchvision包含一些常用的数据集、模型、转换函数等等。
import torchvision.transforms as transforms
from torch.utils.data import DataLoader  ##DataLoader是一个数据载入类。

##一个训练集，一个测试集
data_train = MNIST('./data',download=True ##数据下载到本地
                   ,transform=transforms.Compose([ ##Compose()
        transforms.Resize((32,32)), #转为Lenet-5需要的32*32格式，原始是28*28。默认是双线性插值。
        transforms.ToTensor() ##将图像归一化(除以255)，转变为浮点型张量,并且将图像形状从[H,W,C]转为[C,H,W]
                              ##Height,Width,Channal
    ]))

data_test = MNIST('./data',
                  train=False,
                  download=True,
                  transform=transforms.Compose([
                      transforms.Resize((32,32)),
                      transforms.ToTensor()
                  ]))

##载入数据
data_train_loader = DataLoader(data_train, batch_size=256, shuffle=True)
data_test_loader = DataLoader(data_test, batch_size=1024, shuffle=True)

#以下是图形展示MNIST
import matplotlib.pyplot as plt
figure = plt.figure()
num_of_images = 60

for imgs, targets in data_train_loader:
    break

for index in range(num_of_images):
    plt.subplot(6, 10, index+1)
    plt.axis('off')
    img = imgs[index, ...]
    plt.imshow(img.numpy().squeeze(), cmap='gray_r')
plt.show()