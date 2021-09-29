# _*_ coding:utf-8_*_
# 编写人员：王桢罡
# 编写时间：2021/1/6 10:35
# 文件名称：train
# 开发工具：pycharm
import torch
import torch.nn as nn
from model import LeNet
from torch.utils.tensorboard import SummaryWriter

"""
    导入数据集MNIST数据集，代码类似于data.py文件。
"""
##导入MNIST数据集
from torchvision.datasets import MNIST ##torchvision包含一些常用的数据集、模型、转换函数等等。
import torchvision.transforms as transforms
from torch.utils.data import DataLoader  ##DataLoader是一个数据载入类。

##一个训练集，一个测试集
data_train = MNIST('./data',download=True ##数据下载到本地
                   ,transform=transforms.Compose([
        transforms.Resize((32,32)), #转为Lenet-5需要的32*32格式，原始是28*28。
        transforms.ToTensor() ##将图像转变为浮点型张量
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

"""##########################################################################"""

model = LeNet()
model.train()  ##切换模型到训练状态
lr = 0.01 ##定义学习率
criterion = nn.CrossEntropyLoss() ##定义交叉熵损失函数
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4) ##定义随机梯度下降优化器
##momentum参数可以实现，当此次更新和上次更新方向相同时，加速更新。此次更新和上次更新方向相反时，减缓更新。
##weight_decay称权值衰减，实际上是L2正则化。

##可视化
writer = SummaryWriter()
#
step=1
for epoch in range(10):
    train_loss = 0
    total = 0
    correct = 0

    for batch_idx, (inputs, targets) in enumerate(data_train_loader):
        optimizer.zero_grad() ##清空梯度
        outputs = model(inputs) ##计算输出结果
        loss = criterion(outputs, targets) ##计算损失函数
        loss.backward() ##自动求导，反向传播
        optimizer.step() ##参数优化

        train_loss += loss.item()
        _, predicted = outputs.max(1) ##返回最大数值所在位置代表的类别，不加"_,"返回最大值。
        total += targets.size(0) ##计算参与训练样本总量
        correct += predicted.eq(targets).sum().item() ##预测结果和真实情况相符则为正确。

        writer.add_scalar(tag="Loss/train", scalar_value=loss, global_step=step)
        step += 1

        print(step, len(data_train_loader), 'Loss: %.3f | Acc: %.3f%%(%d/%d)'
              % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
"""--------------------------------------------------------------------------"""
##保存模型
torch.save(model.state_dict(), "./model.pkl") ##把参数保存为字典状态

# ./tensorboard --logdir=F:\pycharm\wzglianxi\神经网络\Pytorch\Lenet-5\runs\Jan08_10-22-33_LAPTOP-D80V03QS