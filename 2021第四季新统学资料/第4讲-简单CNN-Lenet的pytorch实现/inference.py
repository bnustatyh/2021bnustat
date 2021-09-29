# _*_ coding:utf-8_*_
# 编写人员：王桢罡
# 编写时间：2021/1/6 11:02
# 文件名称：inference
# 开发工具：pycharm
import torch
import torch.nn as nn
from model import LeNet
##导入MNIST数据集
from torchvision.datasets import MNIST ##torchvision包含一些常用的数据集、模型、转换函数等等。
import torchvision.transforms as transforms
from torch.utils.data import DataLoader  ##DataLoader是一个数据载入类。

##一个训练集，一个测试集
data_train = MNIST('./data',download=True ##数据下载到本地
                   ,transform=transforms.Compose([ ##Compose()
        transforms.Resize((32,32)), #转为Lenet-5需要的32*32格式，原始是28*28。
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
"""--------------------------------------------------------------------------"""
##调用保存的模型
save_info = torch.load("./model.pkl") #载入模型
model = LeNet() #定义LeNet模型
criterion = nn.CrossEntropyLoss() #交叉熵损失函数
model.load_state_dict(save_info)  ##载入模型参数
model.eval() ##切换模型到测试状态

test_loss = 0
correct = 0
total = 0
with torch.no_grad(): ##关闭计算图
    for batch_idx, (inputs, targets) in enumerate(data_test_loader):
        # print(batch_idx, "\n")
        # print(inputs, inputs.size(), "\n")
        # print(targets, targets.size(), "\n")

        outputs = model(inputs)
        loss = criterion(outputs, targets) ##计算损失

        test_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0) ##计算参与训练样本总量
        correct += predicted.eq(targets).sum().item() ##预测结果和真实情况相符则为正确。

        print(batch_idx, len(data_test_loader), 'Loss: %.3f | Acc: %.3f %%(%d/%d)'
              % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))