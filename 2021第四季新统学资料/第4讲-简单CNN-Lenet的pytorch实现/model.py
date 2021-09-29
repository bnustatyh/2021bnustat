# _*_ coding:utf-8_*_
# 编写人员：王桢罡
# 编写时间：2021/1/6 10:22
# 文件名称：model
# 开发工具：pycharm
import torch
import torch.nn as nn

class LeNet(nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()  ##定义模型的结构
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5) ##二维卷积层,stride默认为1。
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) ##二维最大值池化
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(16, 120, 5)
        ##全连接层，输入为(N,in_features)的张量，输出为(N,out_features)的张量
        self.fc3 = nn.Linear(in_features=120, out_features=84)
        self.fc4 = nn.Linear(84, 10)

    def forward(self, x):  ##定义张量的运算过程
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = torch.relu(self.conv3(x))
        print(x.shape)
        x = x.view(x.size(0), -1) ##调整张量的形状，张量内的数据元素不会变，某一维是-1时，会自动计算。
        print(x.shape)
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

if __name__ == '__main__':
    model = LeNet()
    ret = model(torch.randn(1, 1, 32, 32))  ##四维分别针对batch_size、channels、height、width
    print(ret.shape)
    print(ret.max())