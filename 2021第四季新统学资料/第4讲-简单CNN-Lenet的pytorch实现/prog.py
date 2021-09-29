# _*_ coding:utf-8_*_
# 编写人员：王桢罡
# 编写时间：2021/1/11 9:49
# 文件名称：prog
# 开发工具：pycharm
"""使用argparse库来处理Python脚本的参数，方便对深度学习超参数的调优"""
import argparse

##构造一个参数处理器的实例
parser = argparse.ArgumentParser(description='Pytorch LeNet Training')

##添加两个参数
parser.add_argument('--lr', default=0.01, type=float, help='Learning rate')
parser.add_argument('--batch-size', '-b', default=256, type=int, help='Batchsize')

args = parser.parse_args()