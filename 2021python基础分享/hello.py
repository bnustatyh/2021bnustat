# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 09:07:13 2021

@author: lhsin
"""
import sys

# sys模块有一个argv变量，用list存储了命令行的所有参数
# argv至少有一个元素，第一个参数永远是该python文件的名称

def test():
    args = sys.argv
    if len(args)==1:
        print('Hello, world!')
    elif len(args)==2:
        print('Hello, %s!' % args[1])
    else:
        print('Too many arguments!')
        
def _private_1(name):
    print('Hello, %s' % name)

def _private_2(name):
    print('Hi, %s' % name)

def greeting(name):
    if len(name) > 3:
        return _private_1(name)
    else:
        return _private_2(name)

# 在命令行运行hello模块文件时，Python解释器把一个特殊变量__name__置为__main__
# 而如果在其他地方导入该hello模块时，下面的if判断将失败
# 这种if测试可以让一个模块通过命令行运行时执行一些额外的代码，比如运行测试。


if __name__=='__main__':
    test()

    


    
  
    
    
    
    