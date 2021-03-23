# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 13:48:34 2021

@author: lhsin
"""

# 1.进入调试按钮（Debug File），程序到达你设置的第一个断点，这是进入断点调试必须的第一步；
# 2.单步调式按钮（Run current line）就可以在设置的断点之后单步调式（即仅测试当前的行）；
# 3.进入到函数体内部（Step into function or method of current line），用于查看当前使用的函数的明细代码；
# 4.Run until current function or method returns，这个方法恰好与3相反，是跳出函数体运行；
# 5.执行直到断点（Continue Exetution until next breakpoint），这是用的最多的一个方法；
# 6.退出调试，（Stop debug，点击或者按快捷键Ctrl+shift+F11）。


import numpy as np

def fun1():
    a=[1,2,4]
    b=[4,5,2]
    c=a+b
    d=a+c
    print(c)
    
def fun2():
    w = -1
    k = 3
    y = np.abs(w*k)
    return y

test1 = fun1()
test2 = fun2()


