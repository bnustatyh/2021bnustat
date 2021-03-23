# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 14:56:48 2021

@author: lhsin
"""

string = ['1','2','3']

print(list(map(float,string)))


print(map(lambda x: x*x,[y for y in range(10)]))
