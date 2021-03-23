# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 09:36:24 2021

@author: lhsin
"""

class Student(object):
    ## 构造方法
    def __init__(self, name, score):
        self.name = name
        self.score = score

    def get_grade(self):
        if self.score >= 90:
            return 'A'
        elif self.score >= 60:
            return 'B'
        else:
            return 'C'


class Animal(object):
    def __init__(self, name , weight):
        self.name = name
        self.weight = weight
        # self.__weight = weight
        print ( "this is an animal")
        print ( "name: %s" %( self.name) )
    
    def print_weight(self):
        print(self.weight)
        
    def upgrade_weight(self,w):
        self.weight = w

    
    def run(self):
        print('Animal is running...')
        
        

        
# 我们在子类中定义了和父类同名的方法，那么子类的方法就会覆盖父类的方法

class Cat(Animal):
    def run(self):
        print('Cat is running...')

# 使用super关键字实现了对父类方法的改写

class Dog(Animal):
    def __init__(self,name,weight): 
        super(Dog,self).__init__(name,weight)
        print( "besides,this is a dog" )    
  
        
if __name__=='__main__':  
    dog1 = Dog("snoopy",60)
    print("\n")
    cat1 = Cat("Garfield",120)
    
    # print(isinstance(cat1, Cat))
    # print(isinstance(cat1, Animal))
    # print(isinstance(cat1, Dog))























