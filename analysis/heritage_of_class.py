# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 14:32:52 2018

@author: luolei

类的继承关系
"""


# class A(object):
# 	def __init__(self):
# 		print('enter A')
# 		print('leave A')
#
#
# class A2(A):
# 	def __init__(self):
# 		print('enter A2')
# 		print('leave A2')
#
#
# class B(A2, A):
# 	def __init__(self):
# 		print('enter B')
# 		super(B, self).__init__()
# 		print('leave B')
#
#
# class B2(A):
# 	def __init__(self):
# 		print('enter B2')
# 		super(B2, self).__init__()
# 		print('leave B2')
#
#
# class C(B, B2):
# 	def __init__(self):
# 		print('enter C')
# 		super(C, self).__init__()
# 		print('leave C')
#
#
# B()


class Person(object):
	def __init__(self, name, age):
		self.name = name
		self.age = age
		self.weight = 'weight'

	def talk(self):
		print('{} is talking....'.format(self.name))


class Chinese(Person):
	def __init__(self, name, age, language):  # 先继承，在重构
		Person.__init__(self, name, age)  # 继承父类的构造方法，也可以写成：super(Chinese,self).__init__(name,age), 表示调用Chinese的父类的__init__方法把name和age加在self上
		self.language = language  # 定义类的本身属性

	def walk(self):
		print('{} is walking...'.format(self.name))

	def talk(self):  # 子类对父类中函数的改写
		print('{} is talking too...'.format(self.name))


chinese = Chinese('Dreisteine', '27', 'chinese')
chinese.talk()
chinese.walk()
