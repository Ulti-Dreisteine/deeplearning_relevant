# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 14:32:52 2018

@author: luolei

使用最小二乘解释过拟合
"""
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append('../')


def func(x, w, b):
	"""
	定义线性方程
	:param x: 自变量array
	:param w: 一次项系数
	:param b: 常数项
	:return: y: 结果array
	"""
	y = w * x + b
	return y


def fitting_func(x, ws):
	"""
	多项式拟合函数
	:param ws: 系数array, np.array([w0, w1, w2, w3, ...])
	:param x: 自变量
	:return: y: 拟合值
	"""
	order = len(ws) - 1
	x_arr = []
	for i in range(order + 1):
		x_arr.append(pow(x, i))
	x_arr = np.array(x_arr).T
	y = np.dot(np.array(ws).reshape(1, -1), x_arr)
	return y


def gen_samples(x, w, b, scale = 1, show_plot = False):
	"""
	生成样本
	:param scale:
	:param show_plot: 显示样本
	:param x: 自变量array
	:param w: 一次项系数
	:param b: 常数项
	:return: samples: 样本array
	"""
	y = func(x, w, b)
	samples = y + scale * np.random.random(len(x))

	if show_plot:
		plt.figure()
		plt.plot(x, samples)
		plt.grid(True)
		plt.tight_layout()
		plt.show()

	return samples


def cal_polynomial(x_arr, ws):
	"""
	最小二乘回归
	:param x_arr: 自变量arr
	:param ws: 系数arr
	:return: y_pred
	"""
	x_arr = x_arr.reshape(x_arr.shape[0], 1)
	y_pred = np.apply_along_axis(lambda a: fitting_func(a[0], ws), 1, x_arr)
	return y_pred


def least_square_fitting(x, y_true, order):
	"""
	最小二乘拟合
	:param x:
	:param y_true:
	:param order:
	:return:
	"""
	x = np.array(x).reshape(-1, 1)
	y_true = np.array(y_true).reshape(-1, 1)
	X = []
	for i in range(order):
		X.append(pow(x, i))
	X = np.array(X).reshape(order, -1).T
	best_params = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, y_true))
	return best_params


if __name__ == '__main__':
	# 生成样本
	x = np.arange(0, 20, 1)
	y_true = gen_samples(x, 2, 1, 50, show_plot = False)
	best_params = least_square_fitting(x, y_true, order = 10)

	plt.plot(x, y_true)
	plt.plot(x, cal_polynomial(x, best_params))

