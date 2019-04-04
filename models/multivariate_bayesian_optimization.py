# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 14:32:52 2018

@author: luolei

多元函数的贝叶斯参数优化
"""
import numpy as np
import copy


def objective_fn(x):
	"""
	目标函数
	:param x: np.array, 一维或高维向量
	:return: y: float, 目标函数值
	"""
	y = np.linalg.norm(x, 2)
	return y


def sub_objective_fn(x_obs, param_value, param_loc):
	"""固定其他维度值，求取某个子维度上的目标函数"""
	x_obs = copy.deepcopy(x_obs)
	x_obs[param_loc] = param_value
	y = objective_fn(x_obs)
	return y


def kernal_fn(series_0, series_1, sigma = 1.0, l = 1.0):
	"""核函数"""
	dx = np.expand_dims(series_0, 1) - np.expand_dims(series_1, 0)
	return (sigma ** 2) * np.exp(-((dx / l) ** 2) / 2)


if __name__ == '__main__':
	# 确定待优化参数范围和精度
	param_dim = 2  # 待优化参数向量维数
	resolution = 100
	xs = np.linspace(-10, 10, resolution)
	epochs = 50

	# 初始化参数
	x_obs = 10 * np.random.random(2)
	y_obs = objective_fn(x_obs)

	# 一步迭代
	x_opt_epoch = []
	for param_loc in range(param_dim):
		y_obs = sub_objective_fn(x_obs, x_obs[param_loc], param_loc)

		k = kernal_fn(x_obs[param_loc], x_obs[param_loc])
		k_s = kernal_fn(x_obs[param_loc], xs)
		k_ss = kernal_fn(xs, xs)
		k_sTk_inv = np.matmul(k_s.T, np.linalg.pinv(k))

		miu_s = np.mean(xs.reshape(1, -1), axis = 0) + np.matmul(k_sTk_inv, y_obs - np.mean(x_obs.reshape(1, -1), axis = 0))
		sigma_s = k_ss - np.matmul(k_sTk_inv, k_s)

		std_var = np.sqrt(np.abs(sigma_s.diagonal()))
		upper_bound = miu_s + std_var
		lower_bound = miu_s - std_var

		optimal_x = xs[np.argmax(upper_bound)]
		x_opt_epoch.append(optimal_x)






