# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 14:32:52 2018

@author: luolei

一维函数的贝叶斯优化
"""
import numpy as np
import matplotlib.pyplot as plt


def multivariate_gaussian_sampling(dim, samples_len, cov_matirx, mius = None, show_plot = False):
	"""
	多维高斯分布采样
	:param mius:
	:param show_plot:
	:param dim: int, 样本维数
	:param samples_len: int, 样本长度
	:param cov_matirx: np.ndarray, 样本各维度间的协方差矩阵
	:return:
	"""
	if mius is None:
		mius = np.zeros(dim)

	samples = []
	for i in range(samples_len):
		samples.append(np.random.multivariate_normal(mius, cov_matirx))

	if show_plot:
		plt.figure()
		for i in range(samples_len):
			plt.plot(samples[i])

	return samples


def kernal_fn(series_0, series_1, sigma = 10.0, l = 1.0):
	"""核函数"""
	dx = np.expand_dims(series_0, 1) - np.expand_dims(series_1, 0)
	return (sigma ** 2) * np.exp(-((dx / l) ** 2) / 2)


def objective_fn(x):
	return 5 * np.sin(x) - 3 * np.cos(2 * x) - x ** 2


def beyesian_optimization(objective_func, xs, x_obs, show_plot = False):
	"""
	贝叶斯寻优
	:param objective_func: func, 待优化的目标函数
	:param xs: array, 寻优参数点
	:param x_obs: 已有观测参数点
	:param show_plot: 是否显示寻优结果
	:return:
	"""
	y_obs = objective_func(x_obs)

	k = kernal_fn(x_obs, x_obs)
	k_s = kernal_fn(x_obs, xs)
	k_ss = kernal_fn(xs, xs)
	k_sTk_inv = np.matmul(k_s.T, np.linalg.pinv(k))

	miu_s = np.mean(xs.reshape(1, -1), axis = 0) + np.matmul(k_sTk_inv, y_obs - np.mean(x_obs.reshape(1, -1), axis = 0))
	sigma_s = k_ss - np.matmul(k_sTk_inv, k_s)

	std_var = np.sqrt(np.abs(sigma_s.diagonal()))
	upper_bound = miu_s + std_var
	lower_bound = miu_s - std_var
	y_true = objective_fn(xs)
	if show_plot:
		# 显示估计上下界
		plt.plot(xs, upper_bound, 'k--')
		plt.plot(xs, lower_bound, 'k--')
		plt.fill_between(xs, upper_bound, lower_bound, facecolor = 'lightgray')

		# 显示真实值
		plt.plot(xs, y_true, '--', color = '0.5')
		plt.scatter(x_obs, y_obs)
		# 标记最优点
		print(xs[np.argmax(upper_bound)], np.max(upper_bound))
		plt.scatter(xs[np.argmax(upper_bound)], np.max(upper_bound), marker = '*', color = 'r', s = 80)

		plt.xlim([np.min(xs), np.max(xs)])
		plt.xlabel('x')
		plt.ylabel('y')
		plt.grid(True)
		plt.tight_layout()

	optimal_x = xs[np.argmax(upper_bound)]

	return miu_s, sigma_s, optimal_x


if __name__ == '__main__':
	# 生成高维高斯采样样本
	dim = 10
	samples_len = 5
	cov_matrix = kernal_fn(np.linspace(0, 1, dim), np.linspace(0, 1, dim))  # 相邻维数之间维度编号距离越近，相关性越强
	samples = multivariate_gaussian_sampling(dim, samples_len, cov_matrix, show_plot = False)  # TODO: 为什么这一步可以平滑？

	# 进行贝叶斯寻优
	dim = 80
	xs = np.linspace(-10, 10, dim)
	epochs = 20
	x_obs = np.array([-4])

	plt.figure('bayesian optimization', figsize = [8, 6])
	for epoch in range(epochs):
		miu_s, sigma_s, optimal_x = beyesian_optimization(objective_fn, xs, x_obs, show_plot = True)
		plt.pause(1.0)

		if epoch != epochs - 1:
			plt.clf()
		x_obs = np.hstack((x_obs, np.array([optimal_x])))




