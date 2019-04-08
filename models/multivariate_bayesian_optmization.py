# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 14:32:52 2018

@author: luolei

多元函数的贝叶斯参数优化
"""
import numpy as np
import matplotlib.pyplot as plt
import copy
import sys

sys.path.append('../')


# def objective_func(x):
# 	"""
# 	目标函数
# 	:param x: np.array, 一维或多维自变量
# 	:return: y: float, 目标函数值
# 	"""
# 	x = np.array(x)
# 	y = np.sin(x[0]) + np.cos(x[1]) ** 2 + 0.01 * x[0] ** 2 + 0.01 * x[1] ** 2
# 	return y


def sub_objective_func(objective_func, x, param_value, param_loc):
	"""
	固定其他维度值，求取某个子维度上的目标函数
	:param objective_func: function, 需要定义的目标函数
	:param x_obs: np.array, 观测值向量, shape = (n,)
	:param param_value: float, 待修改的参数值
	:param param_loc: int, 待修改的参数位置
	:return: y: float, 子目标函数值
	"""
	y = copy.deepcopy(x)
	y[param_loc] = param_value  # fixme: 添加一维的情况
	sub_obj = objective_func(y)
	return sub_obj


def kernal_fn(series_0, series_1, sigma = 1.0, l = 1.0):  # TODO: 研究sigma和l参数对模型计算效率的影响
	"""核函数"""
	dx = np.expand_dims(series_0, 1) - np.expand_dims(series_1, 0)
	return (sigma ** 2) * np.exp(-((dx / l) ** 2) / 2)


def kernal_func(array_0, array_1, sigma = 1.0, l = 1.0):  # TODO: 研究sigma和l参数对模型计算效率的影响
	"""核函数, 用于计算两个array的协方差矩阵"""
	dx = np.expand_dims(array_0, 1) - np.expand_dims(array_1, 0)
	return (sigma ** 2) * np.exp(-((dx / l) ** 2) / 2)


def one_dimensional_bayesian_optimization(obj_func, xs, x_obs, show_plot = False, **kwargs):
	"""一维贝叶斯参数优化"""
	y_obs = np.array([obj_func(p) for p in x_obs])  # TODO: 这一步需要优化，每步迭代并不需要全部计算

	k = kernal_func(x_obs, x_obs, **kwargs)
	k_s = kernal_func(x_obs, xs, **kwargs)
	k_ss = kernal_func(xs, xs, **kwargs)
	k_sTk_inv = np.matmul(k_s.T, np.linalg.pinv(k))

	miu_s = np.mean(xs.reshape(1, -1), axis = 0) + np.matmul(k_sTk_inv, y_obs - np.mean(x_obs.reshape(1, -1), axis = 0))
	sigma_s = k_ss - np.matmul(k_sTk_inv, k_s)

	std_var = np.sqrt(np.abs(sigma_s.diagonal()))
	upper_bound = miu_s + std_var
	lower_bound = miu_s - std_var
	# y_true = [obj_func(p) for p in xs]

	if show_plot:
		# 显示估计上下界
		plt.plot(xs, upper_bound, 'k--')
		plt.plot(xs, lower_bound, 'k--')
		plt.fill_between(xs, upper_bound, lower_bound, facecolor = 'lightgray')

		# # 显示真实值
		# plt.plot(xs, y_true, '--', color = '0.5')
		# plt.scatter(x_obs, y_obs)

		# 标记最优点
		plt.scatter(xs[np.argmax(upper_bound)], np.max(upper_bound), marker = '*', color = 'r', s = 80)

		plt.xlim([np.min(xs), np.max(xs)])
		plt.xlabel('x')
		plt.ylabel('y')
		plt.grid(True)
		plt.tight_layout()

	optimal_x = xs[np.argmax(upper_bound)]

	return miu_s, sigma_s, optimal_x


def multivariate_bayesian_optimization(objective_func, x_obs, bounds, resolutions, steps, epochs, param_dim, eps, show_plot = False, **kwargs):
	"""
	多元贝叶斯优化
	:param objective_func: function, 目标函数
	:param x_obs: np.array, 观测值, shape = (-1, param_dim)
	:param bounds: list, 各维度上搜索的最低最高界限, [[param_0_min, param_0_max], [param_1_min, param_1_max], ...]
	:param resolutions: list, 各维度上的分辨率, [res_0, res_1, ...]
	:param steps: int, 优化迭代的次数
	:param epochs: int, 每个维度上迭代的次数
	:param param_dim: int, 参数维数
	:param eps: 相对精度控制
	:param show_plot: bool, 显示过程
	:return: x_obs: np.ndarray, 观测值优化过程记录
	"""
	for step in range(steps):
		print('step = %s' % step)

		if show_plot & (step > 0):
			plt.clf()

		new_obs = []
		for param_loc in range(param_dim):
			xs = np.linspace(bounds[param_loc][0], bounds[param_loc][1], resolutions[param_loc])
			sub_obj = lambda x: sub_objective_func(objective_func, x_obs[-1, :], x, param_loc)
			sub_x_obs = x_obs[:, param_loc]

			if show_plot & (param_loc > 0):
				plt.clf()

			for epoch in range(epochs):
				miu_s, sigma_s, optimal_x = one_dimensional_bayesian_optimization(sub_obj, xs, sub_x_obs, show_plot = show_plot, **kwargs)
				sub_x_obs = np.hstack((sub_x_obs, np.array([optimal_x])))

				if show_plot:
					plt.pause(0.5)
					if epoch != epochs - 1:
						plt.clf()

				# 终止epoch迭代条件
				if epoch > 0:
					if np.abs(sub_x_obs[-1] - sub_x_obs[-2]) / np.abs(sub_x_obs[-2]) < eps:
						break

			new_obs.append(optimal_x)

			print('param_loc = %s, optimal_x = %.4f, optimal_func_value = %.4f' % (param_loc, optimal_x, sub_obj(optimal_x)))

		x_obs = np.vstack((x_obs, np.array(new_obs)))

		# 终止step迭代条件
		if step > 0:
			if np.linalg.norm(x_obs[-1, :] - x_obs[-2, :]) < eps:
				break

	return x_obs








