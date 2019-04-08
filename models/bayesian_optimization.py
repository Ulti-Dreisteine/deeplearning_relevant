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
		plt.figure(figsize = [6, 4])
		plt.title('Gaussian Process')
		for i in range(samples_len):
			plt.plot(samples[i])
			plt.xlabel('time')
			plt.ylabel('value')
			plt.grid(True)
			plt.tight_layout()

	return samples


def objective_func(x):
	"""
	目标函数
	:param x: np.array, 一维或多维自变量, 不接受单一数值输入
	:return: y: float, 目标函数值
	"""
	if isinstance(x, list):
		x = np.array(x)
	else:
		x = np.array([x])

	# y = np.sin(x[0]) + np.cos(x[1]) ** 3 + 0.01 * x[0] ** 2 + 0.01 * x[1] ** 2
	y = (0.15 * x[0]) ** 2 - np.sin(x[0]) + np.cos(3 * x[0]) - np.sin(5 * x[0]) + np.cos(7 * x[0])
	return y


def sub_objective_func(objective_func, x, param_value, param_loc):
	"""
	固定其他维度值，求取某个子维度上的目标函数，这样可以使用一维贝叶斯优化求解器
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
	y_true = [obj_func(p) for p in xs]

	if show_plot:
		plt.title('Bayesian Optimization')
		# 显示真实值
		plt.plot(xs, y_true, '-', color = '0.5')
		plt.scatter(x_obs, y_obs)

		# 显示估计上下界
		plt.plot(xs, upper_bound, 'k--')
		plt.plot(xs, lower_bound, 'k--')
		plt.fill_between(xs, upper_bound, lower_bound, facecolor = 'lightgray')
		plt.legend(['true value', 'guessed bounds'])

		# 标记最优点
		plt.scatter(xs[np.argmin(lower_bound)], np.min(lower_bound), marker = '*', color = 'r', s = 80)

		plt.xlim([np.min(xs), np.max(xs)])
		plt.xlabel('params')
		plt.ylabel('values')
		plt.grid(True)
		plt.tight_layout()

	optimal_x = xs[np.argmin(lower_bound)]  # attention: 求取最小值

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
	if (param_dim != x_obs.shape[1]) | (param_dim != len(bounds)) | (param_dim != len(resolutions)):
		raise ValueError('param_dim与x_obs、bounds或resolutions的设置不匹配')

	x_obs = copy.deepcopy(x_obs)
	x_obs = x_obs.astype(float)

	for step in range(steps):
		print('\nstep = %s' % step)

		if show_plot & (step > 0):
			plt.clf()

		for param_loc in range(param_dim):
			print('processing param_loc %s' % param_loc)
			xs = np.linspace(bounds[param_loc][0], bounds[param_loc][1], resolutions[param_loc])
			sub_obj = lambda x: sub_objective_func(objective_func, x_obs[-1, :], x, param_loc)
			sub_x_obs = np.array([copy.deepcopy(x_obs)[-1, param_loc]])  # TODO: 修改 x_obs[:, param_loc] -> np.array([x_obs[-1, param_loc]])
			print('sub_x_obs = %s' % sub_x_obs)

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
					if np.abs(sub_x_obs[-1] - sub_x_obs[-2]) / np.abs(sub_x_obs[-2]) < eps:  # 相对误差控制
						break

			print('endding epoch = %s, param_loc = %s, optimal_value = %.4f, optimal_func_value = %.4f' % (epoch, param_loc, optimal_x, sub_obj(optimal_x)))
			new_obs = x_obs[-1, :]
			new_obs[param_loc] = sub_x_obs[-1]

			x_obs = np.vstack((x_obs, np.array(new_obs)))

		# 终止step迭代条件
		if (x_obs.shape[0] > 1) & (np.linalg.norm(x_obs[-1, :] - x_obs[-2, :]) / np.linalg.norm(x_obs[-2, :]) < eps):  # 相对误差控制
			break

	return x_obs


if __name__ == '__main__':
	# # 生成高维高斯采样样本
	# dim = 100
	# samples_len = 5
	# cov_matrix = kernal_func(np.linspace(0, 1, dim), np.linspace(0, 1, dim), sigma = 1, l = 0.1)  # 相邻维数之间维度编号距离越近，相关性越强
	# samples = multivariate_gaussian_sampling(dim, samples_len, cov_matrix, show_plot = False)  # TODO: 为什么这一步可以平滑？

	# import seaborn as sns
	# plt.figure(figsize = [5, 4])
	# plt.title('Covariance Matrix')
	# sns.heatmap(cov_matrix)
	# plt.xlabel('variable num')
	# plt.ylabel('variable num')
	# plt.grid(True)
	# plt.tight_layout()

	# 进行一维贝叶斯寻优
	dim = 400
	xs = np.linspace(-60, 60, dim)
	epochs = 200
	x_obs = np.array([-4])

	plt.figure('bayesian optimization', figsize = [8, 6])
	for epoch in range(epochs):
		miu_s, sigma_s, optimal_x = one_dimensional_bayesian_optimization(objective_func, xs, x_obs, show_plot = True, sigma = 50.0, l = 5)
		plt.xlim([-60, 60])
		plt.ylim([-10, 40])
		plt.pause(0.3)

		x_obs = np.hstack((x_obs, np.array([optimal_x])))

		if epoch != (epochs - 1):
			plt.clf()

	# # 进行二维贝叶斯寻优
	# x_obs = np.array([[-9, 2]]).reshape(1, -1)  # attention: 一定要reshape
	# bounds = [[-30, 30], [-30, 30]]
	# resolutions = [300, 300]
	# steps = 100
	# epochs = 100
	# param_dim = 2
	# eps = 1e-6
	# show_plot = False
	#
	# x_obs = multivariate_bayesian_optimization(objective_func, x_obs, bounds, resolutions, steps, epochs, param_dim, eps, show_plot = show_plot)
	#
	# # 验证二维寻优结果
	# x, y = np.linspace(bounds[0][0], bounds[0][1], 101), np.linspace(bounds[1][0], bounds[1][1], 101)
	# mesh_x, mesh_y = np.meshgrid(x, y)
	# xx = np.stack((mesh_x, mesh_y), axis = 0)
	# z = objective_func(xx)
	# plt.contourf(mesh_x, mesh_y, z, 30)
	# plt.scatter(x_obs[-1, 0], x_obs[-1, 1], marker = '*', color = 'r', s = 80)
	# plt.xlabel('x')
	# plt.ylabel('y')
	# plt.tight_layout()












