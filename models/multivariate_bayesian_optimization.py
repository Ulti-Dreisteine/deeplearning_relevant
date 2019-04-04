# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 14:32:52 2018

@author: luolei

多元函数的贝叶斯参数优化
"""
import numpy as np
import copy
import matplotlib.pyplot as plt


def objective_fn(x):
	"""
	目标函数
	:param x: np.array, 一维或高维向量, shape = (1, -1)
	:return: y: float, 目标函数值
	"""
	y = np.sin(x[0, 0]) + np.cos(x[0, 1]) ** 2 + 0.01 * x[0, 0] ** 2 + 0.01 * x[0, 1] ** 2
	return y


def sub_objective_fn(x_obs, param_value, param_loc):
	"""固定其他维度值，求取某个子维度上的目标函数"""
	x_obs = copy.deepcopy(x_obs)
	y = []
	for i in range(len(param_value)):
		x_obs[0, param_loc] = param_value[i]
		y.append(objective_fn(x_obs))
	y = np.array(y)
	return y


def kernal_fn(series_0, series_1, sigma = 1.0, l = 1.0):  # TODO: 研究sigma和l参数对模型计算效率的影响
	"""核函数"""
	dx = np.expand_dims(series_0, 1) - np.expand_dims(series_1, 0)
	return (sigma ** 2) * np.exp(-((dx / l) ** 2) / 2)


def beyesian_optimization(objective_func, xs, x_obs, show_plot = False):
	"""
	贝叶斯寻优
	:param objective_func: func, 待优化的目标函数
	:param xs: array, 寻优参数点
	:param x_obs: 已有观测参数点
	:param show_plot: 是否显示寻优结果
	:return:
	"""
	y_obs = objective_func(x_obs) 	# TODO: 这一步需要优化，每步迭代并不需要全部计算

	k = kernal_fn(x_obs, x_obs)
	k_s = kernal_fn(x_obs, xs)
	k_ss = kernal_fn(xs, xs)
	k_sTk_inv = np.matmul(k_s.T, np.linalg.pinv(k))

	miu_s = np.mean(xs.reshape(1, -1), axis = 0) + np.matmul(k_sTk_inv, y_obs - np.mean(x_obs.reshape(1, -1), axis = 0))
	sigma_s = k_ss - np.matmul(k_sTk_inv, k_s)

	std_var = np.sqrt(np.abs(sigma_s.diagonal()))
	upper_bound = miu_s + std_var
	lower_bound = miu_s - std_var
	y_true = [objective_func([p]) for p in xs]
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
	# 确定待优化参数范围和精度
	param_dim = 2  # 待优化参数向量维数
	resolution = 100
	xs = np.linspace(-10, 10, resolution)
	steps = 50
	epochs = 50
	eps = 1e-4

	# 初始化参数
	x_obs = 10 * np.random.random(2).reshape(1, -1)
	for step in range(steps):
		print('step = %s' % step)
		y_obs = objective_fn(x_obs)

		# 一步迭代
		x_opt_epoch = []
		for param_loc in range(param_dim):
			x_sub_obs = x_obs[:, param_loc].flatten()
			for epoch in range(epochs):
				miu_s, sigma_s, optimal_x = beyesian_optimization(lambda x: sub_objective_fn(x_obs, x, param_loc), xs, x_sub_obs, show_plot = True)
				plt.pause(1.0)

				x_sub_obs = np.hstack((x_sub_obs, np.array([optimal_x])))

				if epoch > 0:
					if np.abs(x_sub_obs[-1] - x_sub_obs[-2]) < eps:
						break
				if epoch != epochs - 1:
					plt.clf()
			x_opt_epoch.append(x_sub_obs[-1])
			plt.clf()

		x_opt_epoch = np.array(x_opt_epoch).reshape(1, -1)
		x_obs = np.vstack((x_obs, x_opt_epoch))

		if step > 0:
			if np.abs(np.linalg.norm(x_obs[-1, :]) - np.linalg.norm(x_obs[-2, :])) < eps:
				break
		if step != steps - 1:
			plt.clf()

	# 效果展示
	x, y = xs, xs
	mesh_x, mesh_y = np.meshgrid(x, y)
	value = np.sin(mesh_x) + np.cos(mesh_y) ** 2 + 0.01 * mesh_x ** 2 + 0.01 * mesh_y ** 2
	plt.figure()
	plt.contourf(mesh_x, mesh_y, value)
	plt.colorbar()
	plt.scatter(x_obs[-1, 0], x_obs[-1, 1], marker = '*', color = 'r', s = 80)






