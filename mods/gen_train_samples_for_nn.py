# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 14:32:52 2018

@author: luolei

构造神经网络训练样本
"""
import numpy as np
import sys
import copy
import logging

sys.path.append('../')

_logger = logging.getLogger(__name__)


def get_neighbors(array, loc):
	"""
	获取数组中位置为loc的点的八个方向的邻点值序列
	:param array: 待取值数组
	:param loc: 点的位置[row, column]
	:return: neighbors: array, 该中心点周围八个方向邻点值
	:return: center: float, 该中心点的值
	"""
	neighbors = array[(loc[0] - 1): (loc[0] + 2), (loc[1] - 1): (loc[1] + 2)].flatten()
	center = neighbors[4]
	neighbors = np.delete(neighbors.reshape(1, -1), 4, 1).flatten()
	return neighbors, center


def gen_samples_and_centers(interp_arr, wind_vels):
	"""
	在一个array中获取中心点和邻点
	:param interp_arr: array, 某时刻的插值结果矩阵
	:param wind_vels: [vel_x, vel_y], 风速
	:return: samples: array, 该时刻所有目标中心点周围插值结果构成的样本
	:return: centers: array, 该时刻所有目标中心点的测量值
	"""
	# TODO: 该步构造样本的方式可以优化，但对计算性能影响不大
	samples = []
	centers = []
	for row in range(1, interp_arr.shape[0] - 1):
		for column in range(1, interp_arr.shape[1] - 1):
			neighbors, center = get_neighbors(interp_arr, [row, column])
			samples.append(list(neighbors) + list(wind_vels))
			centers.append(center)
	samples = np.array(samples)
	centers = np.array(centers)
	return samples, centers


# @time_cost
def gen_train_samples_and_targets(kriging_interp_results, wind_vels):
	"""
	构造神经网络的训练样本
	:param wind_vels: [vel_x, vel_y], 风速
	:param kriging_interp_results: array, kriging插值结果, shape = (3, :, :)
	:return: X_train: array, 训练样本
	:return: y_train: array, 训练样本
	"""
	kriging_interp_results = copy.deepcopy(kriging_interp_results)

	if kriging_interp_results.shape[0] != 3:
		_logger.error('ERROR mod.gen_train_samples_for_nn: the shape of kriging_interp_results should be (3, n_0, n_1)')
		raise ValueError('the shape of kriging_interp_results is not correct')

	else:
		_logger.info('mod.gen_train_samples_for_nn: get train samples from the interp results')
		interp_arrs = []
		for i in range(3):  # 会用到三步时间以内的信息
			interp_arrs.append(kriging_interp_results[i, :, :])

		# 构造样本
		samples_list = []
		centers_list = []
		for i in range(3):
			samples, centers = gen_samples_and_centers(interp_arrs[i], wind_vels)
			samples_list.append(samples)
			centers_list.append(centers)

		y_train = (centers_list[-1] - centers_list[-2]).reshape(-1, 1)
		X_train = np.hstack((samples_list[0], samples_list[1]))  # np.array([samples_list[0], samples_list[1]]), shape = (-1, 20)

		return X_train, y_train


