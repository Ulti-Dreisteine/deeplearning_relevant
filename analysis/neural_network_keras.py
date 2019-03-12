# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 14:32:52 2018

@author: luolei

keras神经网络训练和预测
"""
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.datasets import mnist


def gen_cluster(samples_len, center_loc = None):
	"""
	生成聚团样本
	:param center_loc:
	:param samples_len: 样本长度
	:return:
	"""
	if center_loc is None:
		center_loc = [0, 0]

	samples = pd.DataFrame(np.random.normal(center_loc, [1, 1], (samples_len, 2)))
	samples['index'] = samples.index
	return samples


def gen_circle(samples_len, center_loc = None):
	"""
	生成圈状样本
	:param samples_len:
	:param center_loc:
	:return:
	"""
	if center_loc is None:
		center_loc = [0, 0]

	x = 2.0 * np.pi * np.random.random((samples_len, 1))
	samples = np.hstack(
		(
			5 * np.sin(x),
			5 * np.cos(x)
		),
	) + center_loc
	samples = samples + 1 * np.random.random((samples_len, 2))
	samples = pd.DataFrame(samples)
	samples['index'] = samples.index
	return samples


def gen_samples(samples_len, show_plot = False):
	"""
	生成样本
	:param show_plot:
	:param samples_len:
	:return:
	"""
	cluster = gen_cluster(round(samples_len / 2))
	cluster['label'] = cluster.loc[:, 'index'].apply(lambda x: 0)
	circle = gen_circle(round(samples_len / 2))
	circle['label'] = circle.loc[:, 'index'].apply(lambda x: 1)
	samples = pd.concat([cluster, circle], axis = 0)

	if show_plot:
		plt.figure('distribution of samples')
		plt.scatter(samples[0], samples[1])
		plt.grid(True)
		plt.pause(1)

	return samples


if __name__ == '__main__':
	# 生成样本
	raw_data = gen_samples(samples_len = 1000, show_plot = False)
	test_data = gen_samples(samples_len = 100)

	# 输入训练数据 keras接收numpy数组类型的数据
	samples = np.array(raw_data[[0, 1]])
	labels = np.array(raw_data['label'])
	test_samples = np.array(test_data[[0, 1]])
	test_labels = np.array(test_data['label'])

	# 整理数据格式
	samples = samples.reshape(samples.shape[0], -1)
	labels = labels.reshape(labels.shape[0], -1)
	test_samples = test_samples.reshape(test_samples.shape[0], -1)
	test_labels = test_labels.reshape(test_labels.shape[0], -1)

	# 使用序贯模型
	model = Sequential()
	model.add(Dense(8, input_dim = 2, activation = 'tanh'))
	model.add(Dense(8, input_dim = 2, activation = 'tanh'))
	model.add(Dense(1))
	model.compile(optimizer = 'sgd', loss = 'mse')

	train_losses = []
	test_losses = []
	for step in range(10000):
		train_loss = model.train_on_batch(samples, labels)
		test_loss = model.test_on_batch(test_samples, test_labels)
		train_losses.append(train_loss)
		test_losses.append(test_loss)
		print('\ntrain loss: %.4f, test loss: %.4f' % (train_loss, test_loss))

	plt.plot(train_losses)
	plt.plot(test_losses)