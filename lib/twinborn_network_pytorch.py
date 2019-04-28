# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 14:32:52 2018

@author: luolei

pytorch孪生网络训练模型
"""
import numpy as np
import sys
import torch
from torch import nn
from torch.autograd import Variable
import copy
import pandas as pd

sys.path.append('../')

from mods.gen_train_samples_for_nn import gen_train_samples_and_targets


def criterion(y_true, y_pred):
	"""
	损失函数
	:param y_true: np.ndarray, 真实值, shape = (-1,)
	:param y_pred: np.ndarray, 预测值, shape = (-1,)
	:return: loss, 损失函数, float
	"""
	l1, mse = nn.L1Loss(), nn.MSELoss()
	loss = torch.add(l1(y_true, y_pred), mse(y_true, y_pred))
	return loss

class NN(nn.Module):
	def __init__(self, input_size, hidden_size):
		super(NN, self).__init__()
		self.linear_0 = nn.Linear(input_size, hidden_size)
		self.relu_0 = nn.ReLU()
		self.linear_1 = nn.Linear(hidden_size, hidden_size)
		self.relu_1 = nn.ReLU()
		self.linear_2 = nn.Linear(hidden_size, 1)
		self.relu_2 = nn.ReLU()

	def forward(self, x):
		x = self.linear_0(x)
		x = self.relu_0(x)
		x = self.linear_1(x)
		x = self.relu_1(x)
		x = self.linear_2(x)
		x = self.relu_2(x)
		return x


def partial_derivative_vector(sub_nn_model, sample_arr, eps = 1e-4):
	"""
	计算偏导数
	:param sub_nn_model:
	:param sample_arr:
	:param eps:
	:return:
	"""
	partial_derivatives = []
	for i in range(len(sample_arr)):
		sample_arr_copy = copy.deepcopy(sample_arr)
		original_value = sub_nn_model(sample_arr_copy.reshape(1, -1))[0, 0]

		if sample_arr_copy[i] == 0:
			sample_arr_copy[i] += eps
			new_value = sub_nn_model.predict_on_batch(sample_arr_copy.reshape(1, -1))[0, 0]
			partial_derivatives.append((new_value - original_value) / eps)
		else:
			sample_arr_copy[i] += eps * sample_arr_copy[i]
			new_value = sub_nn_model.predict_on_batch(sample_arr_copy.reshape(1, -1))[0, 0]
			partial_derivatives.append((new_value - original_value) / (eps * sample_arr_copy[i]))

	return np.array(partial_derivatives)


def cal_partial_derivative_vectors(sample_arrs, sub_nn_model):
	"""
	计算所有样本的偏导向量
	:param sample_arrs:
	:param sub_nn_model:
	:return:
	"""
	sample_arrs = pd.DataFrame(sample_arrs)
	sample_arrs['partial_derivative_vector'] = sample_arrs.apply(lambda x: partial_derivative_vector(sub_nn_model, np.array(x)), axis = 1)
	return np.array(list(sample_arrs['partial_derivative_vector'])).reshape(sample_arrs.shape[0], sample_arrs.shape[1] - 1)


def cal_contributions(X_train, sub_nn_model, kriging_interp_results, show_plot = False):
	"""
	计算输入对输出的贡献情况
	:param show_plot:
	:param X_train:
	:param sub_nn_model:
	:param kriging_interp_results:
	:return:
	"""
	kriging_interp_results = copy.deepcopy(kriging_interp_results)
	total_contributions = kriging_interp_results[2, 1:-1, 1:-1].flatten()  # 预测第三个时刻的值

	sample_arrs = X_train[:, 10:]
	partial_derivative_vectors = cal_partial_derivative_vectors(sample_arrs, sub_nn_model)  # t时刻偏微分

	external_contributions = sub_nn_model.predict_on_batch(sample_arrs).flatten()

	# 利用四个角上的均值消除背景误差, 这个误差由sub_nn_model训练过程中subtract操作导致
	external_contributions = external_contributions - np.mean(
		[
			external_contributions[0],
			external_contributions[0 + kriging_interp_results[1].shape[1] - 1],
			external_contributions[-kriging_interp_results[1].shape[1]],
			external_contributions[-1]
		]
	)

	directional_external_contributions = np.multiply(partial_derivative_vectors, sample_arrs)

	internal_contributions = total_contributions - external_contributions

	internal_contributions_ratio = np.sum(internal_contributions) / np.sum(total_contributions)
	external_contributions_ratio = np.sum(external_contributions) / np.sum(total_contributions)

	pollutant_ratios = {'internal': internal_contributions_ratio, 'external': external_contributions_ratio}

	if show_plot:
		edge_len = [kriging_interp_results.shape[1] - 2, kriging_interp_results.shape[2] - 2]
		mesh_x, mesh_y = np.meshgrid(np.arange(edge_len[0]), np.arange(edge_len[1]))
		plt.figure(figsize = [12, 3.5])
		plt.subplot(1, 3, 1)
		plt.title('total')
		plt.contour(mesh_x, mesh_y, flipping_arr(total_contributions.reshape(edge_len[0], edge_len[1])), vmin = -200.0, vmax = 200.0, cmap = 'seismic', levels = 15)
		plt.xlabel('longitude')
		plt.ylabel('latitude')
		plt.grid(True)
		plt.subplot(1, 3, 2)
		plt.title('external')
		plt.contour(mesh_x, mesh_y, flipping_arr(external_contributions.reshape(edge_len[0], edge_len[1])), vmin = -200.0, vmax = 200.0, cmap = 'seismic', levels = 15)
		plt.xlabel('longitude')
		plt.ylabel('latitude')
		plt.grid(True)
		plt.subplot(1, 3, 3)
		plt.title('internal')
		plt.contour(mesh_x, mesh_y, flipping_arr(internal_contributions.reshape(edge_len[0], edge_len[1])), vmin = -200.0, vmax = 200.0, cmap = 'seismic', levels = 15)
		plt.xlabel('longitude')
		plt.ylabel('latitude')
		plt.grid(True)
		plt.tight_layout()

		plt.figure(figsize = [10, 4])
		sns.heatmap(directional_external_contributions, vmin = -100, vmax = 100, cmap = 'seismic')
		plt.xticks(np.arange(0.5, 10.5, 1), ['NW', 'N', 'NE', 'W', 'E', 'SW', 'S', 'SE', 'WE-Wind', 'NS-Wind'])
		plt.tight_layout()

	return total_contributions, internal_contributions, external_contributions, directional_external_contributions, pollutant_ratios



if __name__ == '__main__':
	# 计算参数
	wind_vels = [1.0, 1.0]
	lr = 0.001
	epochs = 20000
	show_plot = True

	# 载入数据
	file_names = ['interp_arr_step_0', 'interp_arr_step_1', 'interp_arr_step_2']
	kriging_interp_results = []
	for file in file_names:
		kriging_interp_results.append(np.load('../tmp/' + file + '.npy'))
	kriging_interp_results = np.array(kriging_interp_results)

	# 获取神经网络训练样本
	X_train, y_train = gen_train_samples_and_targets(kriging_interp_results, wind_vels)

	# 构建网络模型
	X_train_0, X_train_1 = X_train[:, :10], X_train[:, 10:]
	input_size = X_train_0.shape[1]
	hidden_size = 20
	sub_nn_model = NN(input_size, hidden_size)
	optimizer = torch.optim.Adam(sub_nn_model.parameters(), lr = lr)

	# 准备样本
	X_train_0_model, X_train_1_model, y_train_model = torch.from_numpy(X_train_0.astype(np.float32)), torch.from_numpy(X_train_1.astype(np.float32)), torch.from_numpy(y_train.astype(np.float32))
	var_x_0, var_x_1, var_y = Variable(X_train_0_model), Variable(X_train_1_model), Variable(y_train_model)

	# 进行训练
	loss_record = []
	for epoch in range(epochs):
		# 前向传播
		y_train_p = sub_nn_model(var_x_1) - sub_nn_model(var_x_0)
		y_train_t = var_y
		train_loss = criterion(y_train_p[:, 0], y_train_t[:, 0])

		# 反向传播
		optimizer.zero_grad()
		train_loss.backward()
		optimizer.step()

		if (epoch + 1) % 100 == 0:  # 每 100 次输出结果
			train_loss = train_loss.data.cpu().numpy()
			loss_record.append([epoch + 1, float(train_loss)])
			print('epoch: %s, train_loss: %.10f' % (epoch + 1, train_loss))

		# 计算外部贡献值
		total_contributions, internal_contributions, external_contributions, directional_external_contributions, ratios = cal_contributions(
			X_train, sub_nn_model, kriging_interp_results, show_plot = show_plot
		)





