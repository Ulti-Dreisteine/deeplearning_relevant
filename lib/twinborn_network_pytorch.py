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
	# loss = torch.add(l1(y_true, y_pred), mse(y_true, y_pred))
	loss = l1(y_true, y_pred)
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


if __name__ == '__main__':
	# 计算参数
	wind_vels = [2.0, 1.0]
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

	# 参数初始化
	sub_nn_model.linear_0.weight.data = torch.rand(hidden_size, input_size)  # attention: 注意shape是转置关系
	sub_nn_model.linear_0.bias.data = torch.rand(hidden_size)
	sub_nn_model.linear_1.weight.data = torch.rand(hidden_size, hidden_size)
	sub_nn_model.linear_1.bias.data = torch.rand(hidden_size)
	sub_nn_model.linear_2.weight.data = torch.rand(1, hidden_size)
	sub_nn_model.linear_2.bias.data = torch.rand(1)

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




