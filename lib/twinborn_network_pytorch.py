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

sys.path.append('../')

from mods.gen_train_samples_for_nn import gen_train_samples_and_targets


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
	wind_vels = [1.0, 1.0]
	# 载入数据
	file_names = ['interp_arr_step_0', 'interp_arr_step_1', 'interp_arr_step_2']
	kriging_interp_results = []
	for file in file_names:
		kriging_interp_results.append(np.load('../tmp/' + file + '.npy'))
	kriging_interp_results = np.array(kriging_interp_results)

	# 获取神经网络训练样本
	X_train, y_train = gen_train_samples_and_targets(kriging_interp_results, wind_vels)



