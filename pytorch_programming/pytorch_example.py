# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 14:32:52 2018

@author: luolei

pytorch 计算示例
"""
import torch
from torch.optim import Adam
import matplotlib.pyplot as plt
from torch.nn import Linear, ReLU, Sequential


def g(x, y):
	"""
	定义函数g
	:param x: np.ndarray, 向量, [x0, x1, x2]
	:param y: float, 标量
	:return: 标量
	"""
	return (1.0 + x[1] + x[2] ** 2) * y - y ** 2 - x[1] * x[2] ** 2


if __name__ == '__main__':
	# 搭建网络
	net = Sequential(  # 序贯模型
		Linear(3, 8),  # 输入层维数为3
		ReLU(),
		Linear(8, 8),
		ReLU(),
		Linear(8, 1)   # 输出层维数为1
	)

	# 指定优化器
	optimizer = Adam(net.parameters())		# key point: 指定对网络参数进行优化

	# 进行训练
	loss_record = []
	steps = 10000
	for step in range(steps):
		optimizer.zero_grad()				# key point: 清除上一步梯度信息
		x = torch.randn(1000, 3)
		y = net(x)
		outputs = g(x, y)
		loss = - torch.sum(outputs)
		loss.backward()						# key point: 反向传播, 计算loss对各层参数的梯度信息
		optimizer.step()					# key point: 参数更新，结合梯度信息和模型当前参数信息计算下一步模型参数

		loss_record.append(loss)

		if step % 10 == 0:
			print('第{}次迭代损失 = {}'.format(step, loss))

	plt.plot(loss_record)





