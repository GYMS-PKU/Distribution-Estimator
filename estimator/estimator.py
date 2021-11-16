# Copyright (c) 2021 Dai HBG

"""
该代码定义统一对外接口

开发日志
2021.11.16
-- start
"""


import gan

class Estimator:
    def __init__(self, input_dim=5, output_dim=10, estimator='GAN', params=None, device='cpu'):
        """
        :param input_dim: 原始分布采样维度
        :param output_dim: 目标分布维度
        :param estimator: 估计器
        :param params: 估计器参数，以dict形式传入
        :param device:
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.estimator_name = estimator
        if estimator == 'GAN':
            if params is None:
                params = {'input_dim': self.input_dim, 'output_dim': self.output_dim}
            self.estimator = MyGAN(input_dim=params['input_dim'], output_dim=params['output_dim'])

        self.device = device

    def fit(self, x_true, epochs=1000, batch_size=10000):  # 拟合估计器
        if self.estimator_name == 'GAN':
            for epoch in range(epochs):

