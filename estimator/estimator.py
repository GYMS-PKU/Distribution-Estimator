# Copyright (c) 2021 Dai HBG

"""
该代码定义统一对外接口

开发日志
2021.11.16
-- start
"""


from .gan import *
import numpy as np


class Estimator:
    def __init__(self, input_dim=5, output_dim=10, estimator='GAN', device='cpu'):
        """
        :param input_dim: 原始分布采样维度
        :param output_dim: 目标分布维度
        :param estimator: 估计器
        :param device:
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.estimator_name = estimator
        self.device = device
        if estimator == 'GAN':
            self.estimator = MyGAN(input_dim=self.input_dim, output_dim=self.output_dim, device=self.device)

    def fit(self, x_true, epochs=1000, batch_size=10000):  # 拟合估计器
        self.estimator.fit(x_true, epochs=epochs, batch_size=batch_size)

    def generate(self, n=1):
        return self.estimator.generate(n=n)

