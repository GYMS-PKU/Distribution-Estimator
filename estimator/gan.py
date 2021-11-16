# Copyright (c) 2021 Dai HBG

"""
该代码定义GAN

开发日志
2021.11.16
-- start
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim


class Generator(nn.Module):  # 生成器
    def __init__(self, input_dim=5, output_dim=10, dropout=0.2, alpha=0.2):
        """
        :param input_dim: 原始分布采样维度
        :param output_dim: 目标分布维度
        :param dropout:
        :param alpha:
        """
        super(Generator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.dropout = dropout
        self.alpha = alpha

        self.Dense1 = nn.Linear(input_dim, input_dim * 2)
        if input_dim >= 2:
            self.Dense2 = nn.Linear(input_dim * 2, input_dim // 2)
            self.Dense3 = nn.Linear(input_dim // 2, output_dim)
        else:
            self.Dense2 = nn.Linear(input_dim * 2, input_dim)
        self.Dense3 = nn.Linear(input_dim, output_dim)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, x):
        x = self.leakyrelu(self.Dense1(x))
        x = self.leakyrelu(self.Dense2(x))
        x = self.Dense3(x)
        return x


class Discriminator(nn.Module):  # 判别器
    def __init__(self, input_dim=10, output_dim=1, dropout=0.2, alpha=0.2):
        """
        :param input_dim: 目标分布维度
        :param output_dim: 二分类
        :param dropout:
        :param alpha:
        """
        super(Discriminator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.dropout = dropout
        self.alpha = alpha

        self.Dense1 = nn.Linear(input_dim, input_dim * 2)
        if input_dim >= 2:
            self.Dense2 = nn.Linear(input_dim * 2, input_dim // 2)
            self.Dense3 = nn.Linear(input_dim // 2, output_dim)
        else:
            self.Dense2 = nn.Linear(input_dim * 2, input_dim)
        self.Dense3 = nn.Linear(input_dim, output_dim)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.sigmoid = nn.LogSoftmax()

    def forward(self, x):
        x = self.leakyrelu(self.Dense1(x))
        x = self.leakyrelu(self.Dense2(x))
        x = self.leakyrelu(self.Dense3(x))
        return F.log_softmax(x, dim=1)


class GAN(nn.Module):
    def __init__(self, input_dim=5, output_dim=10):
        """
        :param input_dim: 原始分布采样维度
        :param output_dim: 目标分布维度
        """
        super(GAN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.generator = Generator(input_dim=self.input_dim, output_dim=self.output_dim)
        self.discriminator = Discriminator(input_dim=self.output_dim)

    def generate(self, x):
        return self.generator(x)

    def discriminate(self, x):
        return self.discriminator(x)
