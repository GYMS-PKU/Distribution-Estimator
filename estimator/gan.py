# Copyright (c) 2021 Dai HBG

"""
该代码定义GAN

开发日志
2021.11.16
-- start
"""


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


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
    def __init__(self, input_dim=10, output_dim=2, dropout=0.2, alpha=0.2):
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


class MyGAN:
    def __init__(self, input_dim=5, output_dim=10, device='cpu'):
        """
        :param input_dim: 原始分布采样维度
        :param output_dim: 目标分布维度
        :param device:
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device
        self.generator = Generator(input_dim=self.input_dim, output_dim=self.output_dim).to(self.device)
        self.discriminator = Discriminator(input_dim=self.output_dim).to(self.device)

        self.g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=1e-3, weight_decay=1e-3)
        self.d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=1e-3, weight_decay=1e-3)

    def generate(self, n=1):
        with torch.no_grad():
            return self.generator(torch.randn(n, self.input_dim).to(self.device)).cpu().numpy()

    def fit(self, x_true, epochs=1000, batch_size=10000):
        lst = [i for i in range(len(x_true))]
        x_fate = np.random.randn(len(x_true), self.input_dim)
        for epoch in range(epochs):
            batch = np.random.choice(lst, batch_size)
            x_t = torch.Tensor(x_true[batch]).to(self.device)
            y_t = torch.ones(batch_size).long().to(self.device)
            x_f = torch.Tensor(x_fate[batch]).to(self.device)  # 从标准正态采样
            y_f = torch.zeros(batch_size).long().to(self.device)

            # 训练生成器
            for i in range(10):
                self.g_optimizer.zero_grad()
                z = self.generator(x_f)  # 假样本
                g_loss = F.nll_loss(self.discriminator(z.detach()), y_t.long())
                g_loss.backward()
                self.g_optimizer.step()

            # 训练分类器
            self.d_optimizer.zero_grad()
            r_loss = F.nll_loss(self.discriminator(x_t), y_t.long())
            r_loss.backward()
            f_loss = F.nll_loss(self.discriminator(z.detach()), y_f.long())
            f_loss.backward()
            self.d_optimizer.step()

            if (epoch+1) % (epochs//10) == 0:
                print('epoch {} g_loss: {:.4f}, d_loss: {:.4f}'.format(epoch+1, g_loss, (r_loss+f_loss)/2))
