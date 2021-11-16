# Distribution-Estimator

Distribution-Estimator是一个研究多种分布估计器性能的项目。主要结构如下：

- **estimator**模块定义估计器，所有关于估计器的代码都必须定义在该模块下，且统一输入输出接口为**numpy.array**格式的**一维**向量（如果后续加入二维分布游可能更改），由**estimator.estimator**类统一对外接口；GAN的损失函数也定义在该模块下；
- **criterion**模块定义评价准则，所有需要尝试的评价准则都定义在该模块下，统一输入接口为**numpy.array**格式；
- **dataconstructor**模块定义数据生成，所有从已知分布或者未知分布中生成样本的方法都定义在该模块下；





### 开发日志

##### 2021.11.16

-- GAN Demo（by dyh）