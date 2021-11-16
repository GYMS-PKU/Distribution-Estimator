## estimator

该模块下定义估计器。**estimator**定义了总体的接口，外部通过调用estimator方法来使用不同的估计器。

#### 添加新的估计器

为了添加新的估计器，需要完成以下工作：

- 定义一个估计器类，可参考**gan.py**，其中必须定义
  - **fit(np.arrray: x_true, int: epochs, int: batch_size)**方法，用于拟合分布，该接口将被estimator类调用；
  - **generate(np.array: x)**方法，用于生成分布；（该方法可能更改）
- 修改estimator类，将新定义的估计器添加到可选的方法中。

