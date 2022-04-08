# Assignment 1 for computer vision

### 文件说明

- chxtorch.py: 基于 numpy，仿照 *pytorch* 实现了简单的深度学习框架，定义了一些神经网络模块和损失函数，包括sigmoid、ReLU、Tanh、全连接层、卷积层、MSE、softmax等
- nn.py: 定义了训练所用的网络模型
- mnist_loader.py：定义了 mnist 数据集的加载方法
- train.py：训练文件
- paramSelect.py：参数查找文件
- test.py：测试文件
- visualize.ipynb：可视化网络参数文件
- output1.txt, output2.txt：两次参数查找的结果
- params1, params2：两次参数查找的最佳模型

### 参数设置

以下训练、测试、参数查找均需要设置以下参数，示例如下

```python
# set the learning rate
lr = 1e-3
# set the penalty coefficient
penalty = 1e-6
# activation should be chosen from 'Sigmoid' and 'ReLU'
activation = 'Sigmoid'
# hidden_layers should be a list
hidden_layers = [128, 64]
# silent means not showing the training information
silent = False
# need_plot means show the plot of the training process
need_plot = True
# if need plot, set your path to save the figure
acc_path = 'acc.png'
loss_path = 'loss.png'
```

具体的设置细节可以参照对应的文件注释

### 训练

在 train.py 中设置参数，并运行文件

### 测试

在 test.py 中设置参数，并运行文件

### 参数查找

在 paramSelect.py 中设置参数，并运行文件

### 可视化

在 visualize.ipynb 中设置参数文件路径，并运行文件
