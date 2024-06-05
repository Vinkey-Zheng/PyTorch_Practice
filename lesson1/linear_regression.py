# -*- coding:utf-8 -*-

import torch
import matplotlib.pyplot as plt
torch.manual_seed(10)

lr = 0.05  # 学习率

# 创建训练数据
x = torch.rand(20, 1) * 10  # x data (tensor), shape=(20, 1)
# torch.randn(20, 1) 用于添加噪声
y = 2*x + (5 + torch.randn(20, 1))  # y data (tensor), shape=(20, 1)

# 构建线性回归参数
w = torch.randn((1), requires_grad=True) # 设置梯度求解为 true
b = torch.zeros((1), requires_grad=True) # 设置梯度求解为 true

# 迭代训练 1000 次
for iteration in range(1000):

    # 前向传播，计算预测值, y = wx + b
    wx = torch.mul(w, x)
    y_pred = torch.add(wx, b)

    # 计算 MSE loss
    loss = (0.5 * (y - y_pred) ** 2).mean()

    # 反向传播
    loss.backward()

    # 更新参数
    b.data.sub_(lr * b.grad)
    w.data.sub_(lr * w.grad)

    # 每次更新参数之后，都要清零张量的梯度
    w.grad.zero_()
    b.grad.zero_()

    # 绘图，每隔 20 次重新绘制直线
    if iteration % 20 == 0:

        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), y_pred.data.numpy(), 'r-', lw=5)
        plt.text(2, 20, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color':  'red'})
        plt.xlim(1.5, 10)
        plt.ylim(8, 28)
        plt.title("Iteration: {}\nw: {} b: {}".format(iteration, w.data.numpy(), b.data.numpy()))
        plt.pause(0.5)


        # plt.plot
        # plt.plot 是Matplotlib库中的一个函数，用于绘制曲线或直线。
        # x.data.numpy() 是训练数据的输入特征x的值，通过.data.numpy()方法将其转换为NumPy数组类型。
        # y_pred.data.numpy() 是模型对训练数据的预测值y_pred，通过.data.numpy()方法将其转换为NumPy数组类型。
        # 'r-' 是绘制直线的样式参数。'r'表示红色（red），'-'表示实线。
        # lw=5 设置直线的线宽为5个像素（可以根据需要进行调整）。

        # plt.text
        # plt.text 是Matplotlib库中的一个函数，用于在图中添加文本。
        # (2, 20) 是文本的坐标位置，即文本显示的位置在图中的x坐标为2，y坐标为20。
        # 'Loss=%.4f' % loss.data.numpy() 是要显示的文本内容。%.4f 是一个格式化字符串，
        # 用于将loss.data.numpy()的值格式化为四位小数的字符串。
        # loss.data.numpy()是当前迭代的损失值，通过.numpy()方法将其转换为NumPy数组类型。
        # fontdict={'size': 20, 'color':  'red'} 是一个字典，用于设置文本的字体样式。
        # size指定文本的字体大小为20，color指定文本的颜色为红色。

        # 如果 MSE 小于 1，则停止训练
        if loss.data.numpy() < 1:
            break