import numpy as np
import matplotlib.pyplot as plt

# 生成随机数据作为曲线
x = np.linspace(0, 10, 100)
y = np.sin(x) + np.random.normal(0, 0.1, 100)

# 定义窗口大小
window_size = 5

# 计算加权平均数
weights = np.repeat(1.0, window_size) / window_size
smoothed_y = np.convolve(y, weights, 'valid')

# 绘制原始曲线和平滑后的曲线
plt.plot(x, y, 'b', label='Original Curve')
plt.plot(x[window_size-1:], smoothed_y, 'r', label='Smoothed Curve')
plt.legend()
plt.show()
