import numpy as np
import matplotlib.pyplot as plt

# 生成含有m个采样点的信号
m = 100
x = np.linspace(0, 10, m)

# 生成频率逐渐降低的正弦波信号
frequencies = 2 * np.exp(-0.1 * x)  # 频率逐渐降低的数组
y = np.sin(2 * np.pi * frequencies * x)

# 绘制图形
fig, ax = plt.subplots()
ax.plot(x, y)
plt.show()
