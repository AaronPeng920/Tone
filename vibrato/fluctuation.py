import librosa
import numpy as np
import matplotlib.pyplot as plt
import os

# 正弦波采样
def sine(A, w, b, n, start=0, end=-1):
    """参数
        A, w, b: 构成正弦波 y = A * sin(wx+b)
        n: 采样 n 个点
        start: 开始采样的位置
        end: 结束采样的位置
        ----------------------------
        return: (采样点x值, 采样点y值), shape:(2,n)
    """
    # 确定采样间隔
    if end == -1 or end <= start:
        step = 1
    else:
        step = (end - start) / (n - 1)

    x = np.array([start + i * step for i in range(n)])     # 采样点的横坐标值
    y = A * np.sin(w * x + b)
    return x, y

# 生成不规律的频率在 [low, high] 均匀分布的正弦波
def uniform_sine(A, low, high, n):
    """参数
        A: 不规律正弦波的系数
        low: 频率下限
        high: 频率上限
        n: 采样点个数
        ---------------------------------
        return: (采样点x值, 采样点y值), shape:(2,n)
    """
    t = np.linspace(0, 2*np.pi, n)
    f = np.random.uniform(low, high, len(t))
    x = A * np.sin(2*np.pi*f*t)

    return t, x

# 生成不规律的频率在均值为 mean, 方差为 square 的正态分布的正弦波
def gause_sine(A, mean, square, n):
    """参数
        A: 不规律的正弦波的系数
        mean: 频率均值
        square: 频率方差
        n: 采样点个数
        --------------------------------
        return: (采样点x值, 采样点y值), shape:(2,n)
    """
    t = np.linspace(0, 2*np.pi, n)
    f = np.random.normal(mean, square, len(t))
    # 使用一个长度为 32 的均值滤波器对频率序列进行平滑处理, 从而让频率的变化更加缓慢
    f_smooth = np.convolve(f, np.ones(32)/32, mode='same')
    # 生成正弦波
    x = A * np.sin(2*np.pi*f_smooth*t)
    
    return t, x

# 生成频率逐渐增大的正弦波
def increase_f_sine(p, n):
    """参数
        p: 波动程度 1-10
        n: 采样点的个数
        -----------------------
        return: (采样点x值, 采样点y值), shape:(2,n)
    """
    # 生成含有n个采样点的信号
    x = np.linspace(0, 10, n)

    # 生成周期变化的信号
    periods = 5 * np.exp(-(p/10) * x)  # 周期变化的数组
    frequencies = 2 * np.pi / periods  # 频率根据周期计算

    y = np.sin(frequencies * x)
    return x, y

# 生成频率逐渐降低的正弦波
def decrease_f_sine(p, n):
    """参数
        p: 波动程度 1-10
        n: 采样点的个数
        -------------------------
        return: (采样点x值, 采样点y值), shape:(2,n)
    """
    # 生成含有n个采样点的信号
    x = np.linspace(0, 10, n)

    # 生成频率逐渐降低的正弦波信号
    frequencies = p * np.exp(-0.1 * x)  # 频率逐渐降低的数组
    y = np.sin(2 * np.pi * frequencies * x)

    return x, y

if __name__ == '__main__':
    _, fluct = decrease_f_sine(1, 100)
    plt.plot(fluct)
    plt.show()

