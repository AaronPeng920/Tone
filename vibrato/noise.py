import numpy as np
import matplotlib.pyplot as plt
from .fluctuation import sine

# 高斯噪声
def gause_noise(mean=0.0, square=1.0, size=10, min=np.finfo(np.float32).min, max=np.finfo(np.float32).max):
    """参数
        mean: 均值
        square: 方差
        size: 生成数据的尺寸
        min: 裁剪的最小值
        max: 裁剪的最大值
        -------------------------------
        return: 指定 size 的并裁剪后的高斯噪声随机数
    """
    res = np.random.normal(loc=mean, scale=square, size=size)
    res_cliped = np.clip(res, min, max)
    return res_cliped




if __name__ == '__main__':
    _, fluct = sine(1, 1, 0, 93, 0, 10 * 2 * np.pi)
    noise = gause_noise(0, 0.25, 93, -0.5, 0.5)
    
    plt.plot(fluct)
    plt.plot(noise)

    fluct = fluct + noise
    plt.plot(fluct)
    plt.show()

