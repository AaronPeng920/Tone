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


# 三角波
def triangle(k, T, n, start=0, end=-1):
    """参数
        k: 三角波第一段线的斜率
        T: 三角波的周期
        n: 采样点的个数
        start: 开始采样的位置
        end: 结束采样的位置
        -----------------------
        return: (采样点x值, 采样点y值), shape:(2,n)
    """
    # 三角波的函数
    def f(k, T, x):
        """参数
            k: 三角波第一段线的斜率
            T: 三角波的周期
            x: 横坐标值
        """
        half_T = T / 2.0     
        half_T_i = np.floor(x / half_T)   # 第几个半个周期
        # 偶数个半个周期表明函数是: y = k * x
        if half_T_i % 2 == 0:
            y = k * (x - half_T_i * half_T)
        # 奇数个半个周期表明函数是: y = -k * x + k * T/2
        else:
            y = -1 * k * (x - half_T_i * half_T) + k * half_T
        return y
    
    # 确定采样间隔
    if end == -1 or end <= start:
        step = 1
    else:
        step = (end - start) / (n - 1)

    x = np.array([start + i * step for i in range(n)])     # 采样点的横坐标值
    y = np.array([f(k, T, t) for t in x])

    return x, y

# 方波




# 锯齿形



# 反锯齿形


# 多项式
def polynomial(h_power, coefficients, n, start=0, end=-1):
    """参数
        h_power: 最高次幂
        coefficients: 系数, list, shape:(h_power+1, )
        n: 采样的个数
        start: 开始采样的位置
        end: 结束采样的位置
        ---------------------------------
        return: (采样点x值, 采样点y值), shape:(2,n)
    """
    # 多项式函数定义
    def f(h_power, coefficients, x):
        """参数
            h_power: 最高次幂, int
            coefficients: 系数, 1-D np.array 或者 list, shape:(h_power+1, )
            x: 横坐标的值
            ----------------------------------------------
            return: 多项式函数值
        """
        assert len(coefficients) == h_power + 1

        y = 0
        for i in range(h_power):
            y += coefficients[-1 * (i+1)] * x ** i
        return y

    # 确定采样间隔
    if end == -1 or end <= start:
        step = 1
    else:
        step = (end - start) / (n - 1)

    x = np.array([start + i * step for i in range(n)])     # 采样点的横坐标值
    y = np.array([f(h_power, coefficients, t) for t in x])

    return x, y

# 波动函数父类
class Fluct:
    def __init__(self, fluctuation):
        """参数
            fluctuation: 波动函数, python 函数, 定义为 f(x)
        """
        self.fluctuation = fluctuation

    # 计算波动值
    def forward(self, x):
        return self.fluctuation(x)

    # 采样
    def sample(self, n, start=0, end=-1):
        """参数
            n: 采样点的个数
            start: 开始采样的位置
            end: 结束采样的位置
            ----------------------------
            return: (采样点x轴值, 采样点y轴值), shape:(2, n)
        """
        # 确定采样间隔
        if end == -1 or end <= start:
            step = 1
        else:
            step = (end - start) / (n - 1)

        x_samples = np.array([start + i * step for i in range(n)])     # 采样点的横坐标值
        y_samples = np.array([self.forward(t) for t in x])

        return x_samples, y_samples

    # 展示波动函数的形状
    def plot(self, save=False, save_filename=None):
        """参数
            save: 是否进行保存, 否则直接显示
            save_filename: 保存的路径
        """
        x = np.arange(-50, 50, 0.1) 
        y = np.array([self.forward(t) for t in x])

        plt.plot(x, y)
        # 进行保存
        if save:
            if os.path.exists(save_filename):
                plt.savefig(save_filename, dpi=600)
            else:
                raise FileNotFoundError('not found file {} to save the fig'.format(save_filename))
        # 进行显示
        else:
            plt.show()
    


if __name__ == '__main__':
    def f(x):
        return 0.2*np.sin(x) + 0.2*np.sin(2*x) + 0.2*np.sin(3*x) + 0.2*np.sin(4*x) + 0.2*np.sin(5*x)

    fluct = Fluct(f)
    fluct.plot()
