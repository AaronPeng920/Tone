import librosa
import matplotlib.pyplot as plt
import matplotlib
from .utils import *
import numpy as np
import os
matplotlib.use('agg')


"""
计算音频的均方根能量, 可以由原始信号或者频谱图计算, 用原始信号更快, 用频谱图则更准确

均方根: 将 N 个项的平方和除以 N 后开平方的结果, 即均方根的结果。

在进行的过程中, 原始信号被分为指定长度的帧, 然后对每个帧求能量

如果用原始信号: power = np.mean(np.abs(signal) ** 2)
如果用频谱图(spec_complex, 即 librosa.stft 输出的复数元素的频谱图): power = np.sum(np.abs(s) ** 2, dim=帧)
"""

# 计算 RMS, 优先用 signal, 其次用 spectrum
def RMS(signal=None, spectrum=None, frame_length=2048, hop_length=512):
    """参数
        signal: librosa 读取的音频数据
        spectrum: librosa.stft() 输出的复数频谱图
        frame_length: 窗口长度, spectrum 做 STFT 时的窗口长度
        hop_length: 跳数
        ---------------------------------
        return: 均方根能量, shape:(帧数量, )
    """
    if signal is not None:
        rms = librosa.feature.rms(y=signal, frame_length=frame_length, hop_length=hop_length)
    elif spectrum is not None:
        rms = librosa.feature.rms(S=spectrum, frame_length=frame_length, hop_length=hop_length)
    else:
        return None

    rms = rms.squeeze()
    return rms

# 保存 RMS 图
def save_rms_figure(signal=None, spectrum=None, frame_length=2048, hop_length=512, save_filename=None):
    """参数
        signal: librosa 读取的音频数据
        spectrum: librosa.stft() 输出的复数频谱图
        frame_length: 窗口长度, spectrum 做 STFT 时的窗口长度
        hop_length: 跳数
        save_filename: 保存的文件名称
        --------------------------------------
        return: RMS 图保存的文件名称
    """
    plt.clf()     # 清除当前图像，避免交杂
    plt.cla()     # 清除当前坐标轴

    rms = RMS(signal=signal, spectrum=spectrum, frame_length=frame_length, hop_length=hop_length)

    plt.plot(range(1, len(rms)+1), rms, label='RMS power')    # 横坐标是帧
    plt.xlabel("Time(frame)")
    plt.ylabel("Root Mean Square Power")
    plt.title("RMS")
    plt.legend(loc='best')

    if save_filename is None:
        save_filename = random_uuid_suffix('jpg')

    plt.savefig(save_filename, dpi=600)

    return save_filename






    


