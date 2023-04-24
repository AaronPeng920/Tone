import librosa
import librosa.display
import matplotlib.pyplot as plt
import matplotlib
from .utils import *
import os
import numpy as np
matplotlib.use('agg')

"""
绘制音频的频谱图
"""

# 获取频谱图
def spectrum(signal, n_fft=2048, win_length=2048, hop_length=512):
    """参数
        signal: librosa 读取的音频数据
        n_fft: N_FFT 大小, 频率的个数
        win_length: 窗口的长度
        hop_length: 跳数
        -------------------------------------
        return: 复数元素的频谱图
    """
    S = librosa.stft(signal, n_fft=n_fft, win_length=win_length, hop_length=hop_length)  
    return S  

# 显示分贝做元素的对数频谱图
def save_spectrum_figure(signal, n_fft=2048, win_length=2048, hop_length=512, sr=22050, save_filename=None):
    """参数
        signal: librosa 读取的音频数据
        n_fft: N_FFT 大小, 频率的个数
        win_length: 窗口的长度
        hop_length: 跳数
        sr: 采样率
        save_filename: 保存的文件名
        --------------------------
        return: 保存图片的文件名
    """

    S = spectrum(signal, n_fft, win_length=win_length, hop_length=hop_length)
    S = librosa.amplitude_to_db(np.abs(S), ref=np.max)

    plt.clf()     # 清除当前图像，避免交杂
    plt.cla()     # 清除当前坐标轴

    librosa.display.specshow(S, y_axis='log', sr=sr)
    plt.colorbar(format='%+2.0f dB')
    plt.xlabel("Time(frame)")
    plt.ylabel("Frequency(Hz)")
    plt.title("STFT power spectrum")

    if save_filename is None:
        save_filename = random_uuid_suffix('jpg')    # 随机的 jpg 文件名

    plt.savefig(save_filename, dpi=600)

    return save_filename

