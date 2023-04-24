import librosa
import librosa.display
import matplotlib.pyplot as plt
import matplotlib
from .utils import *
import numpy as np
import os
matplotlib.use('agg')


"""
提取音频的谱质心:

谱质心: 信号在频谱中能量的集中点，可描述信号音色的明朗度。越亮的声音能量集中在高频部分。频谱质心的值就越大, 是描述音色属性的重要物理参数之一, 是频率成分的重心

第 t 帧的谱质心 = 该帧的频谱图(shape:[n_fft//2+1, 1])的值 * freq / 该帧的频谱图(shape:[n_fft//2+1, 1])的值的和, 即:
centroid[t] = sum_k S[k, t] * freq[k] / (sum_j S[j, t]), 其值表示的是频谱图的频率的质心
"""

# 提取音频的谱质心, 采用输入频谱的方式
def centroid(audio_data, n_fft=2048, frame_length=2048, hop_length=512):
    """参数
        audio_data: librosa 读取的音频数据
        n_fft: FFT 窗口大小
        frame_length: 每帧的长度
        hop_length: 跳数
        ------------------------------
        return: (复数元素的频谱图, 谱质心)
    """
    S, phase = librosa.magphase(librosa.stft(y=audio_data, n_fft=n_fft, frame_length=frame_length, hop_length=hop_length))
    cent = librosa.feature.spectral_centroid(S=S)

    return S, cent
    

# 保存谱质心的图像
def save_centroid_figure(audio_data, n_fft, frame_length=2048, hop_length=512, sr=22050, save_filename=None):
    """
        audio_data: librosa 读取的音频数据
        n_fft: FFT 窗口大小
        frame_length: 每帧的长度
        hop_length: 跳数
        sr: 采样率
        save_filename: 图像保存的文件位置
        -----------------------------------
        return: 图像保存的文件位置
    """
    S, cent = centroid(audio_data, n_fft=n_fft, frame_length=frame_length, hop_length=hop_length)

    plt.clf()     # 清除当前图像，避免交杂
    plt.cla()     # 清除当前坐标轴

    librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max), sr=sr, y_axis='log')

    plt.plot(cent.T, label='Spectral centroid', color='w')
    plt.xlabel("Time(frame)")
    plt.ylabel("Frequency(Hz)")
    plt.title('spectral centroid in log Power spectrogram')
    plt.legend(loc='best')
    
    if save_filename is None:
        save_filename = random_uuid_suffix('jpg')    # 随机的 jpg 文件名

    plt.savefig(save_filename, dpi=600)

    return save_filename
