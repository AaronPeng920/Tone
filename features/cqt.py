import librosa
import librosa.display
import matplotlib.pyplot as plt
import matplotlib
from .utils import *
import numpy as np
import os
matplotlib.use('agg')

# 常数 Q 变换
def cqt(y, sr=22050, hop_length=512, n_bins=84, bins_per_octave=12):
    """参数
        y: 要分析的音频信号
        sr: 采样率
        hop_length: 跳数
        n_bins: 频率箱的个数
        bins_per_octave: 每个音阶含有音符数量
        -------------------------------------
        return: 复数元素表示的频谱图, shape:(n_bins, 帧数量)
    """
    C = librosa.cqt(y, sr=sr, hop_length=hop_length, n_bins=n_bins, bins_per_octave=bins_per_octave)
    return C

# 显示分贝做元素的 CQT 频谱图
def save_cqt_figure(signal, sr=22050, hop_length=512, n_bins=84, bins_per_octave=12, save_filename=None):
    """参数
        signal: librosa 读取的音频数据
        sr: 采样率
        hop_length: 跳数
        n_bins: 频率箱的个数
        bins_per_octave: 每个音阶含有音符数量
        save_filename: 保存的文件名
        --------------------------
        return: 保存图片的文件名
    """

    C = cqt(signal, sr=sr, hop_length=hop_length, n_bins=n_bins, bins_per_octave=bins_per_octave)
    C = librosa.amplitude_to_db(np.abs(C), ref=np.max)

    plt.clf()     # 清除当前图像，避免交杂
    plt.cla()     # 清除当前坐标轴

    librosa.display.specshow(C, y_axis='cqt_note', sr=sr)
    plt.colorbar(format='%+2.0f dB')
    plt.xlabel("Time(frame)")
    plt.ylabel("Note")
    plt.title("Constant-Q power spectrum")

    if save_filename is None:
        save_filename = random_uuid_suffix('jpg')    # 随机的 jpg 文件名

    plt.savefig(save_filename, dpi=600)

    return save_filename

