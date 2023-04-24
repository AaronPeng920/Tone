import librosa
import librosa.display
import matplotlib.pyplot as plt
import matplotlib
from .utils import *
import numpy as np
import os
matplotlib.use('agg')


"""
提取音频的基音频率

基频: 傅立叶变换后频率最低的波就是基音, 对应的频率就是基频, 其他频率为泛音, 频率越高分配到的能量越少
基频是区别音高的主要成分, 泛音则决定乐器或人声的声色

1. 泛音(Overtone)在声学和音乐中, 指一个声音中除了基频外其他频率的音
2. 频率都是某一频率的倍数, 这一频率就称作基频, 也就决定了这个音的音高
3. 基频为 f, 频率为 2f 的音称为第一泛音, 频率为 3f 的音称为第二泛音
4. 泛音越充分的声音越饱满
5. 低频泛音越充分的声音听起来越 "厚实", 越 "有力"
6. 高频泛音越充分的声音穿透力越强, 声音听起来越 "亮", 越 "尖"
7. 高低频都有并且合理分布的声音, 就是比较完美的声音

根据基频可以解耦音色和内容, 参见 https://blog.csdn.net/u013625492/article/details/112564667
1. 第一共振峰, 第二共振峰等的位置, 特别是相对位置, 决定了发音内容. 不同元音对不同倍数泛音共振加强不同, 体现的也有一部分是能量的大小相对差异
2. 基频的高度, 共振峰的绝对高度, 也一定程度和发音内容相关, 但是需要减掉说话人的平均基频值再去看
3. 人的基频, 共振峰等的绝对高度, 和音色相关, 比如性别的不同, FO范围的不同
4. 共振峰的相对位置, 最大的信息时发音内容, 但是相同发音內容, 又会有每个人的发音习惯和口腔结构, 所以次要信息也有音色信息. 这点和 speaker identity 更像, ASV特征
5. 真正的 "厚实,亮,尖,好听" 也算作音色, 但是是同一个人也可以模拟的, 比如单人多角色小说,唱歌等. 不同共振峰频率分配的能量, 会导致听感.人和人之间区别很大, 也可以导致 ASV
"""

# 提起音频的基频
def fundfreq(audio_data, frame_length=2048, hop_length=512, sr=22050, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7')):
    """参数
        audio_data: librosa 读取的音频数据
        frame_length: 帧长度, 默认 2048
        hop_length: 跳数
        sr: 采样率
        fmin: 最小频率, 'C2': 2rd do, 65.4hz
        fmax: 最大频率, 'C7': 7th do, 2093hz
        -----------------------------------------
        return: (基频, 该帧是否是人声, 该帧是人声的可能性), shape:(帧数量,)
    """
    # 返回每一帧的 f0:基频, voiced_flag:该帧是否是人声, voiced_probs:该帧是人声的可能性
    f0, voiced_flag, voiced_probs = librosa.pyin(audio_data, frame_length=frame_length, hop_length=hop_length, sr=sr, fmin=fmin, fmax=fmax) 
    return f0, voiced_flag, voiced_probs

# 保存基频图
def save_fundfreq_figure(audio_data, frame_length=2048, hop_length=512, sr=22050, 
                        fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'),
                        save_filename=None):
    """参数
        audio_data: librosa 读取的音频数据
        frame_length: 帧长度, 默认 2048
        hop_length: 跳数
        sr: 采样率
        fmin: 最小频率, 'C2': 2rd do, 65.4hz
        fmax: 最大频率, 'C7': 7th do, 2093hz
        save_filename: 保存的文件路径
        -----------------------------------------
        return: 保存的文件路径
    """

    f0, voiced_flag, voiced_probs = fundfreq(audio_data, frame_length, hop_length, sr, fmin, fmax)

    plt.clf()     # 清除当前图像，避免交杂
    plt.cla()     # 清除当前坐标轴

    spec = librosa.stft(audio_data, n_fft=frame_length, hop_length=hop_length)
    D = librosa.amplitude_to_db(np.abs(spec), ref=np.max)

    librosa.display.specshow(D, sr=sr, y_axis='log')

    plt.plot(f0, label='fundamental frequency', color='w')
    plt.xlabel("Time(frame)")
    plt.ylabel("Frequency(Hz)")
    plt.title('Fundamental Frequency(F0) in Log Power Spectrogram')
    plt.legend(loc='best')
    
    if save_filename is None:
        save_filename = random_uuid_suffix('jpg')    # 随机的 jpg 文件名

    plt.savefig(save_filename, dpi=600)

    return save_filename


