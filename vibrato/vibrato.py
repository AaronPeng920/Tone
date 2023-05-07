import librosa
import numpy as np
import matplotlib.pyplot as plt
from .utils import frame, spectrum2wav
from .fluctuation import sine, uniform_sine, gause_sine, increase_f_sine, decrease_f_sine
from .noise import gause_noise
import sys
sys.path.append('..')
from features.fundfreq import save_fundfreq_figure
import soundfile as sf

# 标准正弦波颤音
def vibrato_sine(y, A=1, w=1, b=0, T=10, sr=22050, frame_length=2048, hop_length=512):
    """参数
        y: 音频信号
        A, w, b: 三角函数 A*sin(wx)+b
        T: 波动的周期数量
        sr: 音频的采样率
        frame_length: 帧长度
        hop_length: 帧跳数
        fluctuation: 波动函数
        ---------------------------
        return: 颤音化处理后的音频信号
    """
    audio_frames = frame(y, frame_length=frame_length, hop_length=hop_length)
    n = audio_frames.shape[0]


    _, fluct = sine(A, w, b, n, 0, 2 * np.pi / w * T)
    
    target_frames = []    # 存储变调后的帧
    target_spectrums = []
    
    for i in range(n):
        target_frame = librosa.effects.pitch_shift(audio_frames[i], sr=sr, n_steps=fluct[i], bins_per_octave=12)
        target_spectrum = librosa.stft(target_frame, n_fft=frame_length, win_length=frame_length, hop_length=hop_length, center=False).squeeze()

        target_frames.append(target_frame)
        target_spectrums.append(target_spectrum)

    target_frames = np.array(target_frames)
    target_spectrums = np.array(target_spectrums).T

    audio = spectrum2wav(target_spectrums, n_fft=frame_length)

    return audio

# 频率均匀分布的正弦波颤音
def vibrato_uniformsine(y, A=1, low=200, high=300, sr=22050, frame_length=2048, hop_length=512):
    """参数
        y: 待颤音化的音频信号
        A: 正弦波幅度
        low: 频率下限
        high: 频率上限
        sr: 采样率
        frame_length: 每帧的长度
        hop_length: 帧跳数
        ------------------------
        return: 颤音化后的音频信号
    """
    audio_frames = frame(y, frame_length=frame_length, hop_length=hop_length)
    n = audio_frames.shape[0]


    _, fluct = uniform_sine(A, low, high, n)
    
    target_frames = []    # 存储变调后的帧
    target_spectrums = []
    
    for i in range(n):
        target_frame = librosa.effects.pitch_shift(audio_frames[i], sr=sr, n_steps=fluct[i], bins_per_octave=12)
        target_spectrum = librosa.stft(target_frame, n_fft=frame_length, win_length=frame_length, hop_length=hop_length, center=False).squeeze()

        target_frames.append(target_frame)
        target_spectrums.append(target_spectrum)

    target_frames = np.array(target_frames)
    target_spectrums = np.array(target_spectrums).T

    audio = spectrum2wav(target_spectrums, n_fft=frame_length)

    return audio

# 频率高斯分布的正弦波颤音
def vibrato_gausesine(y, A=1, mean=0, square=1, sr=22050, frame_length=2048, hop_length=512):
    """参数
        y: 待颤音化的音频信号
        A: 正弦波幅度
        mean: 频率均值
        square: 频率方差
        sr: 采样率
        frame_length: 每帧的长度
        hop_length: 帧跳数
        ------------------------
        return: 颤音化后的音频信号
    """
    audio_frames = frame(y, frame_length=frame_length, hop_length=hop_length)
    n = audio_frames.shape[0]


    _, fluct = gause_sine(A, mean, square, n)
    
    target_frames = []    # 存储变调后的帧
    target_spectrums = []
    
    for i in range(n):
        target_frame = librosa.effects.pitch_shift(audio_frames[i], sr=sr, n_steps=fluct[i], bins_per_octave=12)
        target_spectrum = librosa.stft(target_frame, n_fft=frame_length, win_length=frame_length, hop_length=hop_length, center=False).squeeze()

        target_frames.append(target_frame)
        target_spectrums.append(target_spectrum)

    target_frames = np.array(target_frames)
    target_spectrums = np.array(target_spectrums).T

    audio = spectrum2wav(target_spectrums, n_fft=frame_length)

    return audio

# 频率逐渐增高的正弦波颤音
def vibrato_increase_f_sine(y, p=1, sr=22050, frame_length=2048, hop_length=512):
    """参数
        y: 待颤音化的音频信号
        p: 频率变化程度 1-10
        sr: 采样率
        frame_length: 每帧的长度
        hop_length: 帧跳数
        -----------------------------------
        return: 颤音化后的音频信号
    """
    audio_frames = frame(y, frame_length=frame_length, hop_length=hop_length)
    n = audio_frames.shape[0]


    _, fluct = increase_f_sine(p, n)
    
    target_frames = []    # 存储变调后的帧
    target_spectrums = []
    
    for i in range(n):
        target_frame = librosa.effects.pitch_shift(audio_frames[i], sr=sr, n_steps=fluct[i], bins_per_octave=12)
        target_spectrum = librosa.stft(target_frame, n_fft=frame_length, win_length=frame_length, hop_length=hop_length, center=False).squeeze()

        target_frames.append(target_frame)
        target_spectrums.append(target_spectrum)

    target_frames = np.array(target_frames)
    target_spectrums = np.array(target_spectrums).T

    audio = spectrum2wav(target_spectrums, n_fft=frame_length)

    return audio

# 频率逐渐降低的正弦波颤音
def vibrato_decrease_f_sine(y, p=1, sr=22050, frame_length=2048, hop_length=512):
    """参数
        y: 待颤音化的音频信号
        p: 频率变化程度 1-10
        sr: 采样率
        frame_length: 每帧的长度
        hop_length: 帧跳数
        -----------------------------------
        return: 颤音化后的音频信号
    """
    audio_frames = frame(y, frame_length=frame_length, hop_length=hop_length)
    n = audio_frames.shape[0]


    _, fluct = decrease_f_sine(p, n)
    
    target_frames = []    # 存储变调后的帧
    target_spectrums = []
    
    for i in range(n):
        target_frame = librosa.effects.pitch_shift(audio_frames[i], sr=sr, n_steps=fluct[i], bins_per_octave=12)
        target_spectrum = librosa.stft(target_frame, n_fft=frame_length, win_length=frame_length, hop_length=hop_length, center=False).squeeze()

        target_frames.append(target_frame)
        target_spectrums.append(target_spectrum)

    target_frames = np.array(target_frames)
    target_spectrums = np.array(target_spectrums).T

    audio = spectrum2wav(target_spectrums, n_fft=frame_length)

    return audio


# 用于 gradio 的 API, 当前默认使用正弦波
def vibrato_api(filename, flucttype='标准的正弦波', A=1, w=1, b=0, T=10, low=200, high=300, mean=0, square=1, p=1, sr=44100):
    """参数
        filename: gradio 接收的文件名
        flucttype: 波动类型
        A, w, b: 三角函数 A*sin(wx)+b
        T: 波动的周期数量
        low: 频率下限
        high: 频率上限
        mean: 音频波动均值
        square: 音频波动方差
        p: 频率波动程度
        sr: 采样率
        ----------------------------------------
        return: 原音频文件基频图路径, 原音频文件路径, 颤音化后的基频图路径, 颤音化后的音频文件路径
    """
    audio_data, sr = librosa.load(filename.name, sr=sr)

    if flucttype == "标准的正弦波":
        result_audio_data = vibrato_sine(audio_data, A=A, w=w, b=b, T=T, sr=sr)
    elif flucttype == '频率均匀分布的正弦波':
        result_audio_data = vibrato_uniformsine(audio_data, A=A, low=low, high=high, sr=sr)
    elif flucttype == '频率高斯分布的正弦波':
        result_audio_data = vibrato_gausesine(audio_data, A=A, mean=mean, square=square, sr=sr)
    elif flucttype == '频率升高的正弦波':
        result_audio_data = vibrato_increase_f_sine(audio_data, p=p, sr=sr)
    elif flucttype == '频率降低的正弦波':
        result_audio_data = vibrato_decrease_f_sine(audio_data, p=p, sr=sr)


    result_audio_path = 'vibrato_result_audio.wav'
    sf.write(result_audio_path, result_audio_data, sr)

    origin_fundfreq_path = 'vibrato_origin_fundfreq.png'
    save_fundfreq_figure(audio_data, save_filename=origin_fundfreq_path)
    result_fundfreq_path = 'vibrato_result_fundfreq.png'
    result_audio_data, _ = librosa.load(result_audio_path, sr=sr)
    save_fundfreq_figure(result_audio_data, save_filename=result_fundfreq_path)

    
    return origin_fundfreq_path, filename.name, result_fundfreq_path, result_audio_path












