import librosa
import numpy as np
import matplotlib.pyplot as plt
from .utils import frame, spectrum2wav
from .fluctuation import sine, triangle
from .noise import gause_noise

def vibrato(y, sr=22050, frame_length=2048, hop_length=512, fluctuation='sine'):
    """参数
        y: 音频信号
        sr: 音频的采样率
        frame_length: 帧长度
        hop_length: 帧跳数
        fluctuation: 波动函数
        ---------------------------
        return: 颤音化处理后的音频信号
    """
    audio_frames = frame(y, frame_length=frame_length, hop_length=hop_length)
    n = audio_frames.shape[0]

    if fluctuation == 'sine':
        _, fluct = sine(1, 1, 0, n, 0, 10 * 2 * np.pi)
    elif fluctuation == 'triangle':
        _, fluct = triangle(1, 2, n, 0, 20)
    else:
        raise NotImplementedError('Not inplemented fluctuation of {}'.format(fluctuation))


    # 引入噪声
    # noise = gause_noise(0, 0.25, n, -0.5, 0.5)
    # fluct = fluct + noise

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



    







