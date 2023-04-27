import librosa
import numpy as np
from .utils import pitch_shift_steps, frame, shift_sampling, spectrum2wav
import sys
sys.path.append('..')
from features.fundfreq import fundfreq

# 进行音高调整
def glissando(y, reference, sr=22050, frame_length=2048, hop_length=512):
    """参数
        y: 音频信号
        reference: 参考的滑音片段, 由 librosa 读取得到
        sr: 音频的采样率
        frame_length: 帧长度
        hop_length: 帧跳数
        fluctuation: 波动函数
        ---------------------------
        return: 颤音化处理后的音频信号, 参考音频的波动值, 从参考音频的波动值采样的波动值
    """
    audio_frames = frame(y, frame_length=frame_length, hop_length=hop_length)
    n = audio_frames.shape[0]           # 当前音频的帧数量

    f0, _, _ = fundfreq(reference, frame_length=frame_length, hop_length=hop_length, sr=sr)
    shift_reference = pitch_shift_steps(f0)

    shift = shift_sampling(shift_reference, n)        # shift 值


    # 引入噪声
    # noise = gause_noise(0, 0.25, n, -0.5, 0.5)
    # fluct = fluct + noise

    target_frames = []    # 存储变调后的帧
    target_spectrums = []
    
    for i in range(n):
        target_frame = librosa.effects.pitch_shift(audio_frames[i], sr=sr, n_steps=shift[i], bins_per_octave=12)
        target_spectrum = librosa.stft(target_frame, n_fft=frame_length, win_length=frame_length, hop_length=hop_length, center=False).squeeze()

        target_frames.append(target_frame)
        target_spectrums.append(target_spectrum)

    target_frames = np.array(target_frames)
    target_spectrums = np.array(target_spectrums).T

    audio = spectrum2wav(target_spectrums, n_fft=frame_length)

    return audio,shift_reference, shift





    
    

