import numpy as np
import librosa


def frame(y, frame_length=2048, hop_length=512):
    """参数
        y: 待分帧的音频信号
        frame_length: 帧长度
        hop_length: 跳数
        ----------------------------
        return: 帧信号, shape:(帧数量, frame_length)
    """
    frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length, axis=0)
    return frames

def frame_concat(frames, frame_length=2048, hop_length=512):
    """参数
        frames: 待拼接的帧, shape:(帧数量, frame_length)
        frame_length: 帧长度
        hop_length: 帧与帧之间的跳数
        ---------------------------
        return: 拼接后的信号, shape:(采样点数量, )
    """
    pass

# 频谱图转音频
def spectrum2wav(S, n_fft=2048):
    """参数
        S: 复数元素的频谱图
        n_fft: 进行 STFT 时候使用的 n_fft
        ---------------------------------
        return: 频谱图对应的音频
    """
    a = np.abs(S)
    p = 2 * np.pi * np.random.random_sample(a.shape) - np.pi   # 随机相位
    for i in range(50):
        S = a * np.exp(1j * p)
        x = librosa.istft(S, center=False)
        p = np.angle(librosa.stft(x, n_fft = n_fft, center=False))

    return x