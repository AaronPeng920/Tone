import librosa
import numpy as np
import uuid

# 频谱图转音频
def spectrum2wav(S_db, p=None, n_fft=2048):
    """参数
        S: 分贝元素的频谱图
        p: 原相位信息
        n_fft: 进行 STFT 时候使用的 n_fft
        ---------------------------------
        return: 频谱图对应的音频
    """
    a = librosa.db_to_amplitude(S_db)
    if p is None:
        p = 2 * np.pi * np.random.random_sample(a.shape) - np.pi   # 随机相位
        for i in range(50):
            S = a * np.exp(1j * p)
            x = librosa.istft(S)
            p = np.angle(librosa.stft(x, n_fft = n_fft))
    else:
        S = a * np.exp(1j * p)
        x = librosa.istft(S)

    return x


# 频率值转 n_fft 的索引
def hz_to_fft(hz, n_fft=2048, sr=44100):
    """参数
        hz: 实际的频率值
        n_fft: FFT 窗口的大小
        sr: 采样率
        --------------------
        return: hz 对应的 n_fft 的索引 

        hz = (idx * sr) / n_fft
    """
    idx = int((hz * n_fft) / sr)
    return idx

# 生成指定后缀的随机文件名
def random_uuid_suffix(suffix):
    """参数
        suffix: str, 不含 . 的扩展名, 如 '.wav' 音频可以设置为 'wav'
        --------------------------
        return: 指定扩展名的不重复随机文件名
    """
    random_uuid = uuid.uuid4().hex
    res = random_uuid + '.' + suffix
    return res