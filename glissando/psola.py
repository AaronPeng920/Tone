import numpy as np
import scipy.signal as signal
import librosa
import soundfile as sf

def psola_pitch_shift(y, sr, n_steps, ws=2048, hop=512):
    """
    使用PSOLA算法实现Pitch Shift
    :param y: 音频信号
    :param sr: 采样率
    :param n_steps: Pitch shift步数
    :param ws: 窗口大小，默认为2048
    :param hop: 步长，默认为512
    :return: Pitch shift后的音频信号
    """
    # 计算Pitch shift比例
    pitch_ratio = 2 ** (n_steps / 12)

    # 定义变换窗口
    window = signal.hann(ws)

    # 计算需要移动的帧数
    n_frames = int(np.ceil((len(y) - ws) / hop))

    # 初始化输出数组
    y_shifted = np.zeros(len(y) + int(np.ceil(n_frames * hop * pitch_ratio)))

    # 初始化加权系数数组
    window_sum = np.zeros(len(y_shifted))

    # 遍历所有帧
    for i in range(n_frames):
        # 计算当前帧的加权系数
        start = i * hop
        end = start + ws
        window_sum[start:end] += window
        windowed_frame = window * y[start:end]

        # 计算目标帧的位置
        target_start = int(start * pitch_ratio)
        target_end = target_start + ws

        # 将当前帧添加到目标位置
        y_shifted[target_start:target_end] += windowed_frame
    
    

    # 对加权系数取倒数，防止出现除0错误
    window_sum[window_sum == 0] = 1
    y_shifted /= window_sum

    y_shifted = y_shifted[:target_end]

    # 调整采样率
    sr = int(sr * pitch_ratio)

    # 返回Pitch shift后的音频信号和采样率
    return y_shifted, sr

if __name__ == '__main__':
    audio_data, sr = librosa.load('../audios/duan_singing.wav', sr=44100)
    y, sr = psola_pitch_shift(audio_data, sr, 4)
    print(y.shape)
    sf.write('tmp.wav', y, sr)


