import librosa
import numpy as np
from scipy.optimize import leastsq

# NOTES = {'A':0,
#         'A♯':1,
#         'B♭':1,
#         'B':2,
#         'C':3,
#         'C♯':4,
#         'D♭':4,
#         'D':5,
#         'D♯':6,
#         'E♭':6,
#         'E':7,
#         'F':8,
#         'F♯':9,
#         'G♭':9,
#         'G':10,
#         'G♯':11,
#         'A♭':11}

NOTES = {'C':0,
        'C♯':1,
        'D♭':1,
        'D':2,
        'D♯':3,
        'E♭':3,
        'E':4,
        'F':5,
        'F♯':6,
        'G♭':6,
        'G':7,
        'G♯':8,
        'A♭':8,
        'A':9,
        'A♯':10,
        'B♭':10,
        'B':11}

# 音程差的步数计算
def note_step(note_a, note_b, bins_per_octave=12):
    """参数
        note_a: 音符 a
        note_b: 音符 b
        ------------------
        return: 音符差的步数
    """
    # 音阶
    octave_a = eval(note_a[-1])    
    octave_b = eval(note_b[-1]) 
    # 音符
    note_a = note_a[:-1]
    note_b = note_b[:-1]
    # 音程差
    res = NOTES[note_a] - NOTES[note_b] + (octave_a - octave_b) * bins_per_octave
    return res

# 计算音高相对于第一个非 nan 音符的 n_steps
def pitch_shift_steps(fundfreqs, bins_per_octave=12):
    """参数
        fundfreqs: 基频列表
        bins_per_octave: 每个音高分为几步
        ------------------------
        return: 相对于第一个非 nan 基频的音高调整 steps
    """
    f0_notes = [librosa.hz_to_note(i) if not np.isnan(i) else 'nan' for i in fundfreqs] 

    first_note_idx = -1
    shift = []

    # 寻找第一个非 nan 的音符
    for i in range(len(f0_notes)):
        if f0_notes[i] != 'nan':
            first_note_idx = i
            break
    if first_note_idx == -1:
        raise ValueError("This fundfreq has no f0 which is not nan")

    for i in range(len(f0_notes)):
        if i <= first_note_idx:
            shift.append(0)
        else:
            if f0_notes[i] == 'nan':
                shift.append(shift[-1])
            else:
                shift.append(note_step(f0_notes[i], f0_notes[first_note_idx]))

    shift = np.array(shift)
    
    return shift

# 计算音高相对于第一个非 nan 音符的 fft 频率段数量
def pitch_shift_fft_steps(fundfreqs, n_fft=2048, sr=22050):
    """参数
        fundfreqs: 要计算 shift 的 fft 频率段的数量
        n_fft: 计算基频时候的 n_fft
        sr: 采样率
        -----------------------------------------
        return: 相对于第一个非 nan 基频的音高调整 steps
    """
    fft_i = [int(hz*n_fft/sr) if not np.isnan(hz) else np.nan for hz in fundfreqs] 

    first_notnan_idx = -1
    shift = []

    # 寻找第一个非 nan 的音符
    for idx, i in enumerate(fft_i):
        if not np.isnan(i):
            first_notnan_idx = idx
            break
    if first_notnan_idx == -1:
        raise ValueError("This fundfreq has no f0 which is not nan")

    for idx, i in enumerate(fft_i):
        if idx <= first_notnan_idx:
            shift.append(0)
        else:
            if np.isnan(i):
                shift.append(shift[-1])
            else:
                shift.append(i - fft_i[first_notnan_idx])

    shift = np.array(shift)

    return shift

# 分帧
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

# 对音高调整的 steps 进行采样
def shift_sampling(shift, n):
    """参数
        shift: 音高调整的 steps, 一维 np 数组
        n: 采样点的个数
        ----------------------------
        return: 采样点, shape:(n, )
    """
    shift_n = shift.shape[0]
    step = shift_n / n
    res = np.array([shift[int(i*step)] for i in range(n)])
    return res

# 频谱图转音频
def spectrum2wav(S, n_fft=2048, center=False):
    """参数
        S: 复数元素的频谱图
        n_fft: 进行 STFT 时候使用的 n_fft
        center: 是否进行居中
        ---------------------------------
        return: 频谱图对应的音频
    """
    a = np.abs(S)
    p = 2 * np.pi * np.random.random_sample(a.shape) - np.pi   # 随机相位
    for i in range(50):
        S = a * np.exp(1j * p)
        x = librosa.istft(S, center=center)
        p = np.angle(librosa.stft(x, n_fft = n_fft, center=center))

    return x



