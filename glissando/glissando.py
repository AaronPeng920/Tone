import librosa
import pyrubberband.pyrb as rb
import numpy as np
import soundfile as sf
from .utils import pitch_shift_steps, frame, shift_sampling, spectrum2wav, pitch_shift_fft_steps
import sys
sys.path.append('..')
from features.fundfreq import fundfreq, save_fundfreq_figure

# 使用音频信号在时域上进行音高调整
def glissando_audio(y, reference, sr=22050, frame_length=2048, hop_length=512, lib_or_rb='librosa'):
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
    if lib_or_rb == 'librosa':
        pitch_shift = librosa.effetcs.pitch_shift
    elif lib_or_rb == 'rb':
        pitch_shift = rb.pitch_shift
    else:
        raise NotImplementedError('unknown pitch shift type {}'.format(lib_or_rb))

    audio_frames = frame(y, frame_length=frame_length, hop_length=hop_length)
    n = audio_frames.shape[0]           # 当前音频的帧数量


    f0, _, _ = fundfreq(reference, frame_length=frame_length, hop_length=hop_length, sr=sr)
    shift_reference = pitch_shift_steps(f0)

    # 截断到 -6, 6
    shift_reference_cliped = np.clip(shift_reference, -6, 6)

    shift = shift_sampling(shift_reference_cliped, n)        # shift 值   

    # 平滑处理
    window_size = 5
    # 计算加权平均数
    weights = np.repeat(1.0, window_size) / window_size
    shift = np.convolve(shift, weights, 'same')

    target_frames = []    # 存储变调后的帧
    target_spectrums = []
    
    for i in range(n):
        
        # target_frame = librosa.effects.pitch_shift(audio_frames[i], sr=sr, n_steps=shift[i], bins_per_octave=12)
        target_frame = pitch_shift(audio_frames[i], sr=sr, n_steps=shift[i])

        target_spectrum = librosa.stft(target_frame, n_fft=frame_length, win_length=frame_length, hop_length=hop_length, center=False).squeeze()

        target_frames.append(target_frame)
        target_spectrums.append(target_spectrum)

    target_frames = np.array(target_frames)
    target_spectrums = np.array(target_spectrums).T

    audio = spectrum2wav(target_spectrums, n_fft=frame_length)

    return audio, shift_reference, shift

# 使用频谱图在频域上进行音高调整
def glissando_spectrum(y, reference, sr=22050, frame_length=2048, hop_length=512):
    """参数
        y: 音频信号
        reference: 参考的滑音片段, 由 librosa 读取得到
        sr: 音频的采样率
        frame_length: 帧长度
        hop_length: 帧跳数
        ---------------------------
        return: 颤音化处理后的音频信号, 参考音频的波动值, 从参考音频的波动值采样的波动值
    """
    S_comp = librosa.stft(y, n_fft=frame_length, win_length=frame_length, hop_length=hop_length)
    n = S_comp.shape[-1]
    


    f0, _, _ = fundfreq(reference, frame_length=frame_length, hop_length=hop_length, sr=sr)

    shift_reference = pitch_shift_fft_steps(f0, n_fft=frame_length, sr=sr)

    # 截断到 -6, 6
    shift_reference_cliped = np.clip(shift_reference, -6, 6)
    # 采样
    shift = shift_sampling(shift_reference_cliped, n)  # 实际的 shift 值
    # 平滑处理
    window_size = 5
    # 计算加权平均数
    weights = np.repeat(1.0, window_size) / window_size
    shift = np.convolve(shift, weights, 'same')

    shift = np.array([int(i) for i in shift])  # shift 进行整数化

    for i in range(n):
        shift_scale = shift[i]
        S_comp[:,i] = np.roll(S_comp[:,i], shift_scale)

        if shift_scale > 0:
            S_comp[:shift_scale,i] = 0
        elif shift_scale < 0:
            S_comp[shift_scale:,i] = 0
        else:
            pass
    
    target_data = spectrum2wav(S_comp, n_fft=frame_length, center=True)

    return target_data, shift_reference_cliped, shift


##################### 以下是用于 gradio 的 API ####################
def glissando_api(process_filename, reference_filename, type='时域', sr=44100, frame_length=2048, hop_length=512, use_gradio=True):
    """参数
        process_filename: 要处理的音频文件名
        reference_filename: 参考的滑音的音频文件名
        type: 处理的类型, `时域` or `频域`
        sr: 采样率
        frame_length: 帧长度
        hop_length: 帧移
        use_gradio: 上述传递的文件名是否是 gradio 的接收的
        ------------------------------------
        return: 原音频基频图, 原音频, 参考音频基频图, 参考音频, 生成音频基频图, 生成音频
    """

    if use_gradio:
        process_filename = process_filename.name
        reference_filename = reference_filename.name
        
    
    process_data, sr = librosa.load(process_filename, sr=sr)
    reference_data, sr = librosa.load(reference_filename, sr=sr)

    if type == '时域':
        result_audio, shift_ref, shift = glissando_audio(process_data, reference_data, sr=sr, frame_length=frame_length, hop_length=hop_length)
    elif type == '频域':
        result_audio, shift_ref, shift = glissando_spectrum(process_data, reference_data, sr=sr, frame_length=frame_length, hop_length=hop_length)
    else:
        raise NotImplementedError('unknown process type of `{}`'.format(type))

    result_audio_path = 'gradio_temp/glissando_result_audio.wav'
    sf.write(result_audio_path, result_audio, sr)

    origin_fundfreq_path = 'gradio_temp/glissando_origin_fundfreq.png'
    save_fundfreq_figure(process_data, save_filename=origin_fundfreq_path)

    reference_fundfreq_path = 'gradio_temp/glissando_reference_fundfreq.png'
    save_fundfreq_figure(reference_data, save_filename=reference_fundfreq_path)

    result_fundfreq_path = 'gradio_temp/glissando_result_fundfreq.png'
    result_data, sr = librosa.load(result_audio_path, sr=sr)
    save_fundfreq_figure(result_data, save_filename=result_fundfreq_path)

    return origin_fundfreq_path, process_filename, reference_fundfreq_path, reference_filename, result_fundfreq_path, result_audio_path


    



    







    
    

