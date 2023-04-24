import librosa
import numpy as np
import sys
sys.path.append('..')
from features.rms import RMS
from features.spectrum import spectrum

# 频谱图振幅补丁
def amplitude_patch(origin_S_complex, target_S_complex):
    """参数
        origin_S_abs: 待计算补丁的 complex 频谱图, shape:(n_fft//2+1, 帧数量)
        target_S_abs: 目标 complex 频谱图, shape:(n_fft//2+1, 帧数量)
        --------------------------------
        return: abs 频谱图振幅补丁, 保持原频谱图各频率的 abs 幅度平方的比例, shape:(n_fft//2+1, 帧数量)
    """
    assert origin_S_complex.shape == target_S_complex.shape
    

    ffts_count = origin_S_complex.shape[0]
    frame_count = origin_S_complex.shape[1]

    frame_length = int((ffts_count - 1) * 2)

    origin_rms = RMS(signal=None, spectrum=origin_S_complex)
    target_rms = RMS(signal=None, spectrum=target_S_complex)
    # 计算每帧的所有频率的 abs 幅值的平方的和
    origin_rms_ready = (origin_rms ** 2) * (frame_length ** 2) / 2
    target_rms_ready = (target_rms ** 2) * (frame_length ** 2) / 2

    origin_S_abs = np.abs(origin_S_complex)
    target_S_abs = np.abs(target_S_complex)

    
    patch = np.zeros([ffts_count, frame_count])

    # 逐帧的进行处理
    for i in range(frame_count):
        origin_abs_frame_i = origin_S_abs[:,i]   # 该帧的各个频率的原始 abs
        target_rms_frame_i = target_rms_ready[i]  # 第 i 帧的目的 abs 幅值的平方和
        origin_rms_frame_i = origin_rms_ready[i]  # 第 i 帧的原始 abs 幅值的平方和

        # 该帧每个频率的 abs 的平方占总的 abs 平方的和的比例
        # 调用直流和 sr/2 的部分, 即 fft_idx = 0 和 fft_idx = n_fft//2+1
        if frame_length % 2 == 0:
            origin_abs_ratio_frame_i = np.array(
                                                [(origin_abs_frame_i[j] ** 2)/origin_rms_frame_i 
                                                if j != 0 and j != ffts_count-1 
                                                else  (origin_abs_frame_i[j] ** 2 * 0.5)/origin_rms_frame_i 
                                                for j in range(ffts_count)])
        else:
            origin_abs_ratio_frame_i = np.array(
                                                [(origin_abs_frame_i[j] ** 2)/origin_rms_frame_i 
                                                if j != 0 
                                                else  (origin_abs_frame_i[j] ** 2 * 0.5)/origin_rms_frame_i 
                                                for j in range(ffts_count)])

        # 该帧的每个频率的目的 abs 平方
        target_rmsready_frame_i_j = [target_rms_frame_i * ratio for ratio in origin_abs_ratio_frame_i]

        # 该帧的每个频率的初始的 abs 平方
        origin_rmsready_frame_i_j = [origin_rms_frame_i * ratio for ratio in origin_abs_ratio_frame_i]

        for j in range(ffts_count):
            patch[j][i] = np.sqrt(target_rmsready_frame_i_j[j]) - np.sqrt(origin_rmsready_frame_i_j[j])

    return patch