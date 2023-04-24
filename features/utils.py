import uuid
import librosa
import soundfile as sf
from packaging import version
import os
from glob import glob

__all__ = [
    "librosa_load",
    "librosa_write",
    "random_uuid_suffix"
]


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

# 读取音频文件
def librosa_load(filename, sr=22050, start_time=None, end_time=None):
    """参数
        filename: 文件名
        sr: 采样率
        start_time: 开始时间(秒)
        end_time: 结束时间(秒)
        --------------------------------
        return: 读取到的音频数据
    """
    audio_data, _ = librosa.load(filename, sr=sr)

    if (start_time is None) or (start_time < 0) or (start_time * sr > audio_data.shape[0]):
        start_time = 0
    else:
        start_time = start_time * sr
    
    if (end_time is None) or (end_time < 0) or (end_time * sr > audio_data.shape[0]):
        end_time = audio_data.shape[0]
    else:
        end_time =  end_time * sr
    
    audio_data = audio_data[int(start_time):int(end_time)]
    return audio_data

# 将音频写入到文件中
def librosa_write(filename, x, sr=22050):
    """参数
        filename: 写入到的文件路径
        x: 音频数据
        sr: 写入时采用的采样率
    """
    if version.parse(librosa.__version__) < version.parse('0.8.0'):
        librosa.output.write_wav(filename, x, sr)
    else:
        sf.write(filename, x, sr)