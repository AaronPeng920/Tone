import librosa
import numpy as np
import wave
import soundfile as sf
import sys
sys.path.append('..')
from features.fundfreq import fundfreq, save_fundfreq_figure
from features.spectrum import spectrum
import gradio as gr
from utils import spectrum2wav
from amplitude_patch import amplitude_patch

# librosa 实现的音高调整的槽函数
def pitch_shifter_librosa(filename, n_steps=0, bins_per_octave=12, sr=44100):
    """参数
        filename: gradio 接收的文件名
        n_steps: 音高调整的步数
        bins_per_octave: 一个音阶分成指定的步数
        sr: 采样率
        -------------------------------
        return: (原音频基频图路径, 原音频文件路径, 生成音频基频图路径, 生成音频文件路径)
    """
    bins_per_octave = int(bins_per_octave)
    sr = int(sr)

    audio_data, sr = librosa.load(filename.name, sr=sr)

    origin_audio_path = filename.name
    target_audio_path = 'gradio_temp/pitch_shifted_librosa_audio.wav'
    origin_fundfreq_path = 'gradio_temp/pitch_shift_librosa_origin_fundfreq.jpg'
    target_fundfreq_path = 'gradio_temp/pitch_shifted_librosa_fundfreq.jpg'

    origin_fundfreq_path = save_fundfreq_figure(audio_data, sr=sr, save_filename=origin_fundfreq_path)

    target = librosa.effects.pitch_shift(audio_data, sr=sr, n_steps=n_steps, bins_per_octave=bins_per_octave)
    sf.write(target_audio_path, target, sr)
    target_fundfreq_path = save_fundfreq_figure(target, sr=sr, save_filename=target_fundfreq_path)

    return origin_fundfreq_path, origin_audio_path, target_fundfreq_path, target_audio_path

# wave 实现的音高调整的槽函数
def pitch_shifter_spectrum(filename, n_steps=0, bins_per_octave=12, sr=44100):
    """参数
        filename: gradio 接收的文件名
        n_steps: 音高调整的步数
        bins_per_octave: 一个音阶分成指定的步数
        sr: 采样率
        -------------------------------
        return: (原音频基频图路径, 原音频文件路径, 生成音频基频图路径, 生成音频文件路径)
    """
    audio_data, sr = librosa.load(filename.name, sr=sr)
    S_comp = librosa.stft(audio_data)
    f0, _, _ = fundfreq(audio_data)
    n = S_comp.shape[-1]   # 帧数量

    # 记录每一帧的改变量
    shift = []
    for fi in f0:
        if np.isnan(fi):
            shift.append(0)
        else:
            target_f = fi * (np.power(2, 1/bins_per_octave) ** n_steps)       # 目标频率
            f0_i = int(fi * 2048 / sr)           # 原来的 fft 索引
            target_i = int(target_f * 2048 / sr)     # 目标的 fft 索引
            shift.append(target_i - f0_i)
    
    for i in range(n):
        shift_scale = shift[i]
        S_comp[:,i] = np.roll(S_comp[:,i], shift_scale)

        if shift_scale > 0:
            S_comp[:shift_scale,i] = 0
        elif shift_scale < 0:
            S_comp[shift_scale:,i] = 0
        else:
            pass
    
    S_db = librosa.amplitude_to_db(np.abs(S_comp))
    target_data = spectrum2wav(S_db, n_fft=2048)
    target_audio_path = 'gradio_temp/pitch_shifted_spectrum_target_audio.wav'
    sf.write(target_audio_path, target_data, int(sr))

    origin_fundfreq_path = 'gradio_temp/pitch_shifted_spectrum_origin_fundfreq.png'
    save_fundfreq_figure(audio_data, save_filename=origin_fundfreq_path)
    target_audio_data, sr = librosa.load(target_audio_path, sr=sr)
    target_fundfreq_path = 'gradio_temp/pitch_shifted_spectrum_target_fundfreq.png'
    save_fundfreq_figure(target_audio_data, save_filename=target_fundfreq_path)

    return origin_fundfreq_path, filename.name, target_fundfreq_path, target_audio_path
    
# API
def pitch_shift(filename, func, n_steps=0, bins_per_octave=12, sr=44100):
    """参数
        filename: gradio 接收的文件名
        func: 计算的函数类型, `时域` 或者 `频域`
        n_steps: 音高调整的步数
        bins_per_octave: 一个音阶分成指定的步数
        sr: 采样率
        -------------------------------
        return: (原音频基频图路径, 原音频文件路径, 生成音频基频图路径, 生成音频文件路径)
    """
    if func == '时域':
        return pitch_shifter_librosa(filename, n_steps, bins_per_octave, sr)
    elif func == '频域':
        return pitch_shifter_spectrum(filename, n_steps, bins_per_octave, sr)
    else:
        raise NotImplementedError('unknown function type of {}'.format(func))

# 音高调整器类
class PitchShifter:
    def __init__(self):
        with gr.Blocks() as self.demo:
            with gr.Blocks():
                with gr.Row():
                    self.input_file = gr.File(label='音频文件')
                    with gr.Column():
                        self.functype = gr.Dropdown(choices=['时域', '频域'], value='时域', label='算法类型')
                        self.n_steps = gr.Number(label='改变步数', value=0)
                        self.bins_per_octave = gr.Number(label='音阶箱数', value=12)
                        self.sr = gr.Number(label='采样率', value=44100)
                        self.btn = gr.Button(value='计算')

            with gr.Blocks():
                with gr.Row():
                    with gr.Blocks():
                        with gr.Column():
                            self.origin_fundfreq = gr.Image(label='原音频基频图')
                            self.origin_audio = gr.Audio(label='原音频')
                    with gr.Blocks():
                        with gr.Column():
                            self.target_fundfreq = gr.Image(label='生成音频基频图')
                            self.target_audio = gr.Audio(label='生成音频')
            
            
            self.btn.click(pitch_shift, 
                            inputs=[self.input_file, self.functype, self.n_steps, self.bins_per_octave, self.sr], 
                            outputs=[self.origin_fundfreq, self.origin_audio, self.target_fundfreq, self.target_audio])
            
    def launch(self):
        self.demo.launch()


if __name__ == '__main__':
    demo = PitchShifter()
    demo.launch()




