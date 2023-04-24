import librosa
import numpy as np
import soundfile as sf
import sys
sys.path.append('..')
from features.fundfreq import save_fundfreq_figure
from features.spectrum import spectrum
import gradio as gr
from utils import spectrum2wav
from amplitude_patch import amplitude_patch

# 音高调整的槽函数
def pitch_shifter(filename, n_steps=0, bins_per_octave=12, sr=44100):
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
    target_audio_path = 'gradio_temp/pitch_shifted_audio.wav'
    origin_fundfreq_path = 'gradio_temp/pitch_shift_origin_fundfreq.jpg'
    target_fundfreq_path = 'gradio_temp/pitch_shifted_fundfreq.jpg'

    origin_fundfreq_path = save_fundfreq_figure(audio_data, sr=sr, save_filename=origin_fundfreq_path)

    target = librosa.effects.pitch_shift(audio_data, sr=sr, n_steps=n_steps, bins_per_octave=bins_per_octave)
    sf.write(target_audio_path, target, sr)
    target_fundfreq_path = save_fundfreq_figure(target, sr=sr, save_filename=target_fundfreq_path)

    return origin_fundfreq_path, origin_audio_path, target_fundfreq_path, target_audio_path

# 音高调整器类
class PitchShifter:
    def __init__(self, func):
        """参数
            func: 槽函数
        """
        self.func = func

        with gr.Blocks() as self.demo:
            with gr.Blocks():
                with gr.Row():
                    self.input_file = gr.File(label='音频文件')
                    with gr.Column():
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
            
            self.btn.click(self.func, 
                            inputs=[self.input_file, self.n_steps, self.bins_per_octave, self.sr], 
                            outputs=[self.origin_fundfreq, self.origin_audio, self.target_fundfreq, self.target_audio])

    def launch(self):
        self.demo.launch()


if __name__ == '__main__':
    demo = PitchShifter(pitch_shifter)
    demo.launch()




