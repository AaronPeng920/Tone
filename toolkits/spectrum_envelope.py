import librosa
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import gradio as gr
from utils import random_uuid_suffix
matplotlib.use('agg')

# 保存音频的指定帧的频谱包络
def save_spectrum_envelope(signal, frame_idx, sr=44100, save_filename=None):
    """参数
        signal: librosa 读取的音频信号
        frame_idx: 帧索引
        sr: 音频的采样率
        ---------------------------------
        return: 保存的频谱包络的图像的路径
    """
    S = np.abs(librosa.stft(signal))
    S_envelope = S[:,frame_idx]

    xticks = [i*sr/2048 for i in range(1025)]           # x 轴的刻度
    plt.cla()
    plt.clf()
    plt.plot(xticks, S_envelope)
    plt.xlabel("Frequence(Hz)")
    plt.ylabel("Amplitude")
    plt.title("Spectrum Envelope")

    if save_filename is None:
        save_filename = random_uuid_suffix('jpg')

    plt.savefig(save_filename, dpi=600)

    return save_filename

# 比较两个音频的相同帧的频谱包络
def spectrum_envelope(filename1, filename2, frame_idx, sr=44100):
    """参数
        filename1: gradio 传输的第一个音频片段的文件名
        filename2: gradio 传输的第二个音频片段的文件名
        frame_idx: 帧索引
        sr: 音频的采样率
        -------------------------------
        return: (第一个音频的指定的帧频谱包络图片路径, 第二个音频的指定的帧频谱包络图片路径)
    """
    frame_idx = int(frame_idx)
    
    signal1, sr = librosa.load(filename1.name, sr=sr)
    signal2, sr = librosa.load(filename2.name, sr=sr)

    envelope_path1 = 'gradio_temp/envelope_1.jpg'
    envelope_path2 = 'gradio_temp/envelope_2.jpg'

    path1 = save_spectrum_envelope(signal1, frame_idx=frame_idx, sr=sr, save_filename=envelope_path1)
    path2 = save_spectrum_envelope(signal2, frame_idx=frame_idx, sr=sr, save_filename=envelope_path2)

    return path1, path2


# 频谱包络
class SpectrumEnvelope:
    def __init__(self, func):
        """参数
            func: 槽函数
        """
        self.func = func

        with gr.Blocks() as self.demo:
            with gr.Row():
                with gr.Column():
                    self.input_file1 = gr.File(label='音频文件1')
                    self.input_file2 = gr.File(label='音频文件2')
                    self.frame_idx = gr.Number(label='帧索引', value=0)
                    self.btn = gr.Button(value='计算')
                with gr.Column():
                    self.envelope1 = gr.Image(label='音频文件1的频谱包络图')
                    self.envelope2 = gr.Image(label='音频文件2的频谱包络图')

            self.btn.click(self.func, 
                            inputs=[self.input_file1, self.input_file2, self.frame_idx], 
                            outputs=[self.envelope1, self.envelope2])

    def launch(self):
        self.demo.launch()


if __name__ == '__main__':
    demo = SpectrumEnvelope(spectrum_envelope)
    demo.launch()






    
    

    
