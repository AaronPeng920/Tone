import librosa
import gradio as gr
from vibrato.vibrato import vibrato_api

# 颤音化类
class VibratoDemo:
    def __init__(self, api):
        """参数
            api: 用于颤音功能的 API
        """
        self.api = api

        with gr.Blocks() as self.demo:
            with gr.Row():
                self.input_file = gr.File(label='输入音频文件')
                
                with gr.Column():
                    self.flucttype = gr.Dropdown(choices=['标准的正弦波', '频率均匀分布的正弦波', '频率高斯分布的正弦波', '频率升高的正弦波', '频率降低的正弦波'], value='标准的正弦波', label='波动类型')

                    with gr.Row():
                        self.A = gr.Number(value=1, label='A(正弦波振幅)')
                        self.w = gr.Number(value=1, label='w(正弦波频率)')
                        self.b = gr.Number(value=0, label='b(偏置)')
                        self.T = gr.Number(value=10, label='T(振动的周期数)')
                    with gr.Row():
                        self.low = gr.Number(value=100, label='频率下限(Hz)')
                        self.high = gr.Number(value=300, label='频率上限(Hz)')
                    with gr.Row():
                        self.mean = gr.Number(value=0, label='波动频率均值(Hz)')
                        self.square = gr.Number(value=1, label='频率波动方差')
                    with gr.Row():
                        self.p = gr.Slider(minimum=1, maximum=10, value=1, step=1, label='波动程度')

            self.btn = gr.Button(value='颤音化')

            with gr.Row():
                with gr.Column():
                    self.origin_fundfreq = gr.Image(label='原音频基频图')
                    self.origin_audio = gr.Audio(label='原音频')
                with gr.Column():
                    self.result_fundfreq = gr.Image(label='颤音化音频基频图')
                    self.result_audio = gr.Audio(label='颤音化音频')

            self.btn.click(self.api,
                            inputs = [self.input_file, self.flucttype, self.A, self.w, self.b, self.T, self.low, self.high, self.mean, self.square, self.p],
                            outputs = [self.origin_fundfreq, self.origin_audio, self.result_fundfreq, self.result_audio])
    
    def launch(self):
        self.demo.launch()
        


if __name__ == '__main__':
    demo = VibratoDemo(vibrato_api)
    demo.launch()


