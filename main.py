import librosa
import gradio as gr
from vibrato.vibrato import vibrato_sine_api, vibrato_uniformsine_api, vibrato_gausesine_api, vibrato_decrease_f_sine_api, vibrato_increase_f_sine_api
from glissando.glissando import glissando_api
from toolkits.equalizer import Equalizer
from toolkits.pitch_shifter import PitchShifter
from toolkits.overtone import Overtone

# 颤音模块
class VibratoDemo:
    def __init__(self):
        # 输入数据选项卡
        with gr.Blocks() as self.input_ui:
            with gr.Row():
                with gr.Column():
                    self.input_ui_file = gr.File(label='音频文件')
                    self.input_ui_btn = gr.Button(value='加载音频')
                self.input_ui_waveform = gr.Video(label='音频波形')

            self.input_ui_btn.click(self._input_file, inputs=[self.input_ui_file], outputs=[self.input_ui_waveform])

        # 标准的正弦波颤音选项卡
        with gr.Blocks() as self.sin_ui:
            with gr.Row():
                self.sin_ui_A = gr.Number(value=1, label='A(正弦波振幅)')
                self.sin_ui_w = gr.Number(value=1, label='w(正弦波频率)')
                self.sin_ui_b = gr.Number(value=0, label='b(偏置)')
                self.sin_ui_T = gr.Number(value=8, label='T(振动的周期数)')
            self.sine_ui_btn = gr.Button(value='颤音化')

            with gr.Row():
                with gr.Column():
                    self.sin_ui_origin_fundfreq = gr.Image(label='原音频基频图')
                    self.sin_ui_origin_audio = gr.Audio(label='原音频')
                with gr.Column():
                    self.sin_ui_result_fundfreq = gr.Image(label='颤音化音频基频图')
                    self.sin_ui_result_audio = gr.Audio(label='颤音化音频')

            self.sine_ui_btn.click(vibrato_sine_api,
                                    inputs=[self.input_ui_file, self.sin_ui_A, self.sin_ui_w, self.sin_ui_b, self.sin_ui_T],
                                    outputs=[self.sin_ui_origin_fundfreq, self.sin_ui_origin_audio, self.sin_ui_result_fundfreq, self.sin_ui_result_audio])

        # 频率均匀分布的正弦波颤音选项卡
        with gr.Blocks() as self.uniform_sin_ui:
            with gr.Row():
                self.uniform_sin_ui_A = gr.Number(value=1, label='A(正弦波振幅)')
                self.uniform_sin_ui_low = gr.Number(value=100, label='频率下限')
                self.uniform_sin_ui_high = gr.Number(value=300, label='频率上限')
            self.uniform_sin_ui_btn = gr.Button(value='颤音化')

            with gr.Row():
                with gr.Column():
                    self.uniform_sin_ui_origin_fundfreq = gr.Image(label='原音频基频图')
                    self.uniform_sin_ui_origin_audio = gr.Audio(label='原音频')
                with gr.Column():
                    self.uniform_sin_ui_result_fundfreq = gr.Image(label='颤音化音频基频图')
                    self.uniform_sin_ui_result_audio = gr.Audio(label='颤音化音频')

            self.uniform_sin_ui_btn.click(vibrato_uniformsine_api,
                                        inputs=[self.input_ui_file, self.uniform_sin_ui_A, self.uniform_sin_ui_low, self.uniform_sin_ui_high],
                                        outputs=[self.uniform_sin_ui_origin_fundfreq, self.uniform_sin_ui_origin_audio, self.uniform_sin_ui_result_fundfreq, self.uniform_sin_ui_result_audio])
        
        # 频率高斯分布的正弦波颤音选项卡
        with gr.Blocks() as self.gause_sin_ui:
            with gr.Row():
                self.gause_sin_ui_A = gr.Number(value=1, label='A(正弦波振幅)')
                self.gause_sin_ui_mean = gr.Number(value=0, label='频率均值')
                self.gause_sin_ui_square = gr.Number(value=1, label='频率方差')
            self.gause_sin_ui_btn = gr.Button(value='颤音化')

            with gr.Row():
                with gr.Column():
                    self.gause_sin_ui_origin_fundfreq = gr.Image(label='原音频基频图')
                    self.gause_sin_ui_origin_audio = gr.Audio(label='原音频')
                with gr.Column():
                    self.gause_sin_ui_result_fundfreq = gr.Image(label='颤音化音频基频图')
                    self.gause_sin_ui_result_audio = gr.Audio(label='颤音化音频')

            self.gause_sin_ui_btn.click(vibrato_gausesine_api,
                                        inputs=[self.input_ui_file, self.gause_sin_ui_A, self.gause_sin_ui_mean, self.gause_sin_ui_square],
                                        outputs=[self.gause_sin_ui_origin_fundfreq, self.gause_sin_ui_origin_audio, self.gause_sin_ui_result_fundfreq, self.gause_sin_ui_result_audio])
        
        # 频率降低的正弦波颤音选项卡
        with gr.Blocks() as self.decrease_sin_ui:
            with gr.Row():
                self.decrease_sin_ui_A = gr.Number(value=1, label='A(正弦波振幅)')
                self.decrease_sin_ui_p = gr.Slider(minimum=1, maximum=10, value=1, step=0.1, label='频率降低速率')
            self.decrease_sin_ui_btn = gr.Button(value='颤音化')

            with gr.Row():
                with gr.Column():
                    self.decrease_sin_ui_origin_fundfreq = gr.Image(label='原音频基频图')
                    self.decrease_sin_ui_origin_audio = gr.Audio(label='原音频')
                with gr.Column():
                    self.decrease_sin_ui_result_fundfreq = gr.Image(label='颤音化音频基频图')
                    self.decrease_sin_ui_result_audio = gr.Audio(label='颤音化音频')

            self.decrease_sin_ui_btn.click(vibrato_decrease_f_sine_api,
                                        inputs=[self.input_ui_file, self.decrease_sin_ui_A, self.decrease_sin_ui_p],
                                        outputs=[self.decrease_sin_ui_origin_fundfreq, self.decrease_sin_ui_origin_audio, self.decrease_sin_ui_result_fundfreq, self.decrease_sin_ui_result_audio])

        # 频率升高的正弦波颤音选项卡
        with gr.Blocks() as self.increase_sin_ui:
            with gr.Row():
                self.increase_sin_ui_A = gr.Number(value=1, label='A(正弦波振幅)')
                self.increase_sin_ui_p = gr.Slider(minimum=1, maximum=10, value=1, step=0.1, label='频率升高速率')
            self.increase_sin_ui_btn = gr.Button(value='颤音化')

            with gr.Row():
                with gr.Column():
                    self.increase_sin_ui_origin_fundfreq = gr.Image(label='原音频基频图')
                    self.increase_sin_ui_origin_audio = gr.Audio(label='原音频')
                with gr.Column():
                    self.increase_sin_ui_result_fundfreq = gr.Image(label='颤音化音频基频图')
                    self.increase_sin_ui_result_audio = gr.Audio(label='颤音化音频')

            self.increase_sin_ui_btn.click(vibrato_increase_f_sine_api,
                                        inputs=[self.input_ui_file, self.increase_sin_ui_A, self.increase_sin_ui_p],
                                        outputs=[self.increase_sin_ui_origin_fundfreq, self.increase_sin_ui_origin_audio, self.increase_sin_ui_result_fundfreq, self.increase_sin_ui_result_audio])

        # 组合选项卡
        self.vibrato_demo = gr.TabbedInterface(
            [self.input_ui ,self.sin_ui, self.uniform_sin_ui, self.gause_sin_ui, self.decrease_sin_ui, self.increase_sin_ui],
            ["输入", "标准正弦波", '频率均匀分布的正弦波', '频率高斯分布的正弦波', '频率降低的正弦波', '频率升高的正弦波']
        )

    def _input_file(self, filename):
        """参数
            filename: gradio 接收的文件名
        """
        self.input_filename = filename.name
        waveform_video = gr.make_waveform(self.input_filename)
        return waveform_video

    def launch(self):
        self.vibrato_demo.launch()

# 滑音模块
class GlissandoDemo:
    def __init__(self):
        # 文件输入 UI
        with gr.Blocks() as self.input_ui:
            with gr.Row():
                with gr.Column():
                    self.input_ui_process_file = gr.File(label='要处理的音频文件')
                    self.input_ui_reference_file = gr.File(label='参考滑音音频文件')
                    self.input_ui_btn = gr.Button(value='加载音频')
                with gr.Column():
                    self.input_ui_process_waveform = gr.Video(label='要处理音频文件波形')
                    self.input_ui_reference_waveform = gr.Video(label='参考滑音音频文件波形')

            self.input_ui_btn.click(self._input_file, 
                                    inputs=[self.input_ui_process_file, self.input_ui_reference_file],
                                    outputs=[self.input_ui_process_waveform, self.input_ui_reference_waveform])

        # 滑音处理 UI
        with gr.Blocks() as self.glissando_ui:
            with gr.Row():
                self.glissando_ui_process_type = gr.Dropdown(choices=['时域', '频域'], value='时域', label='处理类型')
            
            self.glissando_ui_btn = gr.Button(value='滑音化')

            with gr.Row():
                with gr.Column():
                    self.glissando_ui_process_fundfreq = gr.Image(label='处理音频基频图')
                    self.glissando_ui_process_audio = gr.Audio(label='处理音频')
                with gr.Column():
                    self.glissando_ui_reference_fundfreq = gr.Image(label='参考滑音片段基频图')
                    self.glissando_ui_reference_audio = gr.Audio(label='参考滑音片段')
                with gr.Column():
                    self.glissando_ui_result_fundfreq = gr.Image(label='结果音频基频图')
                    self.glissando_ui_result_audio = gr.Audio(label='结果音频')
            
            self.glissando_ui_btn.click(glissando_api,
                                        inputs=[self.input_ui_process_file, self.input_ui_reference_file, self.glissando_ui_process_type],
                                        outputs=[self.glissando_ui_process_fundfreq, self.glissando_ui_process_audio, 
                                                self.glissando_ui_reference_fundfreq, self.glissando_ui_reference_audio,
                                                self.glissando_ui_result_fundfreq, self.glissando_ui_result_audio])

        self.glissando_demo = gr.TabbedInterface(
            [self.input_ui, self.glissando_ui],
            ['输入', '滑音化']
        )
            

    def _input_file(self, process_filename, reference_filename):
        """参数
            process_filename: gradio 接收的要处理的文件名
            reference_filename: gradio 接收的参考滑音片段的文件名
        """
        self.process_filename = process_filename.name
        self.reference_filename = reference_filename.name
        process_waveform_video = gr.make_waveform(self.process_filename)
        reference_waveform_video = gr.make_waveform(self.reference_filename)
        return process_waveform_video, reference_waveform_video

    def launch(self):
        self.glissando_demo.launch()

# 工具模块
class ToolKitsDemo:
    def __init__(self):
        self.equalizer = Equalizer()
        self.pitch_shifter = PitchShifter()
        self.overtone = Overtone()

        self.toolkits_demo = gr.TabbedInterface(
            [self.equalizer.demo, self.pitch_shifter.demo, self.overtone.demo],
            ['均衡器', '音高调整器', '基频泛音组合器']
        )

    def launch(self):
        self.toolkits_demo.launch()


if __name__ == '__main__':
    demo = gr.TabbedInterface(
        [VibratoDemo().vibrato_demo, GlissandoDemo().glissando_demo, ToolKitsDemo().toolkits_demo],
        ['颤音', '滑音', '工具']
    )

    demo.launch()


    

