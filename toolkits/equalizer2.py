import librosa
import numpy as np
import gradio as gr
import sys
from .utils import spectrum2wav, hz_to_fft
sys.path.append('..')
from features.spectrum import save_spectrum_figure, spectrum
import soundfile as sf

# 频段
FREQ_BAND = {
    '32': [0, 31],
    '64': [32, 63],
    '125': [64, 124],
    '250': [125, 249],
    '500': [250, 499],
    '1k': [500, 999],
    '2k': [1000, 1999],
    '4k': [2000, 3999],
    '8k': [4000, 7999],
    '16k': [8000, 15999]
}

# 对分贝为元素的频谱图指定的频段进行分贝增益
def S_db_gain(S, freq_band, gain_value=0):
    """参数
        S: 以分贝作为元素的频谱图
        freq_band: 要进行增益的频段
        gain_value: 增益的大小(dB)
        -------------------------------
        return: 进行分贝增益后的频谱图 
    """
    if freq_band not in FREQ_BAND.keys():
        raise NotImplementedError('unknown freq_band of {}'.format(freq_band))
    
    hz_start, hz_end = FREQ_BAND[freq_band]
    fft_i_start = hz_to_fft(hz_start)
    fft_i_end = hz_to_fft(hz_end)

    S[fft_i_start:fft_i_end, :] += gain_value

    return S

# 均衡器函数
def equalizer(filename, freq_band_32, freq_band_64, freq_band_125, freq_band_250, freq_band_500, 
            freq_band_1k, freq_band_2k, freq_band_4k, freq_band_8k, freq_band_16k):
    """参数
        filename: gradio 接收的文件名
        freq_band_xx: 要进行处理的频段的分贝增益
        -------------------------------
        return: (原始音频的频谱图路径, 原始音频路径, 生成音频的频谱图, 生成音频路径)
    """
    audio_data, sr = librosa.load(filename.name, sr=44100)
    origin_spectrum_path = save_spectrum_figure(audio_data, sr=sr, save_filename='gradio_temp/equalizer_origin_spectrum.jpg')

    audio_S_complex = spectrum(audio_data)
    audio_S_db = librosa.amplitude_to_db(np.abs(audio_S_complex))   # 原音频分贝谱图
    audio_S_p = np.angle(audio_S_complex)                           # 原音频相位

    freq_bands = {
        '32': freq_band_32,
        '64': freq_band_64,
        '125': freq_band_125,
        '250': freq_band_250,
        '500': freq_band_500,
        '1k': freq_band_1k,
        '2k': freq_band_2k,
        '4k': freq_band_4k,
        '8k': freq_band_8k,
        '16k': freq_band_16k
    }

    target_S_db = audio_S_db
    for freq_band in freq_bands.keys():
        if freq_bands[freq_band] != 0:
            target_S_db = S_db_gain(target_S_db, freq_band, freq_bands[freq_band])


    taregt_audio_data = spectrum2wav(target_S_db, p=audio_S_p)            # 用新的分贝谱图和原相位重构目标音频
    target_audio_path = 'gradio_temp/equalizer_target_audio.wav'
    sf.write(target_audio_path, taregt_audio_data, sr)
    target_spectrum_path = save_spectrum_figure(taregt_audio_data, sr=sr, save_filename='gradio_temp/equalizer_target_spectrum.jpg')

    return origin_spectrum_path, filename.name, target_spectrum_path, target_audio_path

# 均衡器类
class Equalizer:
    def __init__(self, func=equalizer):
        """参数
            func: 槽函数
        """
        self.func = func

        with gr.Blocks() as self.demo:
            self.input_file = gr.File(label='音频文件')
            with gr.Blocks():
                with gr.Row():
                    self.slider_32 = gr.Slider(minimum=-12, maximum=12, value=0, step=1, label='0-32(Hz)')
                    self.slider_64 = gr.Slider(minimum=-12, maximum=12, value=0, step=1, label='32-64(Hz)')
                    self.slider_125 = gr.Slider(minimum=-12, maximum=12, value=0, step=1, label='64-125(Hz)')
                    self.slider_250 = gr.Slider(minimum=-12, maximum=12, value=0, step=1, label='125-250(Hz)')
                    self.slider_500 = gr.Slider(minimum=-12, maximum=12, value=0, step=1, label='250-500(Hz)')
                with gr.Row():
                    self.slider_1k = gr.Slider(minimum=-12, maximum=12, value=0, step=1, label='500-1k(Hz)')
                    self.slider_2k = gr.Slider(minimum=-12, maximum=12, value=0, step=1, label='1k-2k(Hz)')
                    self.slider_4k = gr.Slider(minimum=-12, maximum=12, value=0, step=1, label='2k-4k(Hz)')
                    self.slider_8k = gr.Slider(minimum=-12, maximum=12, value=0, step=1, label='4k-8k(Hz)')
                    self.slider_16k = gr.Slider(minimum=-12, maximum=12, value=0, step=1, label='8k-16k(Hz)')

            self.btn = gr.Button(value='计算')

            with gr.Blocks():
                with gr.Row():
                    with gr.Blocks():
                        with gr.Column():
                            self.origin_spectrum = gr.Image(label='原音频频谱图')
                            self.origin_audio = gr.Audio(label='原音频')
                    with gr.Blocks():
                        with gr.Column():
                            self.target_spectrum = gr.Image(label='生成音频频谱图图')
                            self.target_audio = gr.Audio(label='生成音频')
            
            self.btn.click(self.func, 
                            inputs=[self.input_file, self.slider_32, self.slider_64, self.slider_125, self.slider_250, self.slider_500, 
                                    self.slider_1k, self.slider_2k, self.slider_4k, self.slider_8k, self.slider_16k], 
                            outputs=[self.origin_spectrum, self.origin_audio, self.target_spectrum, self.target_audio])

    def launch(self):
        self.demo.launch()


if __name__ == '__main__':
    demo = Equalizer()
    demo.launch()





