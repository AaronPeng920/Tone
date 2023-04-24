import librosa
import soundfile as sf
import numpy as np
import sys
sys.path.append('..')
import gradio as gr
from features.spectrum import save_spectrum_figure, spectrum
from features.fundfreq import fundfreq
from utils import hz_to_fft, spectrum2wav

# 蒙版过滤
def mask_fliter(arr, mask, pad_value=0):
    """参数
        arr: 待过滤的数组, shape:(a, b)
        mask: 蒙版, shape:(a, b)
        pad_value: 过滤掉(mask = False)的值的填充
        ------------------------
        return: 过滤后的数组
    """
    h, w = arr.shape
    for h_i in range(h):
        for w_i in range(w):
            if not mask[h_i][w_i]:
                arr[h_i][w_i] = pad_value

    return arr

# 保留基频泛音组合
def fundfreq_overtone_group(S_db, fundfreqs, overtone_group):
    """参数
        S_db: 分贝作为元素的频谱图, shape:(n_fft//2+1, 帧数量)
        fundfreqs: 基频, shape:(n_fft//2+1, 帧数量)
        overtone_group: list(int), 要保留的基频和泛音级别
        -------------------------------
        return: 保留指定的基频和泛音后的频谱图
    """
    fft_total = S_db.shape[0]
    frame_count = S_db.shape[1]

    mask = []    
    # 逐帧进行判定, 形成蒙版 mask
    for f in fundfreqs:
        if np.isnan(f):
            mask_x = [True] * fft_total
            mask.append(mask_x)
        else:
            freqs = [f * (i+1) for i in overtone_group]         # 所有的频率
            fft_i = [hz_to_fft(i, sr=44100) for i in freqs]     # 所有的频率对应的 fft 索引
            mask_x = [False] * fft_total
            for fft_i_i in fft_i:
                if fft_i_i >= 0 and fft_i_i < fft_total:
                    mask_x[fft_i_i] = True
            mask.append(mask_x)
    
    mask = np.array(mask, dtype=bool)           # shape:(帧数量, n_fft//2+1)
    mask = mask.T
    

    res = mask_fliter(S_db, mask, pad_value=-80)
    
    return res
    
# 保留原音频的基频泛音组合
def overtone(filename, l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l10,
            l11, l12, l13, l14, l15, l16, l17, l18, l19, sr=44100):
    """参数
        filename: gradio 接收的文件名
        lx: bool, 是否保持第 x 泛音
        sr: 采样率
        ------------------------------
        return: (原音频的频谱图路径, 原音频文件路径, 生成音频的频谱图路径, 生成音频文件路径)
    """
    audio_data, sr = librosa.load(filename.name, sr=sr)
    origin_spectrum_path = 'gradio_temp/overtone_origin_spectrum.jpg'
    origin_spectrum_path = save_spectrum_figure(audio_data, sr=44100, save_filename=origin_spectrum_path)
    origin_spectrum = spectrum(audio_data)
    origin_spectrum_db = librosa.amplitude_to_db(np.abs(origin_spectrum))
    origin_spectrum_p = np.angle(origin_spectrum)

    # 所有的泛音级别
    overtones = [l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l10,
                l11, l12, l13, l14, l15, l16, l17, l18, l19]
    overtone_group = []         # 存储要保持的泛音级别
    for idx, x in enumerate(overtones):
        if x:
            overtone_group.append(idx)
    

    f0, _, _ = fundfreq(audio_data, sr=sr)

    target_spectrum_db = fundfreq_overtone_group(origin_spectrum_db, f0, overtone_group)
    target_audio = spectrum2wav(target_spectrum_db, origin_spectrum_p)

    target_spectrum_path = 'gradio_temp/overtone_target_spectrum.jpg'
    target_spectrum_path = save_spectrum_figure(target_audio, sr=sr, save_filename=target_spectrum_path)
    target_audio_path = 'gradio_temp/overtone_target_audio.wav'
    sf.write(target_audio_path, target_audio, sr)

    return origin_spectrum_path, filename.name, target_spectrum_path, target_audio_path

# 泛音类
class Overtone:
    def __init__(self, func):
        """参数
            func: 槽函数
        """
        self.func = func

        with gr.Blocks() as self.demo:
            self.input_file = gr.File(label='音频文件')
            with gr.Blocks():
                with gr.Row():
                    self.l0 = gr.Checkbox(label='基频', value=True)
                    self.l1 = gr.Checkbox(label='第一泛音', value=False)
                    self.l2 = gr.Checkbox(label='第二泛音', value=False)
                    self.l3 = gr.Checkbox(label='第三泛音', value=False)
                    self.l4 = gr.Checkbox(label='第四泛音', value=False)
                with gr.Row():
                    self.l5 = gr.Checkbox(label='第五泛音', value=False)
                    self.l6 = gr.Checkbox(label='第六泛音', value=False)
                    self.l7 = gr.Checkbox(label='第七泛音', value=False)
                    self.l8 = gr.Checkbox(label='第八泛音', value=False)
                    self.l9 = gr.Checkbox(label='第九泛音', value=False)
                with gr.Row():
                    self.l10 = gr.Checkbox(label='第十泛音', value=False)
                    self.l11 = gr.Checkbox(label='第十一泛音', value=False)
                    self.l12 = gr.Checkbox(label='第十二泛音', value=False)
                    self.l13 = gr.Checkbox(label='第十三泛音', value=False)
                    self.l14 = gr.Checkbox(label='第十四泛音', value=False)
                with gr.Row():
                    self.l15 = gr.Checkbox(label='第十五泛音', value=False)
                    self.l16 = gr.Checkbox(label='第十六泛音', value=False)
                    self.l17 = gr.Checkbox(label='第十七泛音', value=False)
                    self.l18 = gr.Checkbox(label='第十八泛音', value=False)
                    self.l19 = gr.Checkbox(label='第十九泛音', value=False)

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
                            inputs=[self.input_file, self.l0, self.l1, self.l2, self.l3, self.l4, self.l5, self.l6, self.l7, self.l8, self.l9,
                                    self.l10, self.l11, self.l12, self.l13, self.l14, self.l15, self.l16, self.l17, self.l18, self.l19],
                            outputs=[self.origin_spectrum, self.origin_audio, self.target_spectrum, self.target_audio])

    def launch(self):
        self.demo.launch()


if __name__ == '__main__':
    demo = overtone(Overtone)
    demo.launch()






