import numpy as np
import librosa
import matplotlib.pyplot as plt
from features.fundfreq import fundfreq, save_fundfreq_figure
from features.spectrum import save_spectrum_figure
from glissando.utils import pitch_shift_steps, shift_sampling
import matplotlib
matplotlib.use('tkagg')

audio_data, sr = librosa.load('audios/huaqiang2.wav', sr=44100)
audio_data = audio_data[:-20000]
f0, _, _ = fundfreq(audio_data, sr=sr)

fft_i = np.array([int(f * 2048 / sr) if not np.isnan(f) else np.nan for f in f0])
f0_ = np.array([f * np.power(2, 1/12) ** 3 for f in f0])
fft_i_ = np.array([int(f * 2048 / sr) if not np.isnan(f) else np.nan for f in f0_])
fft_step = fft_i_ - fft_i
plt.plot(audio_data)
plt.show()




# f0, _, _ = fundfreq(audio_data)
# n_steps = pitch_shift_steps(f0)
# # 截断
# n_steps = np.clip(n_steps, -6, 6)
# # print(n_steps.shape)
# res = shift_sampling(n_steps, 200)
# # # 平滑化
# window_size = 5
# # 计算加权平均数
# weights = np.repeat(1.0, window_size) / window_size
# shift = np.convolve(res, weights, 'same')

# plt.plot(shift)

# plt.xlabel('frame')
# plt.ylabel('semitones')
# plt.show()












