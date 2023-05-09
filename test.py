import numpy as np
import librosa
import matplotlib.pyplot as plt
from features.fundfreq import fundfreq
from glissando.utils import pitch_shift_steps, shift_sampling
import matplotlib
matplotlib.use('tkagg')

audio_data, sr = librosa.load('audios/huaqiang1.wav', sr=44100)
f0, _, _ = fundfreq(audio_data)
n_steps = pitch_shift_steps(f0)
# 截断
n_steps = np.clip(n_steps, -6, 6)
# print(n_steps.shape)
res = shift_sampling(n_steps, 200)
# # # 平滑化
# window_size = 5
# # 计算加权平均数
# weights = np.repeat(1.0, window_size) / window_size
# shift = np.convolve(res, weights, 'same')

plt.plot(res, label='origin')

plt.xlabel('frame')
plt.ylabel('semitones')
plt.show()














