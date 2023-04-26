from glissando.glissando import glissando
from toolkits.amplitude_patch import amplitude_patch
import librosa
import numpy as np
import soundfile as sf
from toolkits.utils import spectrum2wav
import matplotlib.pyplot as plt

audio_file = 'audios/duan_singing.wav'
reference = 'audios/huaqiang1.wav'

audio_data, sr = librosa.load(audio_file, sr=44100)
reference, sr = librosa.load(reference, sr=44100)
result_data , shift_r, shift = glissando(audio_data, reference, sr=44100)

audio_stft = librosa.stft(audio_data)
S_p = np.angle(audio_stft)
result_stft = librosa.stft(result_data)

amp_patch = amplitude_patch(result_stft, audio_stft)

result_spectrum = librosa.amplitude_to_db(np.abs(result_stft) + amp_patch)

# result_spectrum = librosa.amplitude_to_db(np.abs(result_stft))
wave = spectrum2wav(result_spectrum, p=S_p)
sf.write('generate1.wav', wave, sr)
# import librosa
# import numpy as np

# # audio_file = 'audios/duan_singing.wav'
# # audio_data, sr = librosa.load(audio_file, sr=44100)
# # rate = 2 ** (-40 / 12)
# # y = librosa.effects.time_stretch(audio_data, rate=rate)

# # print("origin shape:", audio_data.shape, "rate:", rate, "after shape:", y.shape)
# # print(audio_data.shape[0] * 1/ rate)

# data = np.array([1., 2., 3., 4., 5., 6., 7., 8., 9., 10])
# y = librosa.effects.time_stretch(data, rate=2)
# print(y)
plt.plot(shift_r)
plt.plot(shift)
plt.show()