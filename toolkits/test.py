import librosa
import numpy as np
from utils import spectrum2wav
import sys
sys.path.append('..')
from features.spectrum import save_spectrum_figure
from features.fundfreq import fundfreq

audio_file = '/Users/aaronpeng/Desktop/Tone/audios/yi_hou_de_yi_hou_01_受.wav'
audio_data, sr = librosa.load(audio_file, sr=44100)
f0, _, _ = fundfreq(audio_data, sr=sr)


s = librosa.stft(audio_data, n_fft=2048)

S_p = np.angle(s)
S_db = librosa.amplitude_to_db(np.abs(s))

save_spectrum_figure(audio_data, sr=sr, save_filename='1.jpg')
y = spectrum2wav(S_db, p=None)
save_spectrum_figure(y, sr=sr, save_filename='2.jpg')



