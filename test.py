import pyrubberband
import librosa
import soundfile as sf


audio_data, sr = librosa.load('audios/duan_singing.wav', sr=44100)
y = pyrubberband.pyrb.pitch_shift(audio_data, sr, 4)
sf.write('tmp1.wav', y, sr)

