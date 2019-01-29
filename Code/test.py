import librosa
import os
import matplotlib.pyplot as plt
import librosa.display
import numpy as np

y, sr = librosa.load("sample/UltraCat_-_01_-_Orbiting_the_Earth.mp3")
Y = librosa.stft(y)
Ydb = librosa.amplitude_to_db(abs(Y))
plt.figure(figsize=(6, 2))
librosa.display.specshow(Ydb, sr=sr, cmap='magma')
img_path = "sample/UltraCat_-_01_-_Orbiting_the_Earth.png"
plt.savefig(img_path)   # save the figure to file
plt.close("all")