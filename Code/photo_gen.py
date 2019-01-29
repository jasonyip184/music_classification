import librosa
import os
import matplotlib.pyplot as plt
import librosa.display
import numpy as np

genres = ['classical','country','electronic','hiphop','rock']

for genre in genres:
    dire = "C:/Users/jasonyip184/Desktop/music/Data/Songs/"+genre
    songs = os.listdir(dire)
    for i in range(0, len(songs)):
        path = dire+'/'+songs[i]
        try:
            y, sr = librosa.load(path)
            Y = librosa.stft(y)
            Ydb = librosa.amplitude_to_db(abs(Y))
            plt.figure(figsize=(6, 2))
            librosa.display.specshow(Ydb, sr=sr, cmap='gray_r')
            img_path = "C:/Users/jasonyip184/Desktop/music/Data/Photos/"+genre+"_photos/"+songs[i][:-4]+'.png'
            plt.savefig(img_path)   # save the figure to file
            plt.close("all")
        except:
            print(songs[i])

