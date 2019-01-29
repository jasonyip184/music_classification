import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import librosa
import librosa.display

from flask import Flask, render_template, request
from flask_restful import reqparse, abort, Api, Resource
from keras.preprocessing import image
from sklearn.model_selection import train_test_split
from keras.models import load_model

# init
app = Flask(__name__)

# load model
model = load_model('model.h5')
model.load_weights("weights.hdf5")

@app.route('/uploader', methods = ['GET', 'POST'])
def uploader():
    if request.method == 'POST':
        # get song and convert to image
        y, sr = librosa.load("../Frontend/MusickApp/music.mp3")
        Y = librosa.stft(y)
        Ydb = librosa.amplitude_to_db(abs(Y))
        plt.figure(figsize=(6, 2))
        librosa.display.specshow(Ydb, sr=sr, cmap='gray_r')
        # image for model
        plt.savefig('gray.png')
        plt.close("all")
        # image to display
        librosa.display.specshow(Ydb, sr=sr)
        plt.savefig('../Frontend/MusickApp/colour.png')

        # image to array
        labels = []
        data = []
        instance = image.load_img('gray.png', target_size=(200, 600), color_mode="grayscale")
        array = image.img_to_array(instance)
        data.append(array)
        data = np.array(data, dtype="float") / 255.0

        # predict
        probs = model.predict(data)
        genres = ['classical','country','electronic','hiphop','rock']
        genre = genres[np.argmax(probs)]
        # genre to txt file
        text_file = open("../Frontend/MusickApp/genre.txt", "w")
        text_file.write(genre)
        text_file.close()
    else:
        return "File not received"

if __name__ == '__main__':
    app.run(debug=True)