import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.callbacks import ModelCheckpoint

# load data and labels
h5f = h5py.File('data.h5','r')
data = h5f['data'][:]
h5f.close()
h5f = h5py.File('labels.h5','r')
labels = h5f['labels'][:]
h5f.close()

# train test val 80/10/10
X_train, X_test, Y_train, Y_test = train_test_split(data, labels, test_size=0.1, random_state=1)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=1/9, random_state=1)
# print(Y_train)
# print(Y_train.shape)
# model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(200,600,1)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(5, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

batch_size = 16

# checkpoint
filepath="weights.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

model.save('model.h5')

history = model.fit(X_train, Y_train,
          epochs=3,
          batch_size=batch_size,
          validation_data=(X_val, Y_val),
          callbacks=callbacks_list,
          verbose=2)