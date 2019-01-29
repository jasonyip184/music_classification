import numpy as np
import os
import matplotlib.pyplot as plt
import h5py
from keras.preprocessing import image
from sklearn.preprocessing import LabelBinarizer

directory = "C:/Users/jasonyip184/Desktop/music/Data/Photos/"
photo_directories = os.listdir(directory)

# X
data = []
# Y
labels = []
labelid = 0

# go thru every category
for photo_directory in photo_directories:
    photos = os.listdir(directory+photo_directory)
    print(photo_directory)
    # every photo in category
    for photo in photos:
        print(photo)
        path = directory+photo_directory+"/"+photo
        instance = image.load_img(path, target_size=(200, 600), color_mode="grayscale")
        # array = image.img_to_array(instance)[:,:,0]
        array = image.img_to_array(instance)
        data.append(array)
        labels.append(labelid)
    # assign diff id if in diff category
    labelid += 1
        
data = np.array(data, dtype="float") / 255.0
# one-hot encode labels
encoder = LabelBinarizer()
labels = encoder.fit_transform(labels)
labels = np.array(labels)

# save data and labels
h5f = h5py.File('data.h5', 'w')
h5f.create_dataset('data', data=data)
h5f.close()
h5f = h5py.File('labels.h5', 'w')
h5f.create_dataset('labels', data=labels)
h5f.close()

# # load data and labels
# h5f = h5py.File('data.h5','r')
# data = h5f['data'][:]
# h5f.close()
# h5f = h5py.File('labels.h5','r')
# labels = h5f['labels'][:]
# h5f.close()


