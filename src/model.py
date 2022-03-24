import gc
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.keras.layers as Layers
import tensorflow.keras.activations as Actications
import tensorflow.keras.models as Models
import tensorflow.keras.optimizers as Optimizer
import tensorflow.keras.metrics as Metrics
import tensorflow.keras.utils as Utils
from keras.utils.vis_utils import model_to_dot
import os
import matplotlib.pyplot as plot
import cv2
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix as CM
from IPython.display import SVG

from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from keras.layers import Dropout, Flatten,Activation
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
import random as rn
from random import randint
import itertools

%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
np.random.seed(123)
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix

def get_images(directory):
    Images = []
    Labels = []
    label = 0

    for labels in os.listdir(directory): #Main Directory where each class label is present as folder name.
        if labels == 'boat': #Folder contain Glacier Images get the '2' class label.
            label = 0
        elif labels == 'buffalo':
            label = 1
        elif labels == 'bull':
            label = 2
        elif labels == 'centaur':
            label = 3
        elif labels == 'elephant':
            label = 4
        elif labels == 'horned_ram':
            label = 5
        elif labels == 'horned_zebra':
            label = 6
        elif labels == 'man_holding_tigers':
            label = 7
        elif labels == 'pashupati':
            label = 8
        elif labels == 'rhino':
            label = 9
        elif labels == 'sharp_horn_and_long_trunk':
            label = 10
        elif labels == 'short_horned_bull_with_head_lowered_toward_a_trough':
            label = 11
        elif labels == 'swastik':
            label = 12
        elif labels == 'three-headed_horned_ram':
            label = 13
        elif labels == 'tiger':
            label = 14
        elif labels == 'tiger_looking_at_man_on_tree':
            label = 15
        elif labels == 'two-headed':
            label = 16
        elif labels == 'unicorn':
            label = 17
        elif labels == 'unicorn_emerging_from_a_star-shaped_object':
            label = 18
        elif labels == 'unicorns_emerging_from_a_tree_trunk':
            label = 19
        elif labels == 'wavy_horned':
            label = 20

        for image_file in os.listdir(directory+labels): #Extracting the file name of the image from Class Label folder
            image = cv2.imread(directory+labels+r'/'+image_file) #Reading the image (OpenCV)
            Images.append(image)
            Labels.append(label)

    return shuffle(Images,Labels,random_state=77) #Shuffle the dataset

Images, Labels = get_images('../data/train/')

Images = np.array(Images)

Labels = np.array(Labels)

Images = Images.astype('float32')/255

Labels = to_categorical(Labels, 21)

(x_train, x_valid) = Images[:-50], Images[-50:]
(y_train, y_valid) = Labels[:-50], Labels[-50:]

del Labels
del Images

### Model

model = Sequential()
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same',activation ='relu', input_shape = (200,200,3)))
model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',activation ='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))


model.add(Conv2D(filters =96, kernel_size = (3,3),padding = 'Same',activation ='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

model.add(Conv2D(filters = 96, kernel_size = (3,3),padding = 'Same',activation ='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(21, activation = "softmax"))

### Compile Model

model.compile(loss='categorical_crossentropy',
             optimizer='adam',
             metrics=['accuracy'])
# model.summary()

history = model.fit(x_train, y_train, batch_size=25, epochs=30, validation_data=(x_valid, y_valid))

## Visualization

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([-1,1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([-1,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.save('Accuracy and Loss during training')

model.save('motif_classifier.h5')
