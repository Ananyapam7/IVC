import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

import os
import cv2
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix as CM
from IPython.display import SVG

from tensorflow import keras
model = keras.models.load_model('motif-classifier.h5')

import random as rn
from random import randint
import itertools

import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
np.random.seed(123)
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix

Images_test, Labels_test = get_images('../data/test/')

Images_test = np.array(Images_test)
Labels_test = np.array(Labels_test)
Labels_test = to_categorical(Labels_test, 21)

Images_test = Images_test.astype('float32')/255

Labels_pred = model.predict(Images_test)

Y_pred_classes = np.argmax(Labels_pred,axis = 1)

Y_true = np.argmax(Labels_test, axis = 1)

from sklearn.metrics import confusion_matrix
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes)

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

plot_confusion_matrix(confusion_mtx, classes = range(21))

label_frac_error = 1 - np.diag(confusion_mtx) / np.sum(confusion_mtx, axis=1)
plt.bar(np.arange(21), label_frac_error)
plt.xlabel('True Label')
plt.ylabel('Fraction Classified Incorrectly')
