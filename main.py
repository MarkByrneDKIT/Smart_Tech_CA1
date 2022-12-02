import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras
from keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
import random
import requests
from PIL import Image
import cv2
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
import pickle
import pandas as pd
import cv2

def gray_scale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def equalise(img):
    img = cv2.equalizeHist(img)
    return img


def preprocess(img):
    img = gray_scale(img)
    img = equalise(img)
    img = img/255
    return


#def le_net_model():
#    model = Sequential()
#    model.add(Conv2D(60, (5, 5), input_shape=(32, 32, 1), activation='relu'))
#    model.add(MaxPooling2D(pool_size=(2, 2)))
#    model.add(Conv2D(30, (3, 3), activation='relu'))
#    model.add(MaxPooling2D(pool_size=(2, 2)))
#    model.add(Flatten())
#    model.add(Dense(500, activation='relu'))
#    model.add(Dropout(0.5))
#    model.add(Dense(num_classes, activation='softmax'))
#    model.compile(Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
#    return model

#open words.txt
with open('tiny-imagenet-200/words.txt') as f:
    for line in f:
        print(line.strip())


# open and get images
test_data = []
train_data = []
val_data = []





# xtrain, xtest, xval + labels
#X_train, y_train = train_data['features'], train_data['labels']
# X_val, y_val = val_data['features'], val_data['labels']
# X_test, y_test = test_data['features'], test_data['labels']

# assert xtrain, xtest, xval




# one hot encode labels

# score

