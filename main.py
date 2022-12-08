import cv2
import keras
from tensorflow import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dropout
import glob
import numpy as np


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
    return img


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

# Set the directories for the train, validation, and test sets
train_dir = "tiny-imagenet-200/train"
val_dir = "tiny-imagenet-200/val"
test_dir = "tiny-imagenet-200/test"

# Use glob to find all images in the train, validation, and test sets
train_images = glob.glob("{}/*/*.JPEG".format(train_dir))
val_images = glob.glob("{}/*/*.JPEG".format(val_dir))
test_images = glob.glob("{}/*.JPEG".format(test_dir))

# Load the images into memory
train_imgs = [cv2.imread(img) for img in train_images]
val_imgs = [cv2.imread(img) for img in val_images]
test_imgs = [cv2.imread(img) for img in test_images]

# Store the images in a list or array for easy access
train_data = np.array(train_imgs)
val_data = np.array(val_imgs)
test_data = np.array(test_imgs)

print(len(train_imgs))





