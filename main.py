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
test_images = glob.glob("{}/*/*.JPEG".format(test_dir))

# Load the images into memory
train_imgs = [cv2.imread(img) for img in train_images]
val_imgs = [cv2.imread(img) for img in val_images]
test_imgs = [cv2.imread(img) for img in test_images]

print("num of train images", len(train_imgs))
print("num of val images", len(val_imgs))
print("num of test images", len(test_imgs))


# Read the labels from the words.txt file
with open("tiny-imagenet-200/words.txt") as f:
    contents = f.read()
    #print(contents)
    labels = {line.split("\t")[0]: line.split("\t")[1] for line in contents}

# Store the images and labels in a list or array
train_data = [(img, labels[img.split("/")[-1]]) for img in train_imgs]
val_data = [(img, labels[img.split("/")[-1]]) for img in val_imgs]
test_data = [(img, labels[img.split("/")[-1]]) for img in test_imgs]









