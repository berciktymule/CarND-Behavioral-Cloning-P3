import tensorflow as tf
import csv
import cv2
import numpy as np
from random import randint

# Initial Setup for Keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten,  Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers import Cropping2D
from keras.layers.normalization import BatchNormalization

#intemediary array to store frame images
images = []
#intemediary array to store steering values
steering = []
#list of directories with data from different recording sessions
base_data_path = ['../corrections/', '../slow_data/', '../data_ud/']

#read the image and resize them to 64x64 for a smaller memory footprint
def readImage(path):
    return cv2.resize(cv2.imread(path), (64, 64))


def readFile(base_data_path):
    #read the csv with the data
    with open(base_data_path + 'driving_log.csv') as csv_file:
        reader = csv.reader(csv_file)
        #read the headers from the first line tto referencee the data column offsets
        headers = next(reader)
        for line in reader:
            steering_value = float(line[headers.index('steering')])
            images.append(readImage(base_data_path + line[headers.index('center')]))
            steering.append(steering_value)

            #mirror images to combat one side bias
            images.append(cv2.flip(readImage(base_data_path + line[headers.index('center')]),1))
            steering.append(-steering_value)

#read all the data from all the directories
[readFile(x) for x in base_data_path]
#convert the data and the labels to numpy arrays
X_train = np.array(images)
y_train = np.array(steering)

#Nvidia model here we go
model = Sequential()
model.add(Cropping2D(cropping=((30,9), (0,0)), input_shape=(64,64,3)))
model.add(Lambda(lambda x: x / 255.0 - 0.5))
model.add(Convolution2D(24, kernel_size=(5, 5), strides=2,  padding="same"))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Convolution2D(36, kernel_size=(5, 5), strides=2, padding="same"))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Convolution2D(48, kernel_size=(5, 5), strides=2, padding="same"))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Convolution2D(64, kernel_size=(3, 3), strides=1, padding="same"))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Convolution2D(64, kernel_size=(3, 3), strides=1, padding="valid"))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(50))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(10))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(1))

#compile and train the model
#thanks to batch normalization the number of epochs is really low
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(X_train, y_train, validation_split = 0.3, shuffle = True, epochs = 6)

import tensorflow as tf
import keras.backend.tensorflow_backend as ktf

#store the model
model.save('model.h5')
