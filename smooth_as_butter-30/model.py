import tensorflow as tf
import csv
import cv2
import numpy as np
from random import randint

# Initial Setup for Keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D
from keras.layers.normalization import BatchNormalization

#config.gpu_options.per_process_gpu_memory_fraction = 0.55

images = []
steering = []
base_data_path = ['../data_ud/']
#14 is where it starts to sviwel but at least it turns
correction = 0.04

def readImage(path):
    return cv2.resize(cv2.imread(path), (64, 64))
def readFile(base_data_path):
    with open(base_data_path + 'driving_log.csv') as csv_file:
        reader = csv.reader(csv_file)
        headers = next(reader)
        for line in reader:
            if float(line[headers.index('steering')]) == 0 and randint(0,9) < 4:
                continue

            #if abs(float(line[headers.index('steering')])) > 0.3:
            #    continue
            steering_value = float(line[headers.index('steering')])
            images.append(readImage(base_data_path + line[headers.index('center')]))
            steering.append(steering_value)

            if  0 <= 0 :

                images.append(readImage(base_data_path + line[headers.index('left')].strip()))
                steering.append(steering_value + correction)

                images.append(np.fliplr(readImage(base_data_path + line[headers.index('left')].strip())))
                steering.append(-steering_value - correction)

            if  0 >= 0 :

                images.append(readImage(base_data_path + line[headers.index('right')].strip()))
                steering.append(steering_value - correction)

                images.append(np.fliplr(readImage(base_data_path + line[headers.index('right')].strip())))
                steering.append(-steering_value + correction)


            #mirror images
            images.append(np.fliplr(readImage(base_data_path + line[headers.index('center')])))
            steering.append(-steering_value)

            #break

[readFile(x) for x in base_data_path]
X_train = np.array(images)
print (X_train.shape )
y_train = np.array(steering)
#print(X_train[0].shape())

#Nvidia model here we go
model = Sequential()
model.add(Cropping2D(cropping=((25,9), (0,0)), input_shape=(64,64,3)))
model.add(Lambda(lambda x: x / 255.0 - 0.5))
#90x320
model.add(Convolution2D(24, 5, 5, border_mode="same", subsample=(1, 1)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Convolution2D(36, 5, 5, border_mode="same", subsample=(2, 2)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Convolution2D(48, 5, 5, border_mode="same", subsample=(2, 2)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3, border_mode="same", subsample=(2, 2)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3, border_mode="valid", subsample=(2, 2)))
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

model.compile(loss = 'mse', optimizer = 'adam', lr = 0.1, decay = 0.1)
model.fit(X_train, y_train, validation_split = 0.2, shuffle = True, nb_epoch = 5)

import tensorflow as tf
import keras.backend.tensorflow_backend as ktf

#def get_session(gpu_fraction=0.6):
#    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction,
#                                allow_growth=True)
#    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

#ktf.set_session(get_session())

model.save('model.h5')
print("changed initial learning reate from 0.001 to 0.1 added decay of 0.1")
