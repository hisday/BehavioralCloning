import csv
import os

import cv2 as openCV
import numpy as np
#import matplotlib.pyplot as plt


from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from random import shuffle

samples = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)
samples.pop(0)

from sklearn.model_selection import train_test_split
import sklearn

train_samples, validation_samples = train_test_split(samples, test_size=0.2)

def generator(samples, batch_size = 90):
    num_samples = len(samples)
    while 1:
        #shuffle(samples)
        size = int(batch_size / 3)
        for offset in range(0, num_samples, size):
            batch_samples = samples[offset:offset+size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                for i in range(3):
                    name = './data/IMG/'+batch_sample[i].split('/')[-1]
                    image = openCV.imread(name)
                    images.append(image) 
                    
                    angle = float(batch_sample[3])

                    if i == 1:
                        angle += 0.09
                    if i == 2:
                        angle -= 0.09
                    angles.append(angle)
                    
            X_train = np.array(images)
            y_train = np.array(angles)
            #print(len(X_train))
            print("bucket size: {}".format(len(y_train)))
            print("size: {}".format(size))

            yield sklearn.utils.shuffle(X_train, y_train)


batchsize = 90
epoch = 3

print ("batchsize : {}".format(batchsize))
print ("epoch : {}".format(epoch))
train_generator = generator(train_samples, batch_size=batchsize)
validation_generator = generator(validation_samples, batch_size=batchsize)

model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(((70,25), (0, 0))))
model.add(Convolution2D(24,(5,5), strides= (2,2), activation="relu"))
model.add(Convolution2D(36,(5,5), strides= (2,2), activation="relu"))
model.add(Convolution2D(48,(5,5), strides= (2,2), activation="relu"))
model.add(Convolution2D(64,(3,3), activation="relu"))
model.add(Convolution2D(64,(3,3), activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))    
model.add(Dense(10))
model.add(Dense(1))
model.compile(loss= 'mse', optimizer='adam')
history_object = model.fit_generator(train_generator, steps_per_epoch= len(train_samples)/batchsize, epochs = epoch, verbose=1, validation_data=validation_generator, validation_steps=len(validation_samples)/batchsize)
model.save('model.h5')

