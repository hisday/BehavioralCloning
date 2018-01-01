# coding: utf-8
# In[1]:
import csv
import cv2 as openCV
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

''' 
lines = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
lines.pop(0) '''

samples = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)
samples.pop(0)

from sklearn.model_selection import train_test_split
import sklearn

train_samples, validation_samples = train_test_split(samples, test_size=0.2)

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:
        #shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                for i in range(3):
                    name = './data/IMG/'+batch_sample[i].split('/')[-1]
                    image = openCV.imread(name)
                    images.append(image)
                    
                    angle = float(batch_sample[3])
                    if i == 1:
                        angle += 0.1
                    if i == 2:
                        angle -= 0.1
                    angles.append(angle)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


images = []
measurements = []
''' 
def generators():
    for line in lines:
        for i in range(3):
            source_path = line[i]
            filename = source_path.split('/')[-1]
            current_path = './data/IMG/' + filename
            image = openCV.imread(current_path)
            images.append(image)
            #image_flipped = np.fliplr(image)
            #images.append(image_flipped)

            measurement = float(line[3])
            if i == 1:
                measurement += 0.1
            if i == 2:
                measurement -= 0.1
            measurements.append(measurement)
            #yield image, measurement
            #measurement_flipped = -measurement
            #measurements.append(measurement_flipped)

X_train = np.array(images)
y_train = np.array(measurements)
 '''

train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

#X_train, y_train = generators()
model = Sequential()


model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(((70,25), (0, 0))))
model.add(Convolution2D(24,(5,5), strides= (2,2), activation="relu"))
model.add(Convolution2D(36,(5,5), strides= (2,2), activation="relu"))
model.add(Convolution2D(48,(5,5), strides= (2,2), activation="relu"))
model.add(Convolution2D(64,(3,3),activation="relu"))
model.add(Convolution2D(64,(3,3),activation="relu"))

#model.add(Convolution2D(6,5,5,activation="relu"))
#model.add(MaxPooling2D())
#model.add(Convolution2D(6,5,5,activation="relu"))
#model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
model.compile(loss= 'mse', optimizer='adam')
#model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=3)

model.fit_generator(train_generator, steps_per_epoch= len(train_samples)/32, epochs = 1, validation_data=validation_generator, validation_steps=len(validation_samples)/32)

model.save('model.h5')

