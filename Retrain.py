# coding: utf-8
# In[1]:
import csv
import cv2 as openCV
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.models import load_model


lines = []
with open('./my_data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
#lines.pop(0)

images = []
measurements = []

#def generators():
for line in lines:
    for i in range(3):
        source_path = line[i]
        filename = source_path.split('/')[-1]
        current_path = './my_data/IMG/' + filename
        image = openCV.imread(current_path)
        images.append(image)
        #image_flipped = np.fliplr(image)
        #images.append(image_flipped)

        measurement = float(line[3])
        if i==1:
            measurement += 0.1
        if i==2:
            measurement -= 0.1
        measurements.append(measurement)
            #yield image, measurement
        #measurement_flipped = -measurement
        #measurements.append(measurement_flipped)

X_train = np.array(images)
y_train = np.array(measurements)

#X_train, y_train = generators()
model = Sequential()

model = load_model('model.h5')
model.fit(X_train, y_train, validation_split=0.1, shuffle=True, epochs=3)
model.save('model.h5')
