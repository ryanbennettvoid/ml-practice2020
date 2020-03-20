#!/usr/bin/env python

# https://machinelearningmastery.com/cnn-models-for-human-activity-recognition-time-series-classification/

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv1D
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import MaxPooling1D

import math
import numpy as np
import json
from util import get_all_data
from util import hot_one_arrays

num_features = 8

# process data for usage with keras:
# split into x/y and normalize
def buildXY():
  data = np.array(get_all_data())

  num_timesteps = -1
  for segment in data:
    segment_size = len(segment)
    num_timesteps = segment_size if segment_size > num_timesteps else num_timesteps

  assert num_timesteps > -1
  print('num_timesteps', num_timesteps)

  xMin = 10000
  xMax = -10000

  x = []
  y = []
  for segment in data:
    sample = np.empty((num_timesteps, num_features)).tolist()
    output = int(segment[0][9] - 1) # starts at 1 so offset to start at 0

    # populate samples with timesteps, timesteps with features
    for idx, row in enumerate(segment):
      features = row[1:9]
      for feature in features:
        xMin = feature if feature < xMin else xMin
        xMax = feature if feature > xMax else xMax
      assert len(features) == num_features
      sample[idx] = features

    # populate remaining timesteps with zero-filled features
    remaining_start = len(segment)
    for idx in range(remaining_start, num_timesteps):
      sample[idx] = np.zeros(num_features).tolist()

    x.append(sample)
    y.append(output)

  x = np.array(x)
  x = (x - xMin) / (xMax - xMin) # normalize
  y = np.array(y, dtype=np.uint8)
  y = hot_one_arrays(y)

  return x, y, xMin, xMax, num_timesteps

x, y, xMin, xMax, num_timesteps = buildXY()
num_classes = len(y[0])

print('x shape:', x.shape)
print('y shape:', y.shape)

num_samples = len(x)
num_train = int(math.floor(num_samples * 0.7))
num_test = num_samples - num_train
x_train = x[:num_train]
y_train = y[:num_train]
x_test = x[num_train:]
y_test = y[num_train:]

print('num_train:', num_train)
print('num_test:', num_test)

model = Sequential()

model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(num_timesteps, num_features)))
model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
model.add(Dropout(0.5))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(
  loss='categorical_crossentropy', 
  optimizer='adam',
  metrics=['accuracy']
)

model.fit(x_train, y_train, epochs=10, batch_size=32)

_, accuracy = model.evaluate(x_test, y_test)

print('accuracy: ', accuracy)