#!/usr/bin/env python

# https://machinelearningmastery.com/cnn-models-for-human-activity-recognition-time-series-classification/

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import MaxPooling2D

from scipy.ndimage.filters import gaussian_filter

import math
import numpy as np
import json
from util import get_all_data
from util import hot_one_arrays

num_features = 8
feature_depth=64

def blur_array(arr):
  return gaussian_filter(arr, sigma=2)

# convert 1d array of normalized values to a 2d array
def to_2d_arr(arr):
  dim1_size = len(arr)
  dim2_size = feature_depth

  new_arr = np.zeros((dim1_size, dim2_size))
  for idx, feature in enumerate(new_arr):
    idx2 = int(math.floor(arr[idx] * dim2_size))
    new_arr[idx][idx2] = 1

  return new_arr


# process data for usage with keras:
# split into x/y and normalize
def build_xy():
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
    sample = np.empty((num_timesteps, num_features, feature_depth))
    output = int(segment[0][9] - 1) # starts at 1 so offset to start at 0

    # populate samples with timesteps, timesteps with features
    for idx, row in enumerate(segment):
      features = row[1:9]
      for feature in features:
        xMin = feature if feature < xMin else xMin
        xMax = feature if feature > xMax else xMax
      assert len(features) == num_features
      sample[idx] = blur_array(to_2d_arr(features))

    # populate remaining timesteps with zero-filled features
    remaining_start = len(segment)
    for idx in range(remaining_start, num_timesteps):
      sample[idx] = np.zeros((num_features, feature_depth)).tolist()

    x.append(sample)
    y.append(output)

  x = np.array(x)
  x = (x - xMin) / (xMax - xMin) # normalize
  y = np.array(y, dtype=np.uint8)
  y = hot_one_arrays(y)

  return x, y, xMin, xMax, num_timesteps

x, y, xMin, xMax, num_timesteps = build_xy()
num_classes = len(y[0])

print('x shape:', x.shape)
print('y shape:', y.shape)

num_samples = len(x)
num_train = int(math.floor(num_samples * 0.9))
num_test = num_samples - num_train
x_train = x[:num_train]
y_train = y[:num_train]
x_test = x[num_train:]
y_test = y[num_train:]

print('num_train:', num_train)
print('num_test:', num_test)

model = Sequential()

model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', input_shape=(num_timesteps, num_features, feature_depth)))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(Dropout(0.5))
model.add(MaxPooling2D(pool_size=2))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(
  loss='categorical_crossentropy', 
  optimizer='adam',
  metrics=['accuracy']
)

model.fit(x_train, y_train, epochs=10, batch_size=128)

_, accuracy = model.evaluate(x_test, y_test)

print('accuracy: ', accuracy)