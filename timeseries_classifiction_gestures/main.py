#!/usr/bin/env python

# https://machinelearningmastery.com/cnn-models-for-human-activity-recognition-time-series-classification/
# https://keras.io/layers/core/
# https://machinelearningmastery.com/pooling-layers-for-convolutional-neural-networks/

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv1D
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import MaxPooling1D
from keras.layers import Masking

import sys
import math
import random
import numpy as np
import json

from util import get_all_data
from util import one_hot_arrays
from util import normalize_arr_3d
from util import get_num_timesteps
from util import load_xy, save_xy
from util import clamp
from util import seeded_shuffle

from constants import num_features

samples_limit = 10000

# process data for usage with keras:
# split into x/y and normalize
def build_xy():

  data = get_all_data(limit=samples_limit)
  num_timesteps = get_num_timesteps(data)

  x = []
  y = []
  for sample in data:
    # sample timestep idx has features from row
    old_sample_len = len(sample)
    new_sample = np.empty((num_timesteps, num_features)).tolist()
    for idx in range(0, num_timesteps):
      if idx < old_sample_len:
        new_sample[idx] = sample[idx][:num_features] # features
      else:
        new_sample[idx] = np.zeros(num_features).tolist()

    output = sample[0][num_features]

    x.append(new_sample)
    y.append(output)

  x = normalize_arr_3d(x, clamped=True)
  x = np.array(x, dtype=np.float32)
  y = np.array(y, dtype=np.uint8)
  y = one_hot_arrays(y)

  return x, y

def run(times=1):
  try:
    x, y = load_xy()
  except:
    x, y = build_xy()
    save_xy(x, y)

  num_timesteps = x.shape[1]
  num_classes = y.shape[1]
  num_samples = len(x)
  num_train = int(math.floor(num_samples * 0.7))
  num_test = num_samples - num_train
  seeds = random.sample(range(100000), times)

  print('num samples:               ', num_samples)
  print('num timesteps per sample:  ', num_timesteps)
  print('num features per timestep: ', num_features)
  print('num output classes:        ', num_classes)
  print('num train samples:         ', num_train)
  print('num test samples:          ', num_test)
  print('seeds:                     ', seeds)

  accuracy_results = []

  for run_iteration in range(0, times):

    seeded_shuffle(x, seeds[run_iteration])
    seeded_shuffle(y, seeds[run_iteration])

    x_train = x[:num_train]
    y_train = y[:num_train]
    x_test = x[num_train:]
    y_test = y[num_train:]

    model = Sequential()

    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(num_timesteps, num_features)))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))

    # Dropout consists in randomly setting a fraction rate of input units
    # to 0 at each update during training time, which helps prevent overfitting.
    model.add(Dropout(0.5))

    # Downsamples the input. Calculate the maximum value for each patch of the feature map.
    model.add(MaxPooling1D(pool_size=2))

    # Flattens the input. Does not affect the batch size.
    model.add(Flatten())

    # If all features for a given sample timestep are equal to mask_value,
    # then the sample timestep will be masked (skipped) in all downstream 
    # layers (as long as they support masking).
    model.add(Masking(mask_value=0.0))

    # Just your regular densely-connected NN layer.
    model.add(Dense(100, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(
      loss='categorical_crossentropy', 
      optimizer='adam',
      metrics=['accuracy']
    )

    model.fit(x_train, y_train, epochs=13, batch_size=32)

    _, accuracy = model.evaluate(x_test, y_test)

    print('accuracy: ', accuracy)
    accuracy_results.append(accuracy)

    if run_iteration == times - 1:
      print('\n')
      print('results: ', accuracy_results) 

run(times=3)