#!/usr/bin/env python

# https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/

import math
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense

total_sample_size = 768
train_size = int(math.floor(total_sample_size * 0.7))
test_size = int(total_sample_size - train_size)
num_features = 8

print('total_sample_size: ', total_sample_size)
print('train_size: ', train_size)
print('test_size: ', test_size)

dataset = loadtxt('./data.csv', delimiter=',')
x = dataset[0:train_size,0:num_features]
y = dataset[0:train_size,num_features]
test_x = dataset[train_size:,0:num_features]
test_y = dataset[train_size:,num_features]
assert(len(test_x) == test_size)
assert(len(test_y) == test_size)

model = Sequential()

# input layer: input_dim is length of input row
# 1st hidden layer: 12 nodes
model.add(Dense(12, input_dim=num_features, activation='relu'))

# 2nd hidden layer: 
model.add(Dense(num_features, activation='relu'))

# output layer
model.add(Dense(1, activation='sigmoid'))

# compile the model
# binary classification problem = use 'binary_crossentropy'
# general-use optimizer: use 'adam'
# fitness function: use 'accuracy'
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# configure how the model will run
# epoch = one iteration over all inputs
# batch = number of samples before updating weights
model.fit(x, y, epochs=150, batch_size=50)

# make predictions
result, accuracy = model.evaluate(test_x, test_y)

print('result: ', result)
print('accuracy: ', accuracy)