#!/usr/bin/env python

# https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/

from keras.models import Sequential
from keras.layers import Dense

from util import get_image_data_train
from util import get_label_data_train
from util import get_image_data_test
from util import get_label_data_test

from util import hot_one_arrays

from constants import num_train_samples
from constants import num_test_samples
from constants import image_size
from constants import num_classes

x = get_image_data_train()
y = hot_one_arrays(get_label_data_train())

test_x = get_image_data_test()
test_y = hot_one_arrays(get_label_data_test())

print('x', x.shape)
print('y', y.shape)
print('test_x', test_x.shape)
print('test_y', test_y.shape)

model = Sequential()

model.add(Dense(12, input_dim=image_size, activation='relu'))
model.add(Dense(image_size, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(
  loss='categorical_crossentropy', 
  optimizer='adam',
  metrics=['accuracy']
)

model.fit(x, y, epochs=10, batch_size=100)

result, accuracy = model.evaluate(test_x, test_y)

print('result', result)
print('accuracy', accuracy)