
import numpy as np

from constants import test_images_filename
from constants import test_labels_filename
from constants import train_images_filename
from constants import train_labels_filename

# https://stackoverflow.com/questions/40427435/extract-images-from-idx3-ubyte-file-or-gzip-via-python

# images

def get_image_data_ubyte(filename, num_images, image_size):
  with open(filename, 'r') as f:
    f.read(16)
    buf = f.read(num_images * image_size)
    data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    data = data.reshape(num_images, image_size)
    return data

def get_image_data_train():
  return get_image_data_ubyte(
    filename=train_images_filename, 
    num_images=60000,
    image_size=28 * 28
  )

def get_image_data_test():
  return get_image_data_ubyte(
    filename=test_images_filename, 
    num_images=10000,
    image_size=28 * 28
  )

# labels

def get_label_data_ubyte(filename, num_labels):
  with open(filename, 'r') as f:
    f.read(8)
    data = np.array([], dtype=np.uint8)
    for i in range(0, num_labels):
      buf = f.read(1)
      value = np.frombuffer(buf, dtype=np.uint8)
      data = np.append(data, value)
    return data

def get_label_data_train():
  return get_label_data_ubyte(
    filename=train_labels_filename, 
    num_labels=60000
  )

def get_label_data_test():
  return get_label_data_ubyte(
    filename=test_labels_filename, 
    num_labels=10000
  )

# other

# https://stackoverflow.com/questions/29831489/convert-array-of-indices-to-1-hot-encoded-numpy-array

def hot_one_arrays(np_arr):
  num_rows = int(np_arr.size)
  num_cols = int(np_arr.max() + 1)
  new_arr = np.zeros((num_rows, num_cols))
  new_arr[np.arange(num_rows), np_arr] = 1
  return new_arr
