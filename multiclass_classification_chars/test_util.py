
import pytest
import numpy as np

from util import get_image_data_ubyte
from util import get_image_data_train
from util import get_image_data_test

from util import get_label_data_ubyte
from util import get_label_data_train
from util import get_label_data_test

from util import hot_one_arrays

from constants import test_images_filename
from constants import test_labels_filename
from constants import num_train_samples
from constants import num_test_samples
from constants import image_size

# images

def test_get_image_data_ubyte():
  filename = test_images_filename
  result = get_image_data_ubyte(
    filename=filename, 
    num_images=num_test_samples,
    image_size=image_size
  )
  assert len(result) == num_test_samples
  assert len(result[0]) == image_size

def test_get_image_data_train():
  result = get_image_data_train()
  assert len(result) == num_train_samples
  assert len(result[0]) == image_size

def test_get_image_data_test():
  result = get_image_data_test()
  assert len(result) == num_test_samples
  assert len(result[0]) == image_size

# labels

def test_get_label_data_ubyte():
  filename = test_labels_filename
  result = get_label_data_ubyte(
    filename=filename, 
    num_labels=num_test_samples
  )
  assert len(result) == num_test_samples
  assert result[0] >= 0 and result[0] <= 9

def test_get_label_data_train():
  result = get_label_data_train()
  assert len(result) == num_train_samples
  assert result[0] >= 0 and result[0] <= 9

def test_get_label_data_test():
  result = get_label_data_test()
  assert len(result) == num_test_samples
  assert result[0] >= 0 and result[0] <= 9

# other

def test_hot_one():
  result = hot_one_arrays(np.array([ 3, 0, 5, 1 ]))
  expected_result = np.array([
    [0, 0, 0, 1, 0, 0],
    [1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1],
    [0, 1, 0, 0, 0, 0]
  ])
  np.testing.assert_equal(result, expected_result)


