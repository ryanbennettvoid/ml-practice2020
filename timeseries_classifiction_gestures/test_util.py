
import pytest
import numpy as np
from os import listdir

from constants import data_dir
from constants import num_data_files
from constants import test_file_01_1_filename
from constants import test_file_01_1_first_sample_time
from constants import test_file_01_1_first_sample_output
from constants import test_file_01_1_first_sample_len
from constants import test_file_01_1_second_sample_time
from constants import test_file_01_1_second_sample_output
from constants import test_file_01_1_second_sample_len

from util import get_filepaths
from util import get_data_from_file
from util import get_all_data
from util import normalize_arr_3d
from util import get_num_timesteps

def test_get_filepaths():
  result = get_filepaths()
  assert len(result) == num_data_files
  assert type(result[0]) is str
  assert result[0].find(data_dir) > -1
  assert result[0].find('.txt') > -1

def test_get_data_from_file():
  filepath = 'data/01/%s' % test_file_01_1_filename
  result = get_data_from_file(filepath)

  key1 = '%s__%s__%s' % (filepath, test_file_01_1_first_sample_time, test_file_01_1_first_sample_output)
  assert key1 in result.keys()
  assert len(result[key1]) == test_file_01_1_first_sample_len

  key2 = '%s__%s__%s' % (filepath, test_file_01_1_second_sample_time, test_file_01_1_second_sample_output)
  assert key2 in result.keys()
  assert len(result[key2]) == test_file_01_1_second_sample_len

def test_get_all_data_limit():
 result1 = get_all_data(limit=2)
 assert len(result1) == 2         # samples
 assert len(result1[0]) > 0       # rows in sample
 assert len(result1[0][0]) == 10  # cols in row

 result1 = get_all_data(limit=3)
 assert len(result1) == 3         # samples
 assert len(result1[0]) > 0       # rows in sample
 assert len(result1[0][0]) == 10  # cols in row

def test_normalize_arr_3d_by_column():
  arr = [
    [[-8, -9, -9], [4, -5, -5], [16, -2, -2]],
    [[64, 3, 3], [256, 4, 4], [1024, 5, 5]],
    [[4096, 6, 6], [5000, 7, 7], [6000, 8, 8]]
  ]
  # fails if max is significantly larger, but okay for this
  # dataset
  result1 = normalize_arr_3d(arr)
  result2 = normalize_arr_3d(arr, clamped=True)
  for result in [result1, result2]:
    print result

    assert result[0][0][0] == 0
    assert result[0][0][1] == 0
    assert result[0][0][2] == 0

    assert result[2][2][0] == 1
    assert result[2][2][1] == 1
    assert result[2][2][2] == 1

    assert not np.isclose(
      [result[1][0][0]],
      result[1][0][1],
      atol=0.01
    ).any()

    assert np.isclose(
      [result[1][0][1]],
      result[1][0][2],
      atol=0.00001
    ).any()

    for d0, row in enumerate(result):
        for d1, col in enumerate(row):
          for d2, depth in enumerate(col):
            val = result[d0][d1][d2]
            assert val >= 0
            assert val <= 1


def test_get_num_timesteps():
  arr = [
    [0],
    [0, 1, 2, 3],
    [0, 1, 2],
  ]
  result = get_num_timesteps(arr)
  assert result == 4