
import pytest
from os import listdir

from constants import data_dir
from constants import num_data_files
from constants import test_file_01_1_filename
from constants import test_file_01_1_first_segment_time
from constants import test_file_01_1_first_segment_output
from constants import test_file_01_1_first_segment_len
from constants import test_file_01_1_second_segment_time
from constants import test_file_01_1_second_segment_output
from constants import test_file_01_1_second_segment_len

from util import get_filepaths
from util import get_data_from_file
from util import get_all_data

def test_get_filepaths():
  result = get_filepaths()
  assert len(result) == num_data_files
  assert type(result[0]) is str
  assert result[0].find(data_dir) > -1
  assert result[0].find('.txt') > -1

def test_get_data_from_file():
  filepath = 'data/01/%s' % test_file_01_1_filename
  result = get_data_from_file(filepath)

  key1 = '%s__%s__%s' % (filepath, test_file_01_1_first_segment_time, test_file_01_1_first_segment_output)
  assert key1 in result.keys()
  assert len(result[key1]) == test_file_01_1_first_segment_len

  key2 = '%s__%s__%s' % (filepath, test_file_01_1_second_segment_time, test_file_01_1_second_segment_output)
  assert key2 in result.keys()
  assert len(result[key2]) == test_file_01_1_second_segment_len

def test_get_all_data_limit():
 result1 = get_all_data(limit=2)
 assert len(result1) == 2         # segments
 assert len(result1[0]) > 0       # rows in segment
 assert len(result1[0][0]) == 10  # cols in row

 result1 = get_all_data(limit=3)
 assert len(result1) == 3         # segments
 assert len(result1[0]) > 0       # rows in segment
 assert len(result1[0][0]) == 10  # cols in row

