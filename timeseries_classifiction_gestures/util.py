
import numpy as np
from os import listdir
from os.path import isfile, join

from constants import data_dir

# returns array of filepaths for all data files
def get_filepaths():
  filepaths = []
  dirs = listdir(data_dir)
  for _, dirname in enumerate(dirs):
    for f in listdir(join(data_dir, dirname)):
      filepaths.append(join(data_dir, dirname, f))
  filepaths.sort()
  return filepaths

# returns a unique key string that represents a segment
def make_segment_key(filepath, last_time, last_output):
  return '%s__%s__%s' % (filepath, int(last_time), int(last_output))

# returns array of all segments from a file
def get_data_from_file(filepath):
  data = np.loadtxt(filepath, delimiter='\t', skiprows=1)
  data_dict = dict()

  last_output = -1
  last_time = -1
  key = None

  for row in data:
    time = int(row[0])
    output = int(row[9])

    if output != last_output:
      last_time = time
      last_output = output
      key = make_segment_key(filepath, last_time, last_output)

    if output > 0:
      if key not in data_dict:
        data_dict[key] = []
      data_dict[key].append(row.tolist())
  print('loaded file: ', filepath)
  return data_dict

# returns an array of all segments from all files
def get_all_data(limit=0):
  filepaths = get_filepaths()
  all_data = []
  for idx, filepath in enumerate(filepaths):
    try:
      file_data = get_data_from_file(filepath)
      for segment in list(file_data.values()):
        all_data.append(segment)
        if limit > 0 and len(all_data) >= limit:
          return all_data
    except:
      print('failed to parse file: ', filepath)
  return all_data

def hot_one_arrays(np_arr):
  num_rows = int(np_arr.size)
  num_cols = int(np_arr.max() + 1)
  new_arr = np.zeros((num_rows, num_cols))
  new_arr[np.arange(num_rows), np_arr] = 1
  return new_arr
