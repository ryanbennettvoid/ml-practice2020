
import sys
from os import listdir
from os.path import isfile, join

import numpy as np
import random

from constants import data_dir
from constants import x_filename
from constants import y_filename

def clamp(n, smallest, largest):
  return max(smallest, min(n, largest))

# returns array of filepaths for all data files
def get_filepaths():
  filepaths = []
  dirs = listdir(data_dir)
  for _, dirname in enumerate(dirs):
    for f in listdir(join(data_dir, dirname)):
      filepaths.append(join(data_dir, dirname, f))
  filepaths.sort()
  return filepaths

# returns a unique key string that represents a sample
def make_sample_key(filepath, last_time, last_output):
  return '%s__%s__%s' % (filepath, int(last_time), int(last_output))

# returns array of all samples from a file
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
      key = make_sample_key(filepath, last_time, last_output)

    if output > 0:
      if key not in data_dict:
        data_dict[key] = []
      data_dict[key].append(row.tolist())
  print('loaded file: ', filepath)
  return data_dict

# returns an array of all samples from all files
def get_all_data(limit=0):
  filepaths = get_filepaths()
  all_data = []
  for idx, filepath in enumerate(filepaths):
    try:
      file_data = get_data_from_file(filepath)
      for sample in list(file_data.values()):
        all_data.append(sample)
        if limit > 0 and len(all_data) >= limit:
          return all_data
    except KeyboardInterrupt:
      sys.exit(1)
    except:
      print('failed to parse file: ', filepath)
  return all_data

# returns array of one-hot arrays for a given array of arrays
def one_hot_arrays(np_arr):
  num_rows = int(np_arr.size)       # num rows
  num_cols = int(np_arr.max() + 1)  # max col value in the row
  new_arr = np.zeros((num_rows, num_cols))
  new_arr[np.arange(num_rows), np_arr] = 1
  return new_arr

# returns a normalized array for a given array
def normalize_arr_3d(arr_3d, clamped=False):
  num_samples = len(arr_3d)
  num_timesteps = len(arr_3d[0])
  num_features = len(arr_3d[0][0])

  col_min = np.zeros(num_features, dtype=np.float64)
  col_min.fill(sys.maxsize)
  col_max = np.zeros(num_features, dtype=np.float64)
  col_max.fill(-sys.maxsize)
  for sample in arr_3d:
    for timestep in sample:
      for idx, feature in enumerate(timestep):
        current_value = feature
        col_min[idx] = min(col_min[idx], feature)
        col_max[idx] = max(col_max[idx], feature)

  new_arr = np.empty((num_samples, num_timesteps, num_features), dtype=np.float64)
  for d0,sample in enumerate(arr_3d):
    for d1, timestep in enumerate(sample):
      for idx, feature in enumerate(timestep):
        feature_min = col_min[idx]
        feature_max = col_max[idx]
        normalized_feature = (feature - feature_min) / (feature_max - feature_min)
        if clamped:
          normalized_feature = clamp(normalized_feature, 0, 1)
          assert normalized_feature >= 0
          assert normalized_feature <= 1
        new_arr[d0][d1][idx] = normalized_feature
  return new_arr

# returns the max number of timesteps in an array of samples
def get_num_timesteps(data):
  num_timesteps = -1
  for sample in data:
    num_timesteps = max(num_timesteps, len(sample))
  assert num_timesteps > -1
  return num_timesteps

def load_xy():
  x = np.load(x_filename)
  y = np.load(y_filename)
  num_samples = len(x)
  print('loaded %s samples from cache' % num_samples)
  return x, y

def save_xy(x, y):
  np.save(x_filename, x)
  np.save(y_filename, y)
  num_samples = len(x)
  print('saved %s samples to cache' % num_samples)

def seeded_shuffle(arr, seed):
  random.Random(seed).shuffle(arr)