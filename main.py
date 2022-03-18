import pandas as pd
import numpy as np 
from libsvm.svmutil import *
from utils import utils 
import os
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--num_disjoint", type=int, default=5,
              help="Number of epoch to train")
args = parser.parse_args()

def read_data():
  # reading csv files
  data =  pd.read_csv('./data/abalone.data').to_numpy()
  
  gender2vec = np.array([[1., 0., 0.]]) * (data[:, :1] == "M").astype(np.float32) + \
               np.array([[0., 1., 0.]]) * (data[:, :1] == "I").astype(np.float32) + \
               np.array([[0., 0., 1.]]) * (data[:, :1] == "F").astype(np.float32) 
  data = np.concatenate([gender2vec, data[:, 1:]], 1)

  label = ((data[:, -1] >= 1) | (data[:, -1] <= 9)).astype(np.int64) + \
          - (data[:, -1] > 9).astype(np.int64)
  train = {"features": data[:3133, :-1],
           "labels": label[:3133]}
  test = {"features": data[3133:, :-1],
           "labels": label[3133:]}
  min_train = np.min(train['features'][:, 3:], 0, keepdims=True)
  max_train = np.max(train['features'][:, 3:], 0, keepdims=True)
  def preprocess(data):
    non_cat = (data["features"][:, 3:] - min_train) / (max_train - min_train)
    data["features"] = np.concatenate([data["features"][:, :3],
                          non_cat], 1)
    return data
  train = preprocess(train)
  test = preprocess(test)
  
  return train, test 

def split_data(data):
  block_range = int(data['features'].shape[0]/args.num_disjoint)
  idx = [i for i in range(0, data['features'].shape[0], 
                   block_range)]
  idx[-1] = data['features'].shape[0]

  first_train_idx = [(0, id) for id in idx][1:-1]
  first_train_idx.insert(0, None)
  second_train_idx = [(idx[i], idx[-1]) for i in range(1, len(idx) - 1)] + [None]
  test_idx = [(idx[i], idx[i+1]) for i in range(len(idx)-1)]
   
  split = []
  for f_id, s_id, t_id in zip(first_train_idx, second_train_idx, test_idx):
    out = {"train": {},
         "test": {}}
    train_features, train_labels = [], []
    if f_id is not None:
      train_features.append(data["features"][f_id[0]: f_id[1]])
      train_labels.append(data["labels"][f_id[0]: f_id[1]])
    if s_id is not None:
      train_features.append(data["features"][s_id[0]: s_id[1]])
      train_labels.append(data["labels"][s_id[0]: s_id[1]])
    
    train_features = np.concatenate(train_features, 0)
    train_labels = np.concatenate(train_labels, 0)
    test_features = data["features"][t_id[0]: t_id[1]]
    test_labels = data["labels"][t_id[0]: t_id[1]]

    out["train"]["features"] = train_features
    out["train"]["labels"] = train_labels
    out["test"]["features"] = test_features
    out["test"]["labels"] = test_labels
   
    split.append(out)
 
  return split 

def main():

  train, test = read_data()
  C = [3**k for k in range(-2, 3)]
  for c_param in C:
    for split_train in split_data(train):
      m = svm_train(split_train['label'], split_train['features'], f'-c {c_param} -t 1')
      p_label, p_acc, p_val = svm_predict(split_train['label'], split_train['features'], m)
  #print(p_label)
  exit()
if __name__ == "__main__":
  sys.excepthook = utils.colored_hook(os.path.dirname(os.path.realpath(__file__)))
  main()