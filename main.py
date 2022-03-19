import pandas as pd
import numpy as np 
from libsvm.svmutil import *
from utils import utils 
import os
import sys
import argparse
import matplotlib.pyplot as plt 

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

def plot(mean):
  for degree in mean.keys():
    all_mean = mean[degree]
    x, y, e = [], [], []
    for key, val in all_mean.items():
      #x.append(key)
      x.append(1)
      y.append(val['mean'])
      e.append(val['std'])
    plt.errorbar(x, y, e, fmt = 'o', color = 'black', 
              linestyle='None', ecolor = 'lightblue',
              capsize=5)
    plt.ylim(min(y) - 0.05, max(y) + 0.05)
    plt.xlabel("C")
    plt.ylabel("Mean Square Error")
    plt.title(f"Cross-validation error for polynomial kernel degree {degree}")
    #plt.yscale("log")
    plt.savefig(f"./{degree}.jpg")
    plt.clf()
 
def question3_2(train, test, best_param):
  c_param = best_param["C"]
  degree = range(1, 9)
  #degree = [1, 2]
  mse = {}
  dataloader = split_data(train)
  loss = np.inf
  best_param = {"degree": None, "C": None}
  for d_param in degree:
    mse[d_param] = {}
    cross_val = []
    acc_val = []
    num_support_vec = []
    for split_train in dataloader:
      train_data = split_train["train"]
      test_data = split_train["test"]
      options = f"-r {c_param} -d {d_param} -t 1 -q"
      m = svm_train(train_data["labels"], train_data["features"], options)
      p_label, p_acc, p_val = svm_predict(test_data["labels"], test_data['features'], m, "-q")
      cross_val.append(p_acc[1])
      acc_val.append(p_acc[0])
      if loss > p_acc[1]:
        loss = p_acc[1] 
        best_param["degree"] = d_param
        best_param["C"] = c_param

      _, _, p_val = svm_predict(train_data["labels"], train_data['features'], m, "-q")
      num_support_vec.append(np.sum((np.abs(p_val) == 1.0).float()))
      
    #print(c_param, p_acc[1])
    cross_val = np.array(cross_val)
    acc_val = np.array(acc_val)
    mse[d_param] = {"mean": np.mean(cross_val), 
                    "std": np.std(cross_val),
                    "acc": np.mean(acc_val),
                    "num_support_vec": np.mean(num_support_vec)}
  print(best_param)
  print('domme')
  exit()
def question3_1(train, test):
  C = 3** np.linspace(2, 15, 10) 
  #print(C)
  degree = [1, 2, 3, 4, 5]
  #degree = [1, 2]
  mse = {}
  dataloader = split_data(train)
  loss = np.inf
  best_param = {"degree": None, "C": None}
  for d_param in degree:
    mse[d_param] = {}
    for c_param in C:
      cross_val = []
      acc_val = []
      for split_train in dataloader:
        train_data = split_train["train"]
        test_data = split_train["test"]
        options = f"-c {c_param} -d {d_param} -t 1 -q"
        m = svm_train(train_data["labels"], train_data["features"], options)
        p_label, p_acc, p_val = svm_predict(test_data["labels"], test_data['features'], m, "-q")
        cross_val.append(p_acc[1])
        acc_val.append(p_acc[0])
        if loss > p_acc[1]:
          loss = p_acc[1] 
          best_param["degree"] = d_param
          best_param["C"] = c_param
      #print(c_param, p_acc[1])
      cross_val = np.array(cross_val)
      acc_val = np.array(acc_val)
      mse[d_param][c_param] = {"mean": np.mean(cross_val), 
                               "std": np.std(cross_val),
                               "acc": np.mean(acc_val)}

  plot(mse)
  return best_param

#def question3_2():

def main():
  train, test = read_data()
  best_param = question3_1(train, test)
  print(best_param)
  exit()
  question3_2(train, test, best_param)

if __name__ == "__main__":
  sys.excepthook = utils.colored_hook(os.path.dirname(os.path.realpath(__file__)))
  main()