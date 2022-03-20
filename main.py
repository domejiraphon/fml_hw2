import pandas as pd
import numpy as np 
from libsvm.svmutil import *
import os
import sys
import argparse

from utils import utils 
from utils import data_utils
from utils import visualized_utils 


parser = argparse.ArgumentParser()
parser.add_argument("--num_disjoint", type=int, default=5,
              help="Number of epoch to train")
args = parser.parse_args()

def plot3(mean):
  for degree in mean.keys():
    all_mean = mean[degree]
    x, y, e = [], [], []
    i = 1
    for key, val in all_mean.items():
      #x.append(key)
      x.append(i)
      y.append(val['mean'])
      e.append(val['std'])
      i += 1
    plt.errorbar(x, y, e, fmt = 'o', color = 'black', 
              linestyle='None', ecolor = 'lightblue',
              capsize=5)
    plt.ylim(min(y) - 0.05, max(y) + 0.05)
    plt.xlabel("C")
    plt.ylabel("Mean Square Error")
    plt.title(f"Cross-validation error for polynomial kernel degree {degree}")
    #plt.yscale("log")
    val = []
    for x_vis in all_mean.keys():
      if x_vis < 1.0:
        val.append(round(x_vis, 3))
      else:
        val.append(round(x_vis))

    plt.xticks(x, val)
    plt.savefig(f"./{degree}.jpg")
    plt.clf()

def question4(train, test, best_param):
  degree = range(1, 4)
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
      options = f"-r {best_param['C']} -d {d_param} -t 1 -q"
      m = svm_train(train_data["labels"], train_data["features"], options)
      p_label, p_acc, p_val = svm_predict(test_data["labels"], test_data['features'], m, "-q")
      cross_val.append(p_acc[1])
      acc_val.append(p_acc[0])
      if loss > p_acc[1]:
        loss = p_acc[1] 
        best_param["degree"] = d_param
        best_param["C"] = c_param

      _, _, p_val = svm_predict(train_data["labels"], train_data['features'], m, "-q")
      p_val = np.array(p_val)
      
      num_support_vec.append(np.sum((np.abs(np.abs(p_val) - 1.0) < 1e-3).astype(np.int64)))
      

    #print(c_param, p_acc[1])
    cross_val = np.array(cross_val)
    acc_val = np.array(acc_val)
    mse[d_param] = {"mean": np.mean(cross_val), 
                    "std": np.std(cross_val),
                    "acc": np.mean(acc_val),
                    "num_support_vec": np.mean(num_support_vec)}
    print(np.mean(num_support_vec))
  print(best_param)
  print('domme')
  exit()

def question3(train, test):
  #global c_range
  #c_range = np.linspace(-5, 20, 8)
  c_range = np.linspace(-5, 5, 2)
  #C = 3** np.linspace(2, 15, 10) 
  #C = 3** np.linspace(-5, 20, 5)
  C = 3** c_range
  
  #print(C)
  #degree = [1, 2, 3, 4, 5]
  degree = [2]
  #degree = [1, 2]
  mse = {}
  dataloader = data_utils.split_data(train, args.num_disjoint)
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

  visualized_utils.plot3(mse)
  return best_param

#def question3_2():

def main():
  train, test = data_utils.read_data()
  best_param = question3(train, test)
  question3_2(train, test, best_param)

if __name__ == "__main__":
  sys.excepthook = utils.colored_hook(os.path.dirname(os.path.realpath(__file__)))
  main()