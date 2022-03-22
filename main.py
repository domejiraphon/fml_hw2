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

def question3(train, test):
  #global c_range
  c_range = np.linspace(-5, 20, 10)
  c_range = np.linspace(-5, 5, 5)
  #c_range = np.linspace(-5, 5, 2)
  C = 3** c_range
  
  #print(C)
  degree = [1, 2, 3, 4, 5]
  
  #degree = [2]
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
        val_data = split_train["val"]
        options = f"-c {c_param} -d {d_param} -t 1 -q"
        m = svm_train(train_data["labels"], train_data["features"], options)
        p_label, p_acc, p_val = svm_predict(val_data["labels"], val_data['features'], m, "-q")
        cross_val.append(p_acc[1])
        acc_val.append(p_acc[0])
        
      #print(c_param, p_acc[1])
      cross_val = np.array(cross_val)
      acc_val = np.array(acc_val)
      if loss > np.mean(cross_val):
        loss = np.mean(cross_val)
        best_param["degree"] = d_param
        best_param["C"] = c_param
      mse[d_param][c_param] = {"mean": np.mean(cross_val), 
                               "std": np.std(cross_val),
                               "acc": np.mean(acc_val)}

  visualized_utils.plot3(mse, c_range)
  return best_param

def question4(train, test, best_param):
  best_c = best_param['C']
  degree = range(1, 11)
  #degree = [1, 2]
  train_mse, test_mse = {}, {}
  dataloader = data_utils.split_data(train, args.num_disjoint)
  loss = np.inf
  best_param = {"degree": None, "C": None}
  for d_param in degree:
    train_mse[d_param], test_mse[d_param] = {}, {}
    cross_val, test_val = [], []
    acc_val, acc_test = [], []
    num_support_vec = []
    for split_train in dataloader:
      train_data = split_train["train"]
      val_data = split_train["val"]
      options = f"-r {best_c} -d {d_param} -t 1 -q"

      m = svm_train(train_data["labels"], train_data["features"], options)

      p_label, p_acc, p_val = svm_predict(val_data["labels"], val_data['features'], m, "-q")
      cross_val.append(p_acc[1])
      acc_val.append(p_acc[0])

      p_label, p_acc, p_val = svm_predict(test["labels"], test['features'], m, "-q")
      test_val.append(p_acc[1])
      acc_test.append(p_acc[0])

      
      _, _, p_val = svm_predict(train_data["labels"], train_data['features'], m, "-q")
      p_val = np.array(p_val)
      num_support_vec.append(np.sum((np.abs(np.abs(p_val) - 1.0) < 1e-3).astype(np.int64)))
      
    cross_val = np.array(cross_val)
    test_val = np.array(test_val)
    acc_val = np.array(acc_val)
    acc_test = np.array(acc_test)
    num_support_vec = np.array(num_support_vec)
    if loss > np.mean(cross_val):
      loss = np.mean(cross_val) 
      best_param["degree"] = d_param
      best_param["C"] = best_c
    train_mse[d_param] = {"mean": np.mean(cross_val), 
                    "std": np.std(cross_val),
                    "acc": np.mean(acc_val),
                    "num_support_vec": np.mean(num_support_vec),
                    "std_support_vec": np.std(num_support_vec)}
    test_mse[d_param] = {"mean": np.mean(test_val), 
                    "std": np.std(test_val),
                    "acc": np.mean(acc_test)}

  visualized_utils.plot4(train_mse, test_mse, degree)
  return best_param
  
def question5(train, test, best_param):
  all_num_dataset = range(train["features"].shape[1], train["features"].shape[0], 300)
  train_mse, test_mse = {}, {}
  for num_dataset in all_num_dataset:
    train_data = data_utils.filter_data(train, num_dataset=num_dataset)
    options = f"-r {best_param['C']} -d {best_param['degree']} -t 1 -q"
    m = svm_train(train_data["labels"], train_data["features"], options)
    _, p_acc, _ = svm_predict(train_data["labels"], train_data['features'], m, "-q")

    train_mse[num_dataset] = {"errors": p_acc[1], 
                              "acc": p_acc[0]}

    _, p_acc, _ = svm_predict(test["labels"], test['features'], m, "-q")
    test_mse[num_dataset] = {"errors": p_acc[1], 
                              "acc": p_acc[0]}
  
  visualized_utils.plot5(train_mse, test_mse, best_param, all_num_dataset)

def main():
  train, test = data_utils.read_data()
  best_param = question3(train, test)
  print(f"q3 {best_param}")

  best_param = question4(train, test, best_param)
  print(f"q4 {best_param}")
  exit()
  question5(train, test, best_param)

if __name__ == "__main__":
  sys.excepthook = utils.colored_hook(os.path.dirname(os.path.realpath(__file__)))
  main()