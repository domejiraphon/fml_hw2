from scipy.optimize import LinearConstraint
from scipy.optimize import minimize
import pandas as pd
import numpy as np 
from libsvm.svmutil import *
import os
import sys
import argparse

from utils import utils 
from utils import data_utils
from utils import visualized_utils 

def kernel_fn(x_i, x_j, c, d):
  #x_i, x_j [n, d]
  kernel = (x_i @ x_j.T + c)**d 
  
  return kernel

def first_constraint(x_i, y_i, x_j, y_j, c, d):
  mat1 = - y_i[:, None] * y_j[:, None] * kernel_fn(x_i, x_j, c, d)
  mat2 = - np.eye(x_i.shape[0])
  mat3 = - y_i[:, None]
  
  mat = np.concatenate([mat1, mat2, mat3], 1)
  lower_bound = - np.inf * np.ones((x_i.shape[0]))
  upper_bound = - np.ones((x_i.shape[0]))
  print(mat)
  print('/n')
  print(lower_bound)
  print(upper_bound)
  exit()
  return mat, lower_bound, upper_bound

def second_and_third_constraint(x_i):
  num_param = 2 * x_i.shape[0] + 1
  num_constraint = 2 * x_i.shape[0]
  mat = np.zeros((num_constraint, num_param))
  mat[:x_i.shape[0], :x_i.shape[0]] = - np.eye(x_i.shape[0])
  mat[x_i.shape[0]: 2 * x_i.shape[0], x_i.shape[0]: 2 * x_i.shape[0]] = - np.eye(x_i.shape[0])

  lower_bound = - np.inf * np.ones((num_constraint))
  upper_bound = np.zeros((num_constraint))
  
  return mat, lower_bound, upper_bound

def combine_constraint(x_i, y_i, x_j, y_j, c, d):
  #f_mat [m, 2m + 1], f_lower, s_upper [m]
  f_mat, f_lower_bound, f_upper_bound = first_constraint(x_i, y_i, x_j, y_j, c, d)
  #s_mat [2m, 2m+1], s_lower, s_upper [2m]
  s_mat, s_lower_bound, s_upper_bound = second_and_third_constraint(x_i)
  
  mat = np.concatenate([f_mat, s_mat], 0)
  lower = np.concatenate([f_lower_bound, s_lower_bound], 0)
  upper = np.concatenate([f_upper_bound, s_upper_bound], 0)
  return mat, lower, upper 

def objective(param):
  alpha = param[:m]
  xi = param[m:2*m]
  b = param[-1]
  obj = 1/2* np.sum(np.abs(alpha)) + \
        C * np.sum(xi)
  return obj 

def grad(param):
  alpha = param[:m]
  xi = param[m:2*m]
  b = param[-1]
  grad_alpha = 1/2 * np.abs(alpha) / (alpha + 1e-12)
  grad_epsilon = C * np.ones_like(xi)
  grad_b = np.zeros((1))
  
  grad = np.concatenate([grad_alpha,
                        grad_epsilon,
                        grad_b], 0)
  
  return grad 

def sparse_svm(train, test):
  global m, C, degree
  c = 1
  C = 1
  degree = 2
  m = train["features"].shape[0]
  x0 = np.random.rand(2 * train["features"].shape[0] + 1)
  
  #cont [3m, 2m+1], lower [3m], upper [3m]
  cont, lower, upper = combine_constraint(x_i=train["features"],
                              y_i=train["labels"],
                              x_j=train["features"],
                              y_j=train["labels"], c=c, d=degree)

  linear_constraint = LinearConstraint(cont, lower, upper)
  res = minimize(objective, x0, method='trust-constr', jac=grad, 
                constraints=[linear_constraint],
                options={'verbose': 1})
  print(res.x)

def main():
  train, test = data_utils.read_data()
  train = {"features": train["features"][:2],
            "labels": train["labels"][:2]}
  sparse_svm(train, test)
  

if __name__ == "__main__":
  sys.excepthook = utils.colored_hook(os.path.dirname(os.path.realpath(__file__)))
  main()