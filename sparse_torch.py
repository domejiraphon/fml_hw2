import numpy as np
import os
import sys
import argparse
import torch
import getpass
from torch import nn

from utils import utils 
from utils import data_utils
from utils import visualized_utils 

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.1,
            help="learning rate for model parameters")
parser.add_argument('--epochs', type=int, default=150,
            help="Number of epochs")
parser.add_argument("--num_disjoint", type=int, default=5,
              help="Number of cross validation set")
args = parser.parse_args()

class Sparse_svm(nn.Module):
   ''' Model for objective function
    Args:
      shape: shape to determine the number of traning set
  '''
  def __init__(self, shape):
    '''
      alpha: tensor size [samples, 1]
      xi: tensor size [samples, 1]
      b: tensor size [1]
    '''
    super(Sparse_svm, self).__init__()
    alpha = torch.empty((shape, 1)).uniform_(0, 1)
    xi = torch.empty((shape, 1)).uniform_(0, 1)
    b = torch.empty((1)).uniform_(-1, 1)
    
    self.alpha = nn.Parameter(alpha)
    self.xi = nn.Parameter(xi)
    self.b = nn.Parameter(b)

    self.relu = nn.ReLU()

  def forward(self, features, coeff, degree, C, lambd):
    ''' Question 6
    Args:
      features: a dictionary of data composed of 
            features and label 
      coeff: coefficient in polynomial kernel
      degree: degree for polynomial kernel
      C: hyperparameters C in objective function
      lambd: hyperparameters for forcing the constraint
    Returns:
      loss value
    '''
    #x_i, x_j [n, 10]
    #y_i, y_j [n]
    x_i = x_j = features["features"]
    y_i = y_j = features["labels"]
    #alpha [n, 1], xi [n, 1]
    alpha = self.relu(self.alpha)
    xi = self.relu(self.xi)
    
    kernel = self.kernel_fn(x_i, x_j, coeff, degree)
    left_cont = y_i[:, None] * (torch.sum(alpha * y_j[:, None] * kernel, dim=1, keepdims=True) + self.b)
    right_cont = 1 - xi 

    loss = 1/2 * torch.sum(torch.abs(alpha)) + \
           C * torch.sum(xi) 
    num_force_cont = torch.sum((self.relu(- (left_cont - right_cont)) < 1e-3).float())

    const = lambd * torch.sum(self.relu(- (left_cont - right_cont)))
  
    return loss, const, num_force_cont

  def kernel_fn(self, x_i, x_j, coeff, degree):
    #x_i, x_j [n, d]
    kernel = (x_i @ x_j.T + coeff) ** degree
    
    return kernel

def inference(model, train_set, test_set, C, coeff, degree):
  ''' Inference
  Args:
    model: model to inference
    train_set: training data
    test_set: test data
    degree: degree for polynomial kernel
    coeff: coefficient in polynomial kernel
    C: hyperparameters C in objective function
    lambd: hyperparameters for forcing the constraint
  Returns:
    loss value
  '''
  x_i = train_set["features"]
  x_j = test_set["features"]
  y_i = train_set["labels"]
  y_j = test_set["labels"]
  alpha = model.relu(model.alpha)
  xi = model.relu(model.xi)

  kernel = model.kernel_fn(x_i, x_j, coeff, degree)
  
  weight = torch.sum(alpha * y_i[:, None] * kernel, dim=0, keepdims=True)
  num_neg = (y_j[None] * (weight + model.b) < 0.0).float()
  loss = torch.mean(num_neg)
 
  return loss 

def get_train_test_loss(train_set, test_set, best_param):
  ''' 
  Args:
    train_set: training data
    test_set: test data
    best_param: best C* parameters from grid search
  Returns:
    plot graph and store it in ./plot_question6
  '''
  dataloader = data_utils.split_data(train_set, args.num_disjoint, tensor=True)
  degree = [1, 2, 3, 4, 5]
  train_mse = {}
  test_mse = {}
  for degree_param in degree:
    cross_val, test_val = [], []
    for split_train in dataloader:
      train_data = split_train["train"]
      val_data = split_train["val"]
      model = Sparse_svm(shape=train_data["features"].shape[0]).cuda()
      optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
      for i in range(args.epochs):
        optimizer.zero_grad()
        loss, const, num_force_cont = model(train_data, 
                coeff=1, degree=degree_param, C=best_param["C"], lambd=100)
        loss.backward()
        optimizer.step()
      with torch.no_grad():
        cross_loss = inference(model, train_data, val_data, C=best_param["C"],
                        coeff=1, degree=degree_param)
        test_loss = inference(model, train_data, test_set, C=best_param["C"],
                        coeff=1, degree=degree_param)
      cross_val.append(cross_loss)
      test_val.append(test_loss)
    
    cross_val = torch.tensor(cross_val)
    test_val = torch.tensor(test_val)
    train_mse[degree_param] = {"loss_mean": torch.mean(cross_val),
                    "loss_std": torch.std(cross_val),}

    test_mse[degree_param] = {"loss_mean": torch.mean(test_val),
                    "loss_std": torch.std(test_val)}

  visualized_utils.plot6(train_mse, test_mse, degree)
  exit()

def train(train_set, test_set):
  ''' Train
  Args:
    train_set: training data
    test_set: test data
  Returns:
    best parameters
  '''
  torch.manual_seed(0)
  
  dataloader = data_utils.split_data(train_set, args.num_disjoint, tensor=True)
  degree = [1, 2, 3, 4, 5]
  c_range = np.linspace(-3, 3, 4)
  C = 3** c_range
  best_param = {"degree": None, "C": None, "loss": np.inf}
  
  for degree_param in degree:
    for c_param in C:
      f_loss = []
      for split_train in dataloader:
        train_data = split_train["train"]
        val_data = split_train["val"]
        model = Sparse_svm(shape=train_data["features"].shape[0]).cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        for i in range(args.epochs):
          optimizer.zero_grad()
         
          loss, const, num_force_cont = model(train_data, 
                  coeff=1, degree=degree_param, C=c_param, lambd=10)
          all_loss = loss + const
          
          all_loss.backward()
          optimizer.step()

        f_loss.append(loss)
      f_loss = torch.tensor(f_loss)
      if torch.mean(f_loss) < best_param["loss"]:
        best_param["degree"] = degree_param
        best_param["C"] = c_param
        best_param["loss"] = torch.mean(f_loss) 
        best_param["num_constraint"] = num_force_cont
  print(best_param)
  
  get_train_test_loss(train_set, test_set, best_param)
  exit()
def np2tensor(dataloader):
  return {"features": torch.from_numpy(dataloader["features"].astype(np.float32)).float().to(args.device),
          "labels": torch.from_numpy(dataloader["labels"].astype(np.float32)).float().to(args.device)}
def main():
  args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  train_dataloader, test_dataloader = data_utils.read_data()
  
  train(train_dataloader, np2tensor(test_dataloader))
  

if __name__ == "__main__":
  sys.excepthook = utils.colored_hook(os.path.dirname(os.path.realpath(__file__)))
  main()