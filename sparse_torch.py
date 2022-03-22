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
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('-epochs', type=int, default=150)
args = parser.parse_args()

class Sparse_svm2(nn.Module):
  def __init__(self, shape):
    super(Sparse_svm, self).__init__()
    alpha = torch.empty((shape, 1)).uniform_(0, 1)
    xi = torch.empty((shape, 1)).uniform_(0, 1)
    b = torch.empty((1)).uniform_(-1, 1)
    
    self.alpha = nn.Parameter(alpha)
    self.xi = nn.Parameter(xi)
    self.b = nn.Parameter(b)

    self.relu = nn.ReLU()

  def forward(self, features, c, degree, C, lambd):
    x_i = features["features"]
    y_i = features["labels"]

    self.alpha = self.relu(self.alpha)
    self.xi = self.relu(self.xi)
    kernel = kernel_fn(x_i, x_j, coeff, degree)
    left_cont = y_i[:, None] * (torch.sum(self.alpha * y_j[:, None] * kernel, dim=1, keepdims=True) + self.b)
    right_cont = 1 - self.xi 
    loss = 1/2 * torch.sum(torch.abs(self.alpha)) + \
           C * torch.sum(self.xi) + \
           lambd * torch.mean(left_cont - right_cont)
    return loss 

  def kernel_fn(self, x_i, x_j, coeff, degree):
    #x_i, x_j [n, d]
    kernel = (x_i @ x_j.T + coeff) ** degree
    
    return kernel

class Sparse_svm(nn.Module):
  def __init__(self, shape):
    super(Sparse_svm, self).__init__()
    alpha = torch.empty((shape, 1)).uniform_(0, 1)
    xi = torch.empty((shape, 1)).uniform_(0, 1)
    b = torch.empty((1)).uniform_(-1, 1)
    
    self.alpha = nn.Parameter(alpha)
    self.xi = nn.Parameter(xi)
    self.b = nn.Parameter(b)

    self.relu = nn.ReLU()

  def forward(self, features, c, degree, C, lambd):
    x_i = features["features"]
    y_i = features["labels"]

    alpha = self.relu(self.alpha)
    print(alpha)
    exit()
    self.xi = self.relu(self.xi)
    kernel = kernel_fn(x_i, x_j, coeff, degree)
    left_cont = y_i[:, None] * (torch.sum(self.alpha * y_j[:, None] * kernel, dim=1, keepdims=True) + self.b)
    right_cont = 1 - self.xi 
    loss = 1/2 * torch.sum(torch.abs(self.alpha)) + \
           C * torch.sum(self.xi) + \
           lambd * torch.mean(left_cont - right_cont)
    return loss 

  def kernel_fn(self, x_i, x_j, coeff, degree):
    #x_i, x_j [n, d]
    kernel = (x_i @ x_j.T + coeff) ** degree
    
    return kernel

def train(train, test):
  model = Sparse_svm(shape=train["features"].shape[0]).cuda()
  optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
  for i in range(args.epochs):
    optimizer.zero_grad()
    loss = model(train, c=1, degree=2, C=1, lambd=1)
    print(f"Loss: {loss}")
    loss.backward()
    optimizer.step()
  exit()

def main():
  train_dataloader, test_dataloader = data_utils.read_data()
  train_dataloader = {"features": train_dataloader["features"][:2],
            "labels": train_dataloader["labels"][:2]}
  train(train_dataloader, test_dataloader)
  

if __name__ == "__main__":
  sys.excepthook = utils.colored_hook(os.path.dirname(os.path.realpath(__file__)))
  main()