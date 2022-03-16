import pandas as pd
import numpy as np 
from libsvm.svmutil import *

def read_data():
  # reading csv files
  data =  pd.read_csv('./data/abalone.data').to_numpy()
  gender2vec = 1.0 * (data[:, :1] == "M").astype(np.float32) + \
               -1.0 * (data[:, :1] == "F").astype(np.float32)
  data = np.concatenate([gender2vec, data[:, 1:]], 1)

  label = ((data[:, -1] >= 1) | (data[:, -1] <= 9)).astype(np.float32) + \
          -1.0 * (data[:, -1] > 9).astype(np.float32)
  train = {"features": data[:3133, :-1],
           "label": label[:3133]}
  test = {"features": data[3133:, :-1],
           "label": label[3133:]}

  return train, test 
train, test = read_data()

m = svm_train(train['label'], train['features'], '-c 4')
print(m)
exit()