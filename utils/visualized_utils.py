import matplotlib.pyplot as plt 
import os 
import numpy as np
plt.rcParams['text.usetex'] = True

def plot3(mean, c_range):
  folder = "./plot_question3"
  if not os.path.exists(folder): os.mkdir(folder)
  xtick_label = [r"$3^{%s}$" %(str(round(val, 3))) for val in c_range]
  for degree in mean.keys():
    all_mean = mean[degree]
    y, e = [], []
    for key, val in all_mean.items():
      y.append(val['mean'])
      e.append(val['std'])
    x = np.arange(1, len(y)+1)
    plt.errorbar(x, y, e, fmt = 'o', color = 'black', 
              linestyle='None', ecolor = 'lightblue',
              capsize=5)
    plt.ylim(min(y) - 0.05, max(y) + 0.05)
    plt.plot(x, y)
    plt.xlabel(r"C")
    plt.ylabel(r"Mean Square Error")
    plt.title(r"Cross-validation error for polynomial kernel degree {}".format(degree))
    plt.xticks(x, xtick_label)
    plt.yscale("log")
    plt.grid(True, which="both", ls="-")
    plt.savefig(os.path.join(folder, f"degree_{degree}.jpg"))
    plt.clf()

def plot4(train_error, test_error, degree):
  folder = "./plot_question4"
  if not os.path.exists(folder): os.mkdir(folder)
  xtick_label = [r"$%s$" %str(val) for val in degree]
  for i, data_typ in enumerate([train_error, test_error]):
    y, e, num, e_num = [], [], [], []
    for degree in data_typ.keys():
      all_mean = data_typ[degree]
      y.append(all_mean["mean"])
      e.append(all_mean["std"])
      if i ==0:
        num.append(all_mean["num_support_vec"])
        e_num.append(all_mean["std_support_vec"])
    x = np.arange(1, len(y)+1)
    plt.errorbar(x, y, e, fmt = 'o', color = 'black', 
              linestyle='None', ecolor = 'lightblue',
              capsize=5)
    plt.ylim(min(y) - 0.05, max(y) + 0.05)
    plt.plot(x, y)
    plt.xlabel(r"Degree")
    plt.ylabel(r"Mean Square Error")
    if i == 0:
      plt.title(r"Cross-validation error for polynomial kernel")
      im_path = f"train.jpg"
    elif i == 1:
      plt.title(r"Test error for polynomial kernel")
      im_path = f"test.jpg"
    plt.xticks(x, xtick_label)
    
    plt.yscale("log")
    plt.grid(True, which="both", ls="-")
    plt.savefig(os.path.join(folder, im_path))
    plt.clf()
    if i == 0:
      plt.errorbar(x, num, e_num, fmt = 'o', color = 'black', 
              linestyle='None', ecolor = 'lightblue',
              capsize=5)
      plt.plot(x, num)
      plt.xlabel(r"Degree")
      plt.ylabel(r"Number of support vectors")
      plt.title(r"Number of suppport vectors for polynomial kernel")
      im_path = f"support.jpg"
      plt.xticks(x, xtick_label)
      plt.savefig(os.path.join(folder, im_path))
      plt.clf()

def plot5(train_error, test_error, best_param, all_num_dataset):
  folder = "./plot_question5"
  if not os.path.exists(folder): os.mkdir(folder)
  xtick_label = [r"$%s$" %str(val) for val in all_num_dataset]
  for i, data_typ in enumerate([train_error, test_error]):
    y = []
    for num_dataset in data_typ.keys():
      all_mean = data_typ[num_dataset]
      y.append(all_mean["errors"])

    x = np.arange(1, len(y)+1)
    plt.plot(x, y)
    plt.xlabel(r"Number of training samples")
    plt.ylabel(r"Mean Square Error")
    if i == 0:
      plt.title(r"Training error for polynomial kernel when C = {}, degree = {}".format(str(best_param["C"]), str(best_param["degree"])))
      im_path = f"train.jpg"
    elif i == 1:
      plt.title(r"Test error for polynomial kernel when C = {}, degree = {}".format(str(best_param["C"]), str(best_param["degree"])))
      im_path = f"test.jpg"
    plt.xticks(x, xtick_label)
    plt.yscale("log")
    plt.grid(True, which="both", ls="-")
    plt.savefig(os.path.join(folder, im_path))
    plt.clf()

def plot6(train_error, test_error, degree):
  folder = "./plot_question6"
  if not os.path.exists(folder): os.mkdir(folder)
  xtick_label = [r"$%s$" %str(val) for val in degree]
  for i, data_typ in enumerate([train_error, test_error]):
    y, e, num, e_num = [], [], [], []
    for degree in data_typ.keys():
      all_mean = data_typ[degree]
      y.append(all_mean["loss_mean"])
      e.append(all_mean["loss_std"])

    x = np.arange(1, len(y)+1)
    plt.errorbar(x, y, e, fmt = 'o', color = 'black', 
              linestyle='None', ecolor = 'lightblue',
              capsize=5)
    
    plt.ylim(min(y) - 0.05, max(y) + 0.05)
    plt.plot(x, y)
    plt.xlabel(r"Degree")
    plt.ylabel(r"Mean Square Error")
    if i == 0:
      plt.title(r"Cross-validation error for polynomial kernel")
      im_path = f"train.jpg"
    elif i == 1:
      plt.title(r"Test error for polynomial kernel")
      im_path = f"test.jpg"

    plt.xticks(x, xtick_label)
    
    #plt.yscale("log")
    plt.grid(True, which="both", ls="-")
    plt.savefig(os.path.join(folder, im_path))
    plt.clf()

