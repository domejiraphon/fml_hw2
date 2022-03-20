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
    plt.savefig(os.path.join(folder, f"degree_{degree}.jpg"))
    plt.clf()

def plot4(train_error, test_error, degree):
  folder = "./plot_question4"
  if not os.path.exists(folder): os.mkdir(folder)
  xtick_label = [r"$%s$" %str(val) for val in degree]
  for i, data_typ in enumerate([train_error, test_error]):
    y, e, num = [], [], []
    for degree in data_typ.keys():
      all_mean = data_typ[degree]
      y.append(all_mean["mean"])
      e.append(all_mean["std"])
      num.append(all_mean["num_support_vec"])
    x = np.arange(1, len(y)+1)
    plt.errorbar(x, y, e, fmt = 'o', color = 'black', 
              linestyle='None', ecolor = 'lightblue',
              capsize=5)
    plt.ylim(min(y) - 0.05, max(y) + 0.05)
    plt.plot(x, y)
    plt.xlabel(r"Degree")
    plt.ylabel(r"Mean Square Error")
    if i == 0:
      plt.title(r"Cross-validation error for polynomial kernel degree {}".format(degree))
      im_path = f"train.jpg"
    elif i == 1:
      plt.title(r"Test error for polynomial kernel degree {}".format(degree))
      im_path = f"test.jpg"
    plt.xticks(x, xtick_label)
    plt.savefig(os.path.join(folder, im_path))
    plt.clf()


