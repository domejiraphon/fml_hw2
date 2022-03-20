import matplotlib.pyplot as plt 
#plt.rcParams['text.usetex'] = True

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
