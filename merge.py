import numpy as np 
import os 
from skimage import io

path3 = "./plot_question3"
def merge_q3():
  im1 = io.imread(os.path.join(path3, "degree_1.jpg"))
  im2 = io.imread(os.path.join(path3, "degree_2.jpg"))
  im3 = io.imread(os.path.join(path3, "degree_3.jpg"))
  im4 = io.imread(os.path.join(path3, "degree_4.jpg"))
  im5 = io.imread(os.path.join(path3, "degree_5.jpg"))
  h_im1 = np.concatenate([im1, im2], 1)
  h_im2 = np.concatenate([im3, im4], 1)
  h_im3 = np.concatenate([im5, 255 * np.ones_like(im1)], 1)
  im = np.concatenate([h_im1, h_im2, h_im3], 0)
  io.imsave(os.path.join(path3, "./degree.jpg"), im)

path4 = "./plot_question4"
path5 =  "./plot_question5"
def merge_q4(path):
  im1 = io.imread(os.path.join(path, "train.jpg"))
  im2 = io.imread(os.path.join(path, "test.jpg"))
  h_im1 = np.concatenate([im1, im2], 1)
  io.imsave(os.path.join(path, "./errors.jpg"), h_im1)
merge_q3()
merge_q4(path4)
merge_q4(path5)