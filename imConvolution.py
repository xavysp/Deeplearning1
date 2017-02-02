## Hello to everybody, I am starting to learn tensorFlow.
# this code is not mine, I am using only for learning.
# If you want information or wanna get the original code
# please feel free to visit this site
# https://github.com/pkmital/CADL
# the dataset which i am using on this project is CelebNet
# Cheers xavysp
''' import matplotlib
matplotlib.use('GTKAgg')'''

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
import tensorflow as tf


# starting loading to dataset
dirname = 'dataset'

filenames = [os.path.join(dirname, fname)
             for fname in os.listdir(dirname)]

filenames =  filenames[:100]
assert (len(filenames) == 100)
# Now we are ready to start reading weach of 100 images in
#  a numpy array

imgs = [plt.imread(fname)[..., :3]
        for fname in filenames]
# we are working with images its size are 100*100*3
# so if images are greater that mentioned the size is cropped
#  this is not our case so y letting below code on comments

imgs = np.array(imgs).astype(np.float32)

