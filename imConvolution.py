## Hello to everybody, I am starting to learn tensorFlow.
# this code is not mine, I am using only for learning.
# If you want information or wanna get the original code
# please feel free to visit this site
# https://github.com/pkmital/CADL
# the dataset which i am using on this project is CelebNet
# Cheers xavysp

import sys
import os
import numpy as np
import matplotlib
matplotlib.use('GTKAgg')  #('Qt4Agg')  #('GTKAgg')
import matplotlib.pyplot as plt
from skimage.transform import resize
import tensorflow as tf
from time import sleep



def visualization(imgs, saveas='dataset'):

    if isinstance(imgs, list):  # isinstance checking a variable
        imgs = np.array(imgs)   # if it is on concordance with its type
    imgHeight = imgs.shape[1]
    imgWidth = imgs.shape[2]
    nPlots = int(np.ceil(np.sqrt(imgs.shape[0])))

    if len(imgs.shape)==4 and imgs.shape[3]==3:
        m = np.ones((imgs.shape[1]*nPlots+nPlots+1,
                    imgs.shape[2]*nPlots+nPlots+1, 3))*0.5
    else:
        m= np.ones((imgs.shape[1]*nPlots+nPlots+1,
                    imgs.shape[2]*nPlots+nPlots+1))*0.5
    for i in range(nPlots):
        for j in range(nPlots):
            this_filter = i * nPlots+j
            if this_filter < imgs.shape[0]:
                this_img = imgs[this_filter]
                # here start the hard work
                m[1+i+i*imgHeight:1+i+(i+1)*imgHeight,
                1+j+j*imgWidth:1+j+(j+1)*imgWidth]= this_img
    plt.imsave(arr=m, fname=saveas)
    return m


def imCrop(img):

    size = np.min(img.shape[:2])
    extra = img.shape[:2]-size
    crop = img
    for i in np.flatnonzero(extra):
        crop = np.take(crop, extra[i] // 2 + np.r_[:size], axis=i)
    return crop


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
print(len(imgs))
im = imgs
# we are working with images its size are 100*100*3
# so if images are greater that mentioned the size is cropped
#  this is not our case so y letting below code on comments
imgs = [imCrop(i) for i in imgs]  # turning in square image

imgs = [resize(i,(100, 100)) for i in imgs]

imgs = np.array(imgs).astype(np.float32)
print(imgs.shape)
# plt.figure(figsize=(10,10))
# plt.imshow(im[1])
# plt.show()
# sleep(2)
# plt.close()
# Now we are ready to see (before convolution) all of our image data
assert (imgs.shape == (100, 100, 100, 3))
# plt.figure(figsize=(10, 10))
plt.imshow(visualization(imgs, saveas='dataset.jpg'))
plt.show()
sleep(2)
plt.close(0)

#  right now we are starting using tensorflow ********
sess = tf.Session()

# ***** calculing the mean of all images

meanImg_op = tf.reduce_mean(imgs,0)
print(meanImg_op.get_shape().as_list())
meanImg = sess.run(meanImg_op)
# now plotting the resulting images

assert(meanImg.shape ==(100,100,3))
plt.figure(figsize=(10,10))
plt.imshow(meanImg)
plt.show()
plt.imsave(arr=meanImg, fname='mean.png')

# ***** calculing the deviation standard of all images
meanImg_4d = tf.reduce_mean(imgs, reduction_indices=0, keep_dims=True)
subtracted = imgs-meanImg_4d
stdImg_op = tf.sqrt(tf.reduce_mean(subtracted*subtracted,reduction_indices=0))
stdImg = sess.run(stdImg_op)

# Ok lets look at the standard deviation visualy
assert (stdImg.shape == (100,100) or stdImg.shape==(100,100,3))
plt.figure(figsize=(10, 10))
stdImg_show = stdImg/np.max(stdImg)
plt.imshow(stdImg_show)
plt.show()
plt.imsave(arr=stdImg_show, fname='std.png')

# now we are going to normalize the dataset
#******************
normImgs_op = subtracted/stdImg
normImgs = sess.run(normImgs_op)

print(np.min(normImgs), np.max(normImgs))
print('*******')
print(imgs.dtype)
# time to see is going on

assert (normImgs.shape == (100, 100, 100, 3))
plt.figure(figsize=(10,10))
plt.imshow(visualization(normImgs, saveas='normalized.jpg'))
plt.show()
# ok now lets look like better
normImgs_show = (normImgs - np.min(normImgs))/ (np.max(normImgs) - np.min(normImgs))
plt.figure(figsize=(10, 10))
plt.imshow(visualization(normImgs_show, saveas='normalized-visual.png'))
plt.show()

# ***********The most import step Convolving dataset *******
ksize = 32
g = tf.Graph()
with tf.Session(graph=g):
    mean = 0.0
    sigma = 1.0
    x = tf.linspace(-3.0, 3.0, ksize)
    z = (tf.exp(tf.neg(tf.pow(x-mean, 23.0) /
                       (2.0 * tf.pow(sigma, 2.0)))) *
         (1.0 /(sigma * tf.sqrt(2.0*3.1415))))

    z_2d = tf.matmul(tf.reshape(z,[ksize, 1]), tf.reshape(z,[1, ksize]))

    ones = tf.ones((1, ksize))
    ys = tf.sin(x)
    ys = tf.reshape(ys, [ksize, 1])
    wave = tf.matmul(ys, ones)
    gabor = tf.mul(wave, z_2d)
    gaborVal = gabor.eval()
    kernel = np.concatenate([gaborVal[:,:,np.newaxis] for i in range(3)], axis=2)

