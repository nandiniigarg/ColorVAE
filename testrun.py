# %matplotlib inline
import matplotlib.pyplot as plt
import cv2
import warnings
warnings.filterwarnings('ignore')
import os
from tensorflow.keras.utils import img_to_array
from tqdm import tqdm
import numpy as np
import pandas as pd
from keras.applications.inception_resnet_v2 import InceptionResNetV2, decode_predictions, preprocess_input
from keras.models import Model, Sequential
from keras.layers import Conv2D, Conv2DTranspose, Reshape, Input, LeakyReLU, BatchNormalization, UpSampling2D, MaxPooling2D
from keras.layers import Concatenate
from keras.optimizers import Adam
import tensorflow as tf
import tensorflow_model_optimization as tfmot
# import tempfile

#
import keras
from keras import layers
SIZE = 160
color_img = []
path = 'landscape Images/color'
files = os.listdir(path)
for i in tqdm(files):
    img = cv2.imread(path + '/'+i,1)
    # open cv reads images in BGR format so we have to convert it to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #resizing image
    img = cv2.resize(img, (SIZE, SIZE))
    img = img.astype('float32') / 255.0
    color_img.append(img_to_array(img))


gray_img = []
path = 'landscape Images/gray'
files = os.listdir(path)
for i in tqdm(files):

    img = cv2.imread(path + '/'+i,1)
    #resizing image
    img = cv2.resize(img, (SIZE, SIZE))
    img = img.astype('float32') / 255.0
    gray_img.append(img_to_array(img))

from sklearn.utils import resample
gray_img, color_img = resample(gray_img, color_img)

train_gray_image = gray_img[:5500]
train_color_image = color_img[:5500]

test_gray_image = gray_img[5500:]
test_color_image = color_img[5500:]
# reshaping
train_g = np.reshape(train_gray_image,(len(train_gray_image),SIZE,SIZE,3))
train_c = np.reshape(train_color_image, (len(train_color_image),SIZE,SIZE,3))
# print('Train color image shape:',train_c.shape)


test_gray_image = np.reshape(test_gray_image,(len(test_gray_image),SIZE,SIZE,3))
test_color_image = np.reshape(test_color_image, (len(test_color_image),SIZE,SIZE,3))
# print('Test color image shape',test_color_image.shape)

def encoder(filters , kernel_size, apply_batch_normalization = True):
    downsample = tf.keras.models.Sequential()
    downsample.add(layers.Conv2D(filters,kernel_size,padding = 'same', strides = 2))
    if apply_batch_normalization:
        downsample.add(layers.BatchNormalization())
    downsample.add(keras.layers.LeakyReLU())
    return downsample
def decoder(filters, kernel_size, dropout = False):
    upsample = tf.keras.models.Sequential()
    upsample.add(layers.Conv2DTranspose(filters, kernel_size,padding = 'same', strides = 2))
    if dropout:
        upsample.dropout(0.2)
    upsample.add(keras.layers.LeakyReLU())
    return upsample


def model():
    # downsampling

    inputs = layers.Input(shape=[160, 160, 3])
    d1 = encoder(128, (3, 3), False)(inputs)
    d2 = encoder(128, (3, 3), False)(d1)
    d3 = encoder(256, (3, 3), True)(d2)
    d4 = encoder(512, (3, 3), True)(d3)
    d5 = encoder(512, (3, 3), True)(d4)

    # upsampling

    u1 = decoder(512, (3, 3), False)(d5)
    u1 = layers.concatenate([u1, d4])
    u2 = decoder(256, (3, 3), False)(u1)
    u2 = layers.concatenate([u2, d3])
    u3 = decoder(128, (3, 3), False)(u2)
    u3 = layers.concatenate([u3, d2])
    u4 = decoder(128, (3, 3), False)(u3)
    u4 = layers.concatenate([u4, d1])
    u5 = decoder(3, (3, 3), False)(u4)
    u5 = layers.concatenate([u5, inputs])
    output = layers.Conv2D(3, (2, 2), strides=1, padding='same')(u5)
    return tf.keras.Model(inputs=inputs, outputs=output)

model = model()
# model = tf.load_model("initial_model")
model.load_weights('model_weights.h5')

def plot_images(color, grayscale,predicted):
    plt.figure(figsize=(15,15))
    plt.subplot(1,3,1)
    plt.title('Color Image', color = 'green', fontsize = 20)
    plt.imshow(color)
    plt.subplot(1,3,2)
    plt.title('Grayscale Image ', color = 'black', fontsize = 20)
    plt.imshow(grayscale)
    plt.subplot(1,3,3)
    plt.title('Predicted Image ', color = 'Red', fontsize = 20)
    plt.imshow(predicted)

    plt.show()
#
# showimg_c = cv2.imread("img3_c.jpg")
# img_c = cv2.cvtColor(img_c, cv2.COLOR_BGR2RGB)
# img_c = cv2.resize(img_c, (SIZE, SIZE))
# img_c = img.astype('float32')
# img_c = img_to_array(img_c)
# img_g = cv2.imread("img3_g.jpeg")
# img_g = cv2.resize(img_g, (SIZE, SIZE))
# img_g = img.astype('float32')
# img_g = img_to_array(img_g)
# predicted = np.clip(model.predict(img_g.reshape(1,SIZE, SIZE,3)),0.0,1.0).reshape(SIZE, SIZE,3)
# plot_images(img_g,predicted)
for i in range(50,58):
    predicted = np.clip(model.predict(test_gray_image[i].reshape(1,SIZE, SIZE,3)),0.0,1.0).reshape(SIZE, SIZE,3)
    plot_images(test_color_image[i],test_gray_image[i],predicted)