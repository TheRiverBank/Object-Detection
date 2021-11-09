import os

from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10


data_path = './data'
labels = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
img_size = (32, 32, 3)

def load_data():
    (X_train, Y_train), (x_test, y_test) = cifar10.load_data()
    X_train = X_train.reshape(-1, 32, 32, 3)
    X_train = X_train.astype('float32')
    x_test = x_test.astype('float32')
    X_train = X_train / 255
    print(X_train)
    x_test = x_test / 255


    return X_train, Y_train, x_test, y_test

def resize_img(img):
    img_resized = cv2.resize(img, dsize=(img_size[0], img_size[1]), interpolation=cv2.INTER_CUBIC)
    img_resized = img_resized.reshape(-1, 32, 32, 3)
    #img_resized = cv2.blur(img_resized, (3,3), 0)
    return img_resized

