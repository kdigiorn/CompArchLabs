#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 14:23:05 2018

Flowers database - csv file to classify images from:
https://www.kaggle.com/olgabelitskaya/the-dataset-of-flower-images

"""

import numpy as np 
import pandas as pd 

from PIL import ImageFile
from tqdm import tqdm

# import matplotlib.pylab as plt

from keras.preprocessing import image as keras_image
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D

#Functions
    
def path_to_tensor(img_path):
    img = keras_image.load_img("flower_images/"+img_path, target_size=(128, 128))
    x = keras_image.img_to_array(img)
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)

ImageFile.LOAD_TRUNCATED_IMAGES = True 


#Explore data

flowers = pd.read_csv("flower_images/flower_labels.csv")
flowers_dict = {  # key value store of flower classes
                0:'phlox', 1:'rose', 2:'calendula', 3:'iris', 4:'leucanthemum maximum',
                5:'bellflower', 6:'viola', 7:'rudbeckia', 8:'peony', 9:'aquilegia'}
flower_files = flowers['file']
flower_targets = flowers['label'].values

flower_tensors = paths_to_tensor(flower_files)
x_train, x_test, y_train, y_test = train_test_split(flower_tensors, flower_targets, 
                                                    test_size = 0.2, random_state = 1)

n = int(len(x_test)/2)
x_valid, y_valid = x_test[:n], y_test[:n]
x_test, y_test = x_test[n:], y_test[n:]
x_train.shape, x_test.shape, x_valid.shape, y_train.shape, y_test.shape, y_valid.shape

# Labeled example
# print('\nLabel: {} - {}'.format(y_train[1], flowers_dict[y_train[1]]))
# plt.figure(figsize=(3,3))
# plt.imshow((x_train[1]/255).reshape(128,128,3))

# Prepare Data
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255
x_valid = x_valid.astype('float32')/255

c_y_train = to_categorical(y_train, 10)
c_y_test = to_categorical(y_test, 10)
c_y_valid = to_categorical(y_valid, 10)

# CNN

def cnn_model():
    model = Sequential()

    model.add(Conv2D(32, (5, 5), padding='same', input_shape=x_train.shape[1:]))
    model.add(Activation('relu'))
    
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(96, (5, 5)))
    model.add(Activation('relu'))
    
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
#   model.add(Flatten())
    model.add(GlobalAveragePooling2D())
    
    model.add(Dense(512, activation='tanh'))
    model.add(Dropout(0.25)) 
    
    model.add(Dense(128, activation='tanh'))
    model.add(Dropout(0.25)) 

    model.add(Dense(10))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=['accuracy'])
    
    return model

cnn_model = cnn_model()

cnn_history = cnn_model.fit(x_train, c_y_train, 
                                 epochs=50, batch_size=64, verbose=2,
                                 validation_data=(x_valid, c_y_valid))

cnn_model.save('flower_model_cnn_50epochs.h5')