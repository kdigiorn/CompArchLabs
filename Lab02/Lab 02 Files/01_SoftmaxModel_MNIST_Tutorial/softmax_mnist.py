#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 09:29:16 2018

@author: mniemier
"""

###################################################################################################
# Import relevant Keras items, numpy, matplotlib
###################################################################################################
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
import numpy as np
import matplotlib.pyplot as plt

###################################################################################################
# load (downloaded if needed) the MNIST dataset
# Assign training data and labels + tesing data and labels to X_train, y_train + X_test, y_test
###################################################################################################
(X_train, y_train), (X_test, y_test) = mnist.load_data()

###################################################################################################
# If you want to visualize MNIST images, uncomment this code...
###################################################################################################
# plt.subplot(221)
# plt.imshow(X_train[0], cmap=plt.get_cmap('gray'))
# plt.subplot(222)
# plt.imshow(X_train[1], cmap=plt.get_cmap('gray'))
# plt.subplot(223)
# plt.imshow(X_train[2], cmap=plt.get_cmap('gray'))
# plt.subplot(224)
# plt.imshow(X_train[3], cmap=plt.get_cmap('gray'))
# plt.show()

###################################################################################################
# Calculate the number of pixels in each image -- i.e., the image height * the image width
#
# Because we are working with a multi-layer perceptron network (i.e., a single layer of input
# neurons), we want to create a single, 1D vector out of each image.
# --> The .reshape function does this; note that X_train.shape[0] is simply the # of images in
#     the train and test set (in this example, 60000)
# --> Each training / testing image is reshaped to a 784 x 1 pixel vector
###################################################################################################
num_pixels = X_train.shape[1] * X_train.shape[2]
X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')

###################################################################################################
# Input pixel values are normalized from 0-255 to 0-1 -- this is common...
###################################################################################################
X_train = X_train / 255
X_test = X_test / 255

###################################################################################################
# Here, we create a 1 hot encoding of our labels -- i.e., the to_categorial command
# converts a class vector (integers) to binary class matrix.
#
# We also record the nubmer of classes
###################################################################################################
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

###################################################################################################
# This code is used to define a network model
# The simplest type of model is a sequential model
#
# The model needs to know what input shape it should expect. For this reason, the first layer 
# in a Sequential model (and only the first, because following layers can do automatic shape 
# inference) needs to receive information about its input shape. 
#
# See more at:  https://keras.io/getting-started/sequential-model-guide/
###################################################################################################
model = Sequential()
model.add(Dense(10, input_dim=784, kernel_initializer='normal', activation='softmax'))

###################################################################################################
# This code is used to summarize a network model (see console output)
###################################################################################################
model.summary()


###################################################################################################
# Before training a model, you need to configure the learning process, which is done via the 
# compile method. It receives three arguments:
#  --> An Optimizer
#  --> A loss function
#  --> A list of metrics
#
#  These will be given to you, but the optimizer is almost always cross_entropy, and for a 
#  classification problem, the metric is accuracy...  Note that if you only have two classes, you
#  might employ "binary_crossentroy", while if there are more than 2 classes, you can use 
#  "categorical_crossentropy" -- again, the training detail is abstracted away here...
#
# See more at: https://keras.io/getting-started/sequential-model-guide/#compilation
###################################################################################################
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

###################################################################################################
# This is the code that actually trains the network
# Note that inputs are 
# --> a Numpy array of training data (and the labels)
# --> a tuple of validation data
# --> the # of epochs in which training occurs (in each epoch, we iterate over all training data)
# --> the batch size -- i.e., the number of samples tested before a gradient update
#       --> if the number if small, you do many weight updates during training, so your weights
#           may be better...
#       --> however, training run time also increases as you do more weight updates
#       --> a batch size of around 100 should be a happy medium
# --> verbose=1 shows more details about the progess of training; verbose=2 shows less detail
#       --> both versions show training (acc) and testing (val_acc) accuracy during training
###################################################################################################
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=100, verbose=2)

###################################################################################################
# Print out the final network accuracy with test data
###################################################################################################
scores = model.evaluate(X_test, y_test, verbose=1)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))


###################################################################################################
# To visualize the learned filters for MNIST, use the code below
# Note that the first line will get the weight values for the first (and only) layer of the network
#   --> first_layer_weigths is a 784, 10 tuple (i.e., 784 weights for 10 classes)
#   --> I simply get all of the weights for a given image class (i.e., test_number = 0) will
#       gather the weights for the 0 class and then illustrate them with a color map
###################################################################################################

first_layer_weights = model.layers[0].get_weights()[0]

test_number = 0

image_array = [0] * len(first_layer_weights)
for j in range(len(first_layer_weights)):
   image_array[j] = first_layer_weights[j][test_number]
   

plt.imshow(np.array(image_array).reshape(28,28), cmap='bwr')
plt.colorbar()

