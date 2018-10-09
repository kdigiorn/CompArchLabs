#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 10:22:49 2018

@author: mniemier
"""

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import plot_model
from keras import backend as K


###################################################################################################
# Define image size for network model -- all input images are scaled to this size.
###################################################################################################   
img_width, img_height = 150, 150

###################################################################################################
# Include pointers to training and validation data folders -- be sure to examine subfolder structure
# --> I also define the # of training samples, validation samples, epochs, and batch size here.
###################################################################################################   
train_data_dir = 'chi_st/train'
validation_data_dir = 'chi_st/validation'
nb_train_samples = 5000
nb_validation_samples = 1000
epochs = 5
batch_size = 100

###################################################################################################
# As before, this code simply organizes input data such that channels either come first or last
# depending on the backend used (TensorFlow or Theano)
###################################################################################################   
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

###################################################################################################
# Define our CNN model
###################################################################################################
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

### INSERT YOUR CODE HERE ###

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

###################################################################################################
# To process images in respective directories, we can use the ImageDataGenerator class
#	The code provided here normalizes image data, etc.
#	The parameters will not be discussed further here, 
#    but more information / options can be found at:  https://keras.io/preprocessing/image/ 
###################################################################################################   
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

###################################################################################################
# This is the augmentation configuration we will use for testing:  only rescaling
###################################################################################################
test_datagen = ImageDataGenerator(rescale=1. / 255)

###################################################################################################
# Subsequent invocations of flow_from_directory() will use the paths to the training and validation 
# data, and generate batches of data
# 
# Note that if you wanted to work with grayscale images (for example) you could simply change 
# color_mode to ‘gray_scale’ (and the number of color channels)
# 
# Again, you will not need to change any parameters here, but a more detailed description of this 
# class can also be found at the link above.
###################################################################################################
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

###################################################################################################
# Finally, the invocation of model.fit_generator will simply train the model on batches of data.  
# More information can be found at:  https://keras.io/models/sequential/
# --> However, this is just analogous to model.fit() discussed in other examples.
###################################################################################################
model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)

###################################################################################################
# Here, we save the model weights and generate an image of the network...
###################################################################################################
model.save('saved_model_long')

plot_model(model, to_file='/Users/mniemier/Desktop/model_long.png', show_shapes=True, show_layer_names=True)
