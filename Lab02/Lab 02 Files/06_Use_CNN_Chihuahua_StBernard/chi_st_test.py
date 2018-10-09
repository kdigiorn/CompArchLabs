#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 15:05:41 2018

@author: Morgan
"""

from keras.models import load_model
from keras.preprocessing import image
import matplotlib.pyplot as plt

###################################################################################################
# 1.  Load the learned/trained model
###################################################################################################   
model=load_model('saved_model')  # path to model from current working directory
model.summary()


###################################################################################################
# 3.  Test image function
###################################################################################################   
def test_image(filename):

    #-----------------------------------------------------------
    # Load the speicified image, display the specified image
    #-----------------------------------------------------------
    img = image.load_img(filename, target_size=(150,150))
   
    plt.imshow(img)    # display image
    plt.show()

    #-----------------------------------------------------------
    # Convert the image to a tensor, noramilze
    #-----------------------------------------------------------
    img_tensor = image.img_to_array(img)                # creates tensor
    img_tensor = img_tensor.astype('float32') / 255.    # divides each element by 255 so it is between 0 and 1

    img_tensor = img_tensor.reshape(1,150,150,3)        # size of input tensor

    #-----------------------------------------------------------
    # Make the prediction, return probability distribution for classes
    #-----------------------------------------------------------
    y = model.predict(img_tensor)
    y1 = y[0]

    #-----------------------------------------------------------
    # Create a dictionary of dog classes that were trained to display labels
    #-----------------------------------------------------------
    dogs_dict = {0:'Chihuahua', 1:'St. Bernard'}
    
    print('Prediction: {} '.format(dogs_dict[int(y1[0])]))
    
###################################################################################################
# 2.  Load a test image (from the local directory)
###################################################################################################   
test_image('image_stbernard01.jpg')
        
