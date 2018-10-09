#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from keras.models import load_model
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np

###################################################################################################
# 1.  Load the learned/trained model
###################################################################################################   
model=load_model('cifar_3_classes_model')  # path to model from current working directory
model.summary()


###################################################################################################
# 3.  Test image function
###################################################################################################   
def test_image(filename):
    
    #-----------------------------------------------------------
    # Load the speicified image, display the specified image
    #-----------------------------------------------------------
    img = image.load_img(filename, target_size=(32,32))
   
    plt.imshow(img)    # display image
    plt.show()

    #-----------------------------------------------------------
    # Convert the image to a tensor, noramilze
    #-----------------------------------------------------------
    img_tensor = image.img_to_array(img)                # creates tensor
    img_tensor = img_tensor.astype('float32') / 255.    # divides each element by 255 so it is between 0 and 1

    img_tensor = img_tensor.reshape(1,32,32,3)          # size of input tensor

    #-----------------------------------------------------------
    # Make the prediction, return probability distribution for classes
    #-----------------------------------------------------------
    y = model.predict(img_tensor)
    print('Model returns probability distribution:', y)


    #-----------------------------------------------------------
    # Create a dictionary of CIFAR classes that were trained to display labels
    #-----------------------------------------------------------
    class_dictionary = {0:'Airplane', 1:'Frog', 2:'Truck'}
    
    print('Prediction: {} '.format(class_dictionary[np.argmax(y)]))
    
    #-----------------------------------------------------------
    # Print probability distribution
    #-----------------------------------------------------------
    for i in range(3):
        percent = float(round(100* y[0][i], 2))
        print('{}: {}%'.format(class_dictionary[i], percent))   
       

###################################################################################################
# 2.  Load a test image (from the local directory)
###################################################################################################   
test_image('image_truck.jpg')
        
