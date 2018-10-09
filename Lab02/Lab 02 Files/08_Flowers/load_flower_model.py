#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from keras.models import load_model
from keras.preprocessing import image
from termcolor import colored
import matplotlib.pyplot as plt
import numpy as np
import os

model=load_model('flower_model_cnn_50epochs.h5')  # path to model from current working directory
model.summary()

#for layer in model.layers:
#    weights = layer.get_weights() # list of numpy arrays
#    print(weights)

def test_image(filename):
    img = image.load_img(filename)
   
    plt.imshow(img)    # display image
    plt.show()

    img_tensor = image.img_to_array(img)   # creates tensor
    img_tensor = img_tensor.astype('float32') / 255.  # divides each element by 255 so it is between 0 and 1

    img_tensor = img_tensor.reshape(1,128,128,3) # size of input tensor

    y = model.predict(img_tensor)
    
    y1 = y[0]
    print('y', y)
    print('y1', int(y1[0]))
    
    # Dictionary of all 10 flower classes 
    flowers_dict = {0:'phlox', 1:'rose', 2:'calendula', 3:'iris', 4:'leucanthemum maximum',
                5:'bellflower', 6:'viola', 7:'rudbeckia', 8:'peony', 9:'aquilegia'}
    
    print('Prediction: {} '.format(flowers_dict[np.argmax(y)]))
    if filename.startswith('test_flowers/' + str(np.argmax(y))):
        print(colored('Correct', 'green'))
    else:
        print('Label:', flowers_dict[int(filename[13])])
        print(colored('Incorrect', 'red'))
    print('   Class probabilities > 1%')
    for i in range(10):
        percent = float(round(100* y[0][i], 2))
        if percent >= 1:
            print('{}: {}%'.format(flowers_dict[i] ,percent))   

for filename in os.listdir('test_flowers'):
    if filename.endswith(".jpg") or filename.endswith(".jpeg"): 
        test_image('test_flowers/' + filename)