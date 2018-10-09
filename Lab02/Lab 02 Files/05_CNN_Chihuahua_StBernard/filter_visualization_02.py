'''
Visualization of the filters of a CNN, via gradient ascent in input space.
This script can run on CPU in a few minutes.
This script is meant for use only and is not discussed in great details as it is beyond the scope of the class.
'''
from __future__ import print_function

import numpy as np
import time
from keras.preprocessing.image import save_img
from keras import backend as K

###################################################################################################
# dimensions of the generated pictures for each filter.
###################################################################################################   
img_width = 150
img_height = 150

###################################################################################################
# the name of the layer we want to visualize 
###################################################################################################   
layer_name='conv2d_136'

###################################################################################################
# util function to convert a tensor into a valid image
###################################################################################################   
def deprocess_image(x):
    
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + K.epsilon())
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_data_format() == 'channels_first':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

###################################################################################################
# Load the model
###################################################################################################   
from keras.models import load_model
model=load_model('saved_model_long')

###################################################################################################
# This is the placeholder for the input images
###################################################################################################   
input_img = model.input

###################################################################################################
# Get the symbolic outputs of each "key" layer (we gave them unique names).
###################################################################################################   
layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])

###################################################################################################
# Utility function to normalize a tensor by its L2 norm
###################################################################################################   
def normalize(x): 
    return x / (K.sqrt(K.mean(K.square(x))) + K.epsilon())

kept_filters = []

###################################################################################################
# Scan through some nuber of filters...
###################################################################################################   
for filter_index in range(32):

    print('Processing filter %d' % filter_index)
    start_time = time.time()

    # we build a loss function that maximizes the activation
    # of the nth filter of the layer considered
    layer_output = layer_dict[layer_name].output
    if K.image_data_format() == 'channels_first':
        loss = K.mean(layer_output[:, filter_index, :, :])
    else:
        loss = K.mean(layer_output[:, :, :, filter_index])

    # we compute the gradient of the input picture wrt this loss
    grads = K.gradients(loss, input_img)[0]

    # normalization trick: we normalize the gradient
    grads = normalize(grads)

    # this function returns the loss and grads given the input picture
    iterate = K.function([input_img], [loss, grads])

    # step size for gradient ascent
    step = 1.

    # we start from a gray image with some random noise
    if K.image_data_format() == 'channels_first':
        input_img_data = np.random.random((1, 3, img_width, img_height))
    else:
        input_img_data = np.random.random((1, img_width, img_height, 3))
    input_img_data = (input_img_data - 0.5) * 20 + 128

    # we run gradient ascent for 20 steps
    for i in range(100):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step

        print('Current loss value:', loss_value)

    # decode the resulting input image
    if True:
        
    #if loss_value > 0
        img = deprocess_image(input_img_data[0])
        kept_filters.append((img, loss_value))
    end_time = time.time()
    print('Filter %d processed in %ds' % (filter_index, end_time - start_time))

###################################################################################################
# we will stich the best n^2 filters on a n x n grid.
###################################################################################################
n = 5

###################################################################################################
# the filters that have the highest loss are assumed to be more intuitive
# we will only keep the top n filters.
###################################################################################################   
kept_filters.sort(key=lambda x: x[1], reverse=True)
kept_filters = kept_filters[:n * n]

###################################################################################################
# build a black picture with enough space for
# our n x n filters of size 128 x 128, with a 5px margin in between
###################################################################################################   
margin = 5
width = n * img_width + (n - 1) * margin
height = n * img_height + (n - 1) * margin
stitched_filters = np.zeros((width, height, 3))

###################################################################################################
# fill the picture with our saved filters 
###################################################################################################   
for i in range(n):
    for j in range(n):
        img, loss = kept_filters[i * n + j]
        stitched_filters[(img_width + margin) * i: (img_width + margin) * i + img_width,
                         (img_height + margin) * j: (img_height + margin) * j + img_height, :] = img

###################################################################################################
# save the result to disk
###################################################################################################   
save_img('Chi_St_Last_5_Layer_%dx%d.png' % (n, n), stitched_filters)
