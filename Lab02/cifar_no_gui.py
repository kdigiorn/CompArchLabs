#!/usr/bin/env python3

'''
Code based in part on:
1.) http://www.cs.nthu.edu.tw/~shwu/courses/ml/labs/11_NN_Regularization/11_NN_Regularization.html
'''


from keras.datasets import cifar10

from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras import backend as K

from keras.utils import np_utils

#import matplotlib.pyplot as plt
import numpy as np

###################################################################################################
# Build model function -- takes image height, width, and class number as arguments
# --> CIFAR 10 has 10 classes and images are 32x32
###################################################################################################    
def build_model(img_width, img_height, num_classes=10):
    
    #---------------------------------------------------------------------
    # Organize the data -- channels first or last depending on TF/Theano backend
    #---------------------------------------------------------------------
	if K.image_data_format() == 'channels_first':
	   input_shape = (1, img_width, img_height)
	else:
	   input_shape = (img_width, img_height, 1)

    #---------------------------------------------------------------------
    # Build the softmax classifier model -- note how the input is still 
    # flattened as the first layer, but it is specified differnetly than in 
    # MNIST
    #---------------------------------------------------------------------
	model = Sequential()
	model.add(Flatten(input_shape=input_shape))
	model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))
	model.summary()

	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

	return model

###################################################################################################
# Set up the data for training, testing
###################################################################################################    
def setup_data(classes=None, dataset='cifar10', color='grayscale'):
    
    #---------------------------------------------------------------------
    # Load CIFAR 10 data -- similar to MNIST
    #---------------------------------------------------------------------
	if dataset == 'cifar10':
		(x_train, y_train), (x_test, y_test) = cifar10.load_data()
	else:
		exit(1)

    #---------------------------------------------------------------------
    # Normalize pixel data
    #---------------------------------------------------------------------
	X_train = x_train / 255.
	X_test = x_test / 255.

    #---------------------------------------------------------------------
    # Optionally convert image data to grayscale...
    #---------------------------------------------------------------------
	if color == 'grayscale':
		X_train = grayscale(X_train) 
		X_test = grayscale(X_test)

    #---------------------------------------------------------------------
    # Create 1-hot vectors to specify class
    #---------------------------------------------------------------------
	Y_train = np_utils.to_categorical(y_train)
	Y_test = np_utils.to_categorical(y_test)

    #---------------------------------------------------------------------
    # Create 1-hot vectors to specify class
    #---------------------------------------------------------------------
	if classes:
		X_train, Y_train, X_test, Y_test = filter_examples(classes, X_train, Y_train, X_test, Y_test)

	return X_train, Y_train, X_test, Y_test

###################################################################################################
# Use model.fit() to train the mode; note that by assigning the result to history, we get the 
# accuracy and loss for each step of the process
###################################################################################################    
def train_model(model, X_train, Y_train, X_test, Y_test, epochs, batch_size):
    # training
	history = model.fit(X_train, Y_train,
					batch_size=batch_size,
					epochs=epochs,
					verbose=1,
					validation_data=(X_test, Y_test))

    #---------------------------------------------------------------------
    # Save the history for each epoch to a test file
    #---------------------------------------------------------------------
	save_history(history, 'history.txt')

	loss, acc = model.evaluate(X_test, Y_test, verbose=0)
	print('Test loss:', loss)
	print('Test acc:', acc)

	return model

###################################################################################################
# Call this function if you wish to visualize the learned filters...
###################################################################################################    
def print_filters(model, classes):
	first_layer_weights = model.layers[1].get_weights()[0]
	image_array = list()

	for i in range(len(classes)):
		image_array.append(list())
		for j in range(1024):
			image_array[i].append(first_layer_weights[j][i])

		#plt.imshow(np.array(image_array[i]).reshape(32,32), cmap='bwr')
		#plt.colorbar()
		#plt.show()

###################################################################################################
# This function converts color images to grayscale...
###################################################################################################    
def grayscale(data, dtype='float32'):
    # luma coding weighted average in video systems
    r, g, b = np.asarray(.3, dtype=dtype), np.asarray(.59, dtype=dtype), np.asarray(.11, dtype=dtype)
    rst = r * data[:, :, :, 0] + g * data[:, :, :, 1] + b * data[:, :, :, 2]
    # add channel dimension
    rst = np.expand_dims(rst, axis=3)
    return rst

###################################################################################################
#  Save information returned from model.fit to text file  (By in large, this is just standard Python)
###################################################################################################   
def save_history(history, result_file):
	loss = history.history['loss']
	acc = history.history['acc']
	val_loss = history.history['val_loss']
	val_acc = history.history['val_acc']
	nb_epoch = len(acc)

	with open(result_file, "w") as fp:
		fp.write("epoch\tloss\tacc\tval_loss\tval_acc\n")
		for i in range(nb_epoch):
			fp.write("%d\t%f\t%f\t%f\t%f\n" % (i, loss[i], acc[i], val_loss[i], val_acc[i]))


###################################################################################################
# This function is used to help extract images associated with certain classes for use in training
# and testing
###################################################################################################   
def filter_examples(classes, X_train, Y_train, X_test, Y_test):
	X_train_filtered = list()
	Y_train_filtered = list()

	num_classes = len(classes)

	for i, y in enumerate(Y_train):
		for j, index in enumerate(classes):
			if y[index] == 1.:
				y_tmp = np.zeros(num_classes)
				y_tmp[j] = 1.
				X_train_filtered.append(X_train[i])
				Y_train_filtered.append(y_tmp)
				break


	X_test_filtered = list()
	Y_test_filtered = list()

	for i, y in enumerate(Y_test):
		for j, index in enumerate(classes):
			if y[index] == 1.:
				y_tmp = np.zeros(num_classes)
				y_tmp[j] = 1.
				X_test_filtered.append(X_test[i])
				Y_test_filtered.append(y_tmp)
				break

	return np.array(X_train_filtered), np.array(Y_train_filtered), np.array(X_test_filtered), np.array(Y_test_filtered)		


def main():
    
    ################################################################################################### 
    # STEP 1:   
    # Build the model -- note that CIFAR 10 images are 32 x 32 pixels
    # You can choose to use as many classes as you would like to ... 
    # --> 0: Airplane
    # --> 1: Car
    # --> 2: Bird
    # --> 3: Cat
    # --> 4: Deer
    # --> 5: Dog
    # --> 6: Frog
    # --> 7: Horse
    # --> 8: Ship
    # --> 9: Truck
    ###################################################################################################
    img_width, img_height = 32, 32
    classes = [0, 6, 9]
    model = build_model(img_width, img_height, len(classes))

    ###################################################################################################
    # STEP 2: 
    # Setup the data for the model
    ###################################################################################################
    X_train, Y_train, X_test, Y_test = setup_data(classes)
    
    print("X_train shape: {}".format(X_train.shape))
    print("Y_train shape: {}".format(Y_train.shape))
    print("X_test shape:  {}".format(X_test.shape))
    print("Y_test shape:  {}".format(Y_test.shape))
    
    #---------------------------------------------------------------------
    # Plot a randomly chosen image
    #---------------------------------------------------------------------
    # for i in range(75):
    #   img = i
    #   plt.imshow(X_train[img, :, :, 0], cmap=plt.get_cmap('gray'), interpolation='none')
    #   #plt.imshow(X_train[img], interpolation='none')
    #   plt.show()
    #   print("Label:", Y_train[img])

    ###################################################################################################
    # STEP 3: 
    # Train the model
    ###################################################################################################    
    epochs = 50
    batch_size = 128
    model = train_model(model, X_train, Y_train, X_test, Y_test, epochs, batch_size)

    ###################################################################################################
    # STEP 4: 
    # Display learned features -- note this information may NOT be that intuitive for this problem
    ###################################################################################################    
    # print_filters(model, classes)
    
    #---------------------------------------------------------------------
    # Save the model
    #---------------------------------------------------------------------
    model.save('cifar_3_classes_model')


if __name__ == '__main__':
	main()
