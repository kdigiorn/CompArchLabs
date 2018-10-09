Flower_images is a folder with all of the data. 

All of the image files are in this folder along with a csv file 
that lists all of the file names and the labels for each one

flowers.py is the script that creates and saves the model. 

flower_model_cnn_50epochs.h5 is a trained model for the flowers 
database - trained with 50 epochs 

load_flower_model.py loads the saved model and tests the image files 
in test_flowers. (These were randomly downloaded online.) 

This mode is somewhat accurate. Seems to base observations on color 
(probably since there were so few training files). 

For visual clarity, only class percentages that are > 1 % so that the 
output isn't crowded with a bunch of small percentages.
