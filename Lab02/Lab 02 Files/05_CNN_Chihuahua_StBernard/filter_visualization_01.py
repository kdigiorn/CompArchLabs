
from __future__ import print_function


# Print a summary of the model
from keras.models import load_model
model=load_model('saved_model_long')

model.summary()

