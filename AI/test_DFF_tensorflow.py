import numpy as np
import os
import tensorflow as tf

os.chdir("\\Users\sbayed\Desktop\sbayed_local\Machine Learning\AI\AI")
from DFF_tensorflow import *
from utilities import *

X_train_orig, Y_train, X_test_orig, Y_test, classes = load_data()
X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0], -1).T
X_test_flatten = X_test_orig.reshape(X_test_orig.shape[0], -1).T

X_train = X_train_flatten/255.
X_test = X_test_flatten/255.


tf.reset_default_graph()

# Inputs
layers_dims = [X_train.shape[0], 20, 7, 5, 1]
activations = ['none','relu','relu','relu','sigmoid']

# Parameters initialization
parameters = initialize_parameters(layers_dims, activations)