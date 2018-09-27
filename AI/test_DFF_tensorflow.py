import numpy as np
import os
import tensorflow as tf
import numpy as np
from DFF_tensorflow import *
from utilities import *

os.chdir("\\Users\sbayed\Desktop\sbayed_local\Machine Learning\AI\AI")

# Data processing
X_train_orig, Y_train, X_test_orig, Y_test, classes = load_data()
X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0], -1).T
X_test_flatten = X_test_orig.reshape(X_test_orig.shape[0], -1).T
C = len(np.unique(Y_train))

X_train = X_train_flatten/255.
X_test = X_test_flatten/255.

Y_train = convert_to_one_hot(Y_train, C)
Y_test = convert_to_one_hot(Y_test, C)

# Inputs
n_x = X_train.shape[0]
n_y = Y_train.shape[0]
layers_dims = [n_x, 20, 7, 5, n_y]
activations = ['none','relu','relu','relu','softmax']
learning_rate = 0.00001
num_epochs = 1
minibatch_size = 64

# Test (Model level)
parameters = dff_nn_classification(X_train, Y_train, X_test, Y_test, layers_dims, activations, learning_rate, num_epochs, minibatch_size)

# Tests
tf.reset_default_graph()
with tf.Session() as sess:
    # Parameters initialization
    parameters = initialize_parameters(layers_dims)
    # Forward propagation
    X = tf.placeholder(dtype = tf.float32, shape = (n_x, None), name = 'X')
    Y = tf.placeholder(dtype = tf.float32, shape = (n_y, None), name = 'Y')
    AL = forward_propagation(X, parameters, activations)
    # Cost function
    cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(AL)))
    cost = tf.losses.softmax_cross_entropy(Y,AL)



