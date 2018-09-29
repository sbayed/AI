import numpy as np
import os
import tensorflow as tf
import numpy as np
from DFF_tensorflow import *
from utilities import *

os.chdir("\\Users\sbayed\Desktop\sbayed_local\Machine Learning\AI\AI")

# Data processing
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
X_train = np.transpose(mnist.train.images[:40000,:])
Y_train = np.transpose(mnist.train.labels[:40000,:])
X_test = np.transpose(mnist.test.images[:10000,:])
Y_test = np.transpose(mnist.test.labels[:10000,:])



# Inputs
n_x = X_train.shape[0]
n_y = Y_train.shape[0]
m = Y_train.shape[1]
layers_dims = [n_x, n_y]
activations = ['none','softmax']
learning_rate = 0.1
num_epochs = 1
minibatch_size = m

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
    j = [0.03, 0.03, 0.01, 0.9, 0.01, 0.01, 0.0025,0.0025, 0.0025, 0.0025]
    k = [0,0,0,1,0,0,0,0,0,0]
    cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(AL),reduction_indices=[1]))
    cost = tf.losses.softmax_cross_entropy(Y,AL)



