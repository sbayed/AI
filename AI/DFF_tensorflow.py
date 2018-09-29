import tensorflow as tf
from tensorflow.python.framework import ops
from utilities import *

def initialize_parameters(layers_dims):
    """
    Initializes parameters to build a DFF neural network with tensorflow. 

    Arguments:
    layers_dims -- list containing the dimensions of each layer (includes layer 0)
    
    Returns:
    parameters -- a dictionary of tensors containing the parameters
    """
    
    parameters = {}
    L = len(layers_dims)

    for l in range(1, L):
        parameters['W' + str(l)] = tf.get_variable(name = 'W' + str(l), shape = (layers_dims[l],layers_dims[l-1]), initializer = tf.contrib.layers.xavier_initializer())
        parameters['b' + str(l)] = tf.get_variable(name = 'b' + str(l), shape = (layers_dims[l],1), initializer = tf.zeros_initializer())

    return parameters
def forward_propagation(X, parameters, activations):
    """
    Implements the forward propagation for the DFF NN model specified by the user
    
    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing the parameters
    activations -- list containing the activation functions of each layer

    Returns:
    ZL-- the output of the last linear unit
    """
    A_prev = X
    L = len(parameters) // 2
    for l in range(1, L+1):
        W = parameters['W' + str(l)]
        b = parameters['b' + str(l)]
        g = activations[l]
        if l < L:
            with tf.name_scope('linear_activation' + str(l)):
                with tf.name_scope('linear' + str(l)):
                    Z = tf.add(tf.matmul(W,A_prev),b, name = 'add' + str(l))
                with tf.name_scope('activation' + str(l)):
                    if g == 'relu':
                        A = tf.nn.relu(Z, name = 'relu' + str(l))
                    elif g == 'sigmoid':
                        A = tf.nn.sigmoid(Z, name = 'sigmoid' + str(l))
            A_prev = A
        elif l == L:
            with tf.name_scope('linear' + str(l)):
                Z = tf.add(tf.matmul(W,A_prev),b, name = 'add' + str(l))
    ZL = Z
    return ZL
def dff_nn_classification(X_train, Y_train, X_test, Y_test, layers_dims, activations, learning_rate, num_epochs, minibatch_size):
    """
    Implements a DFF tensorflow neural network for classification
    
    Arguments:
    X_train -- training set
    Y_train -- test set
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    
    Returns:
    parameters -- parameters learnt by the model
    """

    ops.reset_default_graph() 
    (n_x, m) = X_train.shape
    n_y = Y_train.shape[0]
    
    # Create placeholders
    X = tf.placeholder(dtype = tf.float32, shape = (n_x, None), name = 'X')
    Y = tf.placeholder(dtype = tf.float32, shape = (n_y, None), name = 'Y')

    # Initialize parameters
    parameters = initialize_parameters(layers_dims)

    # Forward propagation: Build the forward propagation in the tensorflow graph
    ZL = forward_propagation(X, parameters, activations)

    # Cost function: Add cost function to tensorflow graph
    with tf.name_scope('cost'):
        logits = tf.transpose(ZL)
        labels = tf.transpose(Y)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = labels))

    # Optimizer: Define the tensorflow optimizer
    optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(cost)

    # Initialize all the variables
    init = tf.global_variables_initializer()

    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:

        # Run the initialization
        sess.run(init)

        # Do the training loop
        for epoch in range(num_epochs):
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size)
            for minibatch in minibatches:
                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch
                _ , minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
            if epoch%100 == 0:
                print('Step:' + str(epoch) + '  Loss = ' + str(minibatch_cost))

        # Save the parameters                
        parameters = sess.run(parameters)

        with tf.name_scope('validation'):
            # Calculate the correct predictions
            correct_prediction = tf.equal(tf.argmax(ZL), tf.argmax(Y))

            # Calculate and print accuracy on the test set
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            print ("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
            print ("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))

        # Save tensorgraph
        writer = tf.summary.FileWriter('./graphs', sess.graph)

    return parameters