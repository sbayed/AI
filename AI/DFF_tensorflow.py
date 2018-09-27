import tensorflow as tf

def initialize_parameters(layers_dims, activations):

    """
    Initializes parameters to build a DFF neural network with tensorflow. 

    Arguments:
    layers_dims -- list containing the dimensions of each layer (includes layer 0)
    activations -- list containing the activation functions of each layer
    
    Returns:
    parameters -- a dictionary of tensors containing the parameters
    """
    
    parameters = {}
    L = len(layers_dims)

    for l in range(1, L):
        parameters['W' + str(l)] = tf.get_variable(name = 'W' + str(l), shape = (layers_dims[l], layers_dims[l-1]), initializer = tf.contrib.layers.xavier_initializer())
        parameters['b' + str(l)] = tf.get_variable(name = 'b' + str(l), shape = (layers_dims[l], 1), initializer = tf.zeros_initializer())
        parameters['g' + str(l)] = activations[l]

    return parameters