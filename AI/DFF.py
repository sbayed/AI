import numpy as np
import matplotlib.pyplot as plt
from utilities import *

def initialize_parameters(layers_dims, activations):
    """
    Arguments:
    layers_dims -- list containing the dimensions of each layer (includes layer 0)
    activations -- list containing the activation functions of each layer
    
    Returns:
    parameters -- dictionary containing parameters

    """
    
    np.random.seed(3)
    parameters = {}
    L = len(layers_dims)

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l-1]) / np.sqrt(layers_dims[l-1])
        parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))
        parameters['g' + str(l)] = activations[l]
        
        assert(parameters['W' + str(l)].shape == (layers_dims[l], layers_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layers_dims[l], 1))

        
    return parameters
def initialize_velocity(parameters):
    """
    Initializes the velocity as a python dictionary with:
                - keys: "dW1", "db1", ..., "dWL", "dbL" 
                - values: numpy arrays of zeros of the same shape as the corresponding gradients/parameters.
    Arguments:
    parameters -- python dictionary containing your parameters.
                    parameters['W' + str(l)] = Wl
                    parameters['b' + str(l)] = bl
    
    Returns:
    v -- python dictionary containing the current velocity.
                    v['dW' + str(l)] = velocity of dWl
                    v['db' + str(l)] = velocity of dbl
    """
    
    L = len(parameters) // 3 # number of layers in the neural networks
    v = {}
    
    # Initialize velocity
    for l in range(L):
        v["dW" + str(l+1)] = np.zeros(parameters["W" + str(l+1)].shape)
        v["db" + str(l+1)] = np.zeros(parameters["b" + str(l+1)].shape)
        
    return v
def initialize_adam(parameters) :
    """
    Initializes v and s as two python dictionaries with:
                - keys: "dW1", "db1", ..., "dWL", "dbL" 
                - values: numpy arrays of zeros of the same shape as the corresponding gradients/parameters.
    
    Arguments:
    parameters -- python dictionary containing your parameters.
                    parameters["W" + str(l)] = Wl
                    parameters["b" + str(l)] = bl
    
    Returns: 
    v -- python dictionary that will contain the exponentially weighted average of the gradient.
                    v["dW" + str(l)] = ...
                    v["db" + str(l)] = ...
    s -- python dictionary that will contain the exponentially weighted average of the squared gradient.
                    s["dW" + str(l)] = ...
                    s["db" + str(l)] = ...

    """
    
    L = len(parameters) // 3 # number of layers in the neural networks
    v = {}
    s = {}
    
    # Initialize v, s. Input: "parameters". Outputs: "v, s".
    for l in range(L):
        v["dW" + str(l+1)] = np.zeros(parameters["W" + str(l+1)].shape)
        v["db" + str(l+1)] = np.zeros(parameters["b" + str(l+1)].shape)
        s["dW" + str(l+1)] = np.zeros(parameters["W" + str(l+1)].shape)
        s["db" + str(l+1)] = np.zeros(parameters["b" + str(l+1)].shape)
    
    return v, s
def linear_forward(A_prev, W, b):
    """
    Implement the linear part of a layer's forward propagation.

    Arguments:
    A_prev -- activations from previous layer (or input data)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)

    Returns:
    Z -- pre-activation parameter
    cache -- dictionary containing "A_prev", "W" and "b" ; stored for computing the backward pass efficiently
    """
    
    Z = W.dot(A_prev) + b
    
    assert(Z.shape == (W.shape[0], A_prev.shape[1]))
    cache = (A_prev, W, b)
    
    return Z, cache
def sigmoid(x):
    """
    Implements the sigmoid function

    Arguments:
    x -- numpy array of any shape

    Returns:
    y -- output of sigmoid(x)

    """

    y = 1 / (1 + np.exp(-x))

    assert (y.shape == x.shape)

    return y
def relu(x):
    """
    Implement the RELU function.

    Arguments:
    x -- numpy array of any shape

    Returns:
    y -- output of ReLu(x)
    """

    y = np.maximum(0, x)

    assert (y.shape == x.shape)

    return y
def activation_forward(Z, activation):
    """
    Implement the activation part of a layer's forward propagation.

    Arguments:
    Z -- pre-activation from linear forward
    activation -- activation function

    Returns:
    A -- post-activation
    cache -- dictionary containing "Z" and "g" (activation function at current layer)
    """
    if activation == "sigmoid":
        A = sigmoid(Z)
    elif activation == "relu":
        A = relu(Z)

    cache = (Z, activation)
    return A, cache
def linear_activation_forward(A_prev, W, b, activation):
    """
    Implement the forward propagation for the LINEAR->ACTIVATION block

    Arguments:
    A_prev -- activations from previous layer (or input data)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)
    activation -- the activation to be used in this layer, stored as a text string

    Returns:
    A -- post-activation value
    cache -- dictionary containing "linear_cache" and "activation_cache";
             stored for computing the backward pass efficiently
    """

    Z, linear_cache = linear_forward(A_prev, W, b)
    A, activation_cache = activation_forward(Z, activation)

    
    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)

    return A, cache
def L_model_forward(X, parameters):
    """
    Implement forward propagation
    
    Arguments:
    X -- input matrix
    parameters -- dictionary containing parameters
    
    Returns:
    AL -- last post-activation value
    caches -- list of caches containing ever cache of linear activation forward
    """

    caches = []
    A = X
    L = len(parameters) // 3

    for l in range(1, L):
        A_prev = A 
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], parameters['g' + str(l)])
        caches.append(cache)

    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], parameters['g' + str(L)])
    caches.append(cache)
    
    assert(AL.shape == (1,X.shape[1]))
            
    return AL, caches
def L2_regularization_cost(parameters, lambd):
    """
    Computes L2 regularization cost

    Arguments:
    parameters -- dictionary containing parameters
    lambd -- regularization parameter


    Returns:
    J_L2 -- cost value
    """
    L = len(parameters) // 3  # number of layers in the neural network
    J_L2 = 0

    for l in range(L):
        J_L2 = J_L2 + np.sum(np.square(parameters["W" + str(l + 1)]))

    J_L2 = J_L2*lambd/2

    return J_L2
def compute_cost(AL, Y, cost, parameters, lambd):
    """
    Implement the cost function

    Arguments:
    AL -- probability vector (output of forward propagation)
    Y -- output vector
    cost -- cost function type


    Returns:
    J -- cost value
    """

    m = Y.shape[1]

    if cost == 'cross-entropy':
        J = (1./m) * (-np.dot(Y,np.log(AL).T) - np.dot(1-Y, np.log(1-AL).T)) + (1./m) * L2_regularization_cost(parameters, lambd)

    
    J = np.squeeze(J)
    assert(J.shape == ())
    
    return J
def initialize_backpropagation(AL, Y, cost):
    """
    Arguments:
    AL -- last post-activation value
    Y -- output vector
    cost -- cost function type

    Returns:
    dAL -- gradient of the cost with respect to the last post-activation for current layer l
    """
    Y = Y.reshape(AL.shape)

    if cost == 'cross-entropy':
        dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

    return dAL
def sigmoid_gradient(x):
    """
    Implement the sigmoid gradient function.

    Arguments:
    x -- np array of real values

    Returns:
    y -- sigmoid gradient transformation
    """
    y = sigmoid(x)*(1-sigmoid(x))
    return y
def relu_gradient(x):
    """
    Implement the relu gradient function.

    Arguments:
    x -- np array of real values

    Returns:
    y -- relu gradient transformation
    """

    y = np.zeros((x.shape))
    y[x > 0] = 1

    return y
def activation_backward(dA, Z, activation):
    """
    Implement the activation part of a layer's backward propagation.

    Arguments:
    dA -- gradient of the cost with respect to post-activation for current layer l
    Z -- pre-activation from linear forward
    activation -- activation function

    Returns:
    dZ -- gradient of the cost with respect to the linear forward for current layer l
    """

    if activation == "sigmoid":
        dZ = dA * sigmoid_gradient(Z)
    elif activation == "relu":
        dZ = dA * relu_gradient(Z)


    return dZ
def linear_backward(dZ, A_prev, W, b, lambd):
    """
    Implement the linear part of a layer's forward propagation.

    Arguments:
    dZ -- gradient of the cost with respect to the linear forward for current layer l
    A_prev -- activations from previous layer
    W -- weights matrix: numpy array of shape
    b -- bias vector, numpy array of shape

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    m = A_prev.shape[1]

    dA_prev = np.dot(W.T,dZ)
    dW = 1/m * np.dot(dZ,A_prev.T) + lambd/m * W
    db = 1/m * np.sum(dZ, axis = 1, keepdims = True)

    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)

    return dA_prev, dW, db
def linear_activation_backward(dA, Z, activation, A_prev, W, b, lambd):
    """
    Implement the backward propagation for the LINEAR->ACTIVATION layer.

    Arguments:
    dA -- gradient of the cost with respect to post-activation gradient for current layer l
    Z -- pre-activation from linear forward
    activation -- activation function
    A_prev -- activations from previous layer
    W -- weights matrix
    b -- bias vector

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    dZ = activation_backward(dA, Z, activation)
    dA_prev, dW, db = linear_backward(dZ, A_prev, W, b, lambd)

    return dA_prev, dW, db
def L_model_backward(dAL, caches, lambd):
    """
    Implement the backward propagation for the [LINEAR->ACTIVATION] * (L) group

    Arguments:
    dAL -- gradient of the cost with respect to the last post-activation
    caches -- list of caches containing every cache of linear_activation_forward()

    Returns:
    grads -- A dictionary with the gradients
             grads["dA" + str(l)] = ...
             grads["dW" + str(l)] = ...
             grads["db" + str(l)] = ...
    """
    grads = {}
    L = len(caches)
    dA_prev = dAL

    for l in reversed(range(L)):
        current_cache = caches[l]
        current_linear_cache, current_activation_cache = current_cache
        A_prev, W, b = current_linear_cache
        Z, activation = current_activation_cache
        dA_prev, dW, db = linear_activation_backward(dA_prev, Z, activation, A_prev, W, b, lambd)
        grads["dA" + str(l)] = dA_prev
        grads["dW" + str(l + 1)] = dW
        grads["db" + str(l + 1)] = db

    return grads
def update_parameters(parameters, grads, learning_rate):
    """
    Update parameters using gradient descent
    
    Arguments:
    parameters -- python dictionary containing your parameters 
    grads -- python dictionary containing your gradients, output of L_model_backward
    
    Returns:
    parameters -- python dictionary containing your updated parameters 
                  parameters["W" + str(l)] = ... 
                  parameters["b" + str(l)] = ...
    """
    
    L = len(parameters) // 3 # number of layers in the neural network

    # Update rule for each parameter
    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]
        parameters['g' + str(l+1)] = parameters['g' + str(l+1)] # keeps activations in memory
        
    return parameters
def update_parameters_with_momentum(parameters, grads, v, beta, learning_rate):
    """
    Update parameters using Momentum
    
    Arguments:
    parameters -- python dictionary containing your parameters:
                    parameters['W' + str(l)] = Wl
                    parameters['b' + str(l)] = bl
    grads -- python dictionary containing your gradients for each parameters:
                    grads['dW' + str(l)] = dWl
                    grads['db' + str(l)] = dbl
    v -- python dictionary containing the current velocity:
                    v['dW' + str(l)] = ...
                    v['db' + str(l)] = ...
    beta -- the momentum hyperparameter, scalar
    learning_rate -- the learning rate, scalar
    
    Returns:
    parameters -- python dictionary containing your updated parameters 
    v -- python dictionary containing your updated velocities
    """

    L = len(parameters) // 3 # number of layers in the neural networks
    
    # Momentum update for each parameter
    for l in range(L):
        # compute velocities
        v["dW" + str(l+1)] = beta*v["dW" + str(l+1)] + (1-beta)*grads["dW" + str(l+1)]
        v["db" + str(l+1)] = beta*v["db" + str(l+1)] + (1-beta)*grads["db" + str(l+1)]
        # update parameters
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate*v["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate*v["db" + str(l+1)]
        parameters['g' + str(l+1)] = parameters['g' + str(l+1)] # keeps activations in memory
        
    return parameters, v
def update_parameters_with_adam(parameters, grads, v, s, t, learning_rate = 0.01, beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8):
    """
    Update parameters using Adam
    
    Arguments:
    parameters -- python dictionary containing your parameters:
                    parameters['W' + str(l)] = Wl
                    parameters['b' + str(l)] = bl
    grads -- python dictionary containing your gradients for each parameters:
                    grads['dW' + str(l)] = dWl
                    grads['db' + str(l)] = dbl
    v -- Adam variable, moving average of the first gradient, python dictionary
    s -- Adam variable, moving average of the squared gradient, python dictionary
    learning_rate -- the learning rate, scalar.
    beta1 -- Exponential decay hyperparameter for the first moment estimates 
    beta2 -- Exponential decay hyperparameter for the second moment estimates 
    epsilon -- hyperparameter preventing division by zero in Adam updates

    Returns:
    parameters -- python dictionary containing your updated parameters 
    v -- Adam variable, moving average of the first gradient, python dictionary
    s -- Adam variable, moving average of the squared gradient, python dictionary
    """
    
    L = len(parameters) // 3                 # number of layers in the neural networks
    v_corrected = {}                         # Initializing first moment estimate, python dictionary
    s_corrected = {}                         # Initializing second moment estimate, python dictionary
    
    # Perform Adam update on all parameters
    for l in range(L):
        # Moving average of the gradients. Inputs: "v, grads, beta1". Output: "v".
        v["dW" + str(l+1)] = beta1*v["dW" + str(l+1)] + (1-beta1)*grads["dW" + str(l+1)]
        v["db" + str(l+1)] = beta1*v["db" + str(l+1)] + (1-beta1)*grads["db" + str(l+1)]

        # Compute bias-corrected first moment estimate. Inputs: "v, beta1, t". Output: "v_corrected".
        v_corrected["dW" + str(l+1)] = np.divide(v["dW" + str(l+1)], 1 - np.power(beta1,t))
        v_corrected["db" + str(l+1)] = np.divide(v["db" + str(l+1)], 1 - np.power(beta1,t))

        # Moving average of the squared gradients. Inputs: "s, grads, beta2". Output: "s".
        s["dW" + str(l+1)] = beta2*s["dW" + str(l+1)] + (1-beta2)*np.square(grads["dW" + str(l+1)])
        s["db" + str(l+1)] = beta2*s["db" + str(l+1)] + (1-beta2)*np.square(grads["db" + str(l+1)])

        # Compute bias-corrected second raw moment estimate. Inputs: "s, beta2, t". Output: "s_corrected".
        s_corrected["dW" + str(l+1)] = np.divide(s["dW" + str(l+1)],1 - np.power(beta2,t))
        s_corrected["db" + str(l+1)] = np.divide(s["db" + str(l+1)],1 - np.power(beta2,t))

        # Update parameters. Inputs: "parameters, learning_rate, v_corrected, s_corrected, epsilon". Output: "parameters".
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate*np.divide(v_corrected["dW" + str(l+1)],np.sqrt(s_corrected["dW" + str(l+1)]) + epsilon)
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate*np.divide(v_corrected["db" + str(l+1)],np.sqrt(s_corrected["db" + str(l+1)]) + epsilon)

    return parameters, v, s
def L_layer_model(X, Y, layers_dims, activations, cost, lambd, learning_rate = 0.0075, num_iterations = 3000, print_cost = False):
    """
    Implements a L-layer neural network

    Arguments:
    X -- input matrix
    Y -- output vector
    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    print_cost -- if True, it prints the cost every 100 steps

    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    np.random.seed(1)
    Js = []

    # Parameters initialization
    parameters = initialize_parameters(layers_dims, activations)

    # Loop (gradient descent)
    for i in range(0, num_iterations):

        # Forward propagation
        AL, caches = L_model_forward(X, parameters)

        # Compute cost
        J = compute_cost(AL, Y, cost, parameters, lambd)

        # Backward propagation initialization
        dAL = initialize_backpropagation(AL, Y, cost)

        # Backward propagation
        grads = L_model_backward(dAL, caches, lambd)

        # Update parameters.
        parameters = update_parameters(parameters, grads, learning_rate)

        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, J))
        if print_cost and i % 100 == 0:
            Js.append(J)

    # plot the cost
    plt.plot(np.squeeze(Js))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()



    return parameters
def model(X, Y, layers_dims, activations, optimizer, cost, lambd, beta = 0.9, beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8, learning_rate = 0.0075, num_epochs = 10000, mini_batch_size = 64, print_cost = False):
    """
    Implements a L-layer neural network with mini-batch gradient descent and different optimization modes

    Arguments:
    X -- input matrix
    Y -- output vector
    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
    activations -- activation functions 
    optimizer -- optimizer type
    learning_rate -- learning rate of the gradient descent update rule
    beta -- Momentum hyperparameter
    beta1 -- Exponential decay hyperparameter for the past gradients estimates 
    beta2 -- Exponential decay hyperparameter for the past squared gradients estimates
    epsilon -- hyperparameter preventing division by zero in Adam updates
    num_epochs -- number of epochs
    mini_batch_size -- the size of a mini batch
    print_cost -- if True, it prints the cost every 100 steps

    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
 
    Js = []
    t = 0

    # Parameters initialization
    parameters = initialize_parameters(layers_dims, activations)

    # Optimizer initization
    if optimizer == "gd":
        pass # no initialization required for gradient descent
    elif optimizer == "momentum":
        v = initialize_velocity(parameters)
    elif optimizer == "adam":
        v, s = initialize_adam(parameters)

    # Loop (gradient descent)
    for i in range(num_epochs):

        # Define the random minibatches
        minibatches = random_mini_batches(X, Y, mini_batch_size)

        for minibatch in minibatches:

            # Select a minibatch
            (minibatch_X, minibatch_Y) = minibatch

            # Forward propagation
            AL, caches = L_model_forward(minibatch_X, parameters)

            # Compute cost
            J = compute_cost(AL, minibatch_Y, cost, parameters, lambd)

            # Backward propagation initialization
            dAL = initialize_backpropagation(AL, minibatch_Y, cost)

            # Backward propagation
            grads = L_model_backward(dAL, caches, lambd)

            # Update parameters
            if optimizer == "gd":
                parameters = update_parameters(parameters, grads, learning_rate)
            elif optimizer == "momentum":
                parameters, v = update_parameters_with_momentum(parameters, grads, v, beta, learning_rate)
            elif optimizer == "adam":
                t = t + 1 # Adam counter
                parameters, v, s = update_parameters_with_adam(parameters, grads, v, s, t, learning_rate, beta1, beta2,  epsilon)

        # Print the cost every 1000 epochs
        if print_cost and i % 1000 == 0:
            print("Cost after epoch %i: %f" % (i, J))
        if print_cost and i % 100 == 0:
            Js.append(J)

    # plot the cost
    plt.plot(Js)
    plt.ylabel('cost')
    plt.xlabel('epochs (per 100)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()



    return parameters
def predict(X, y, parameters):
    """
    This function is used to predict the results of a  L-layer neural network.
    
    Arguments:
    X -- data set of examples you would like to label
    parameters -- parameters of the trained model
    
    Returns:
    p -- predictions for the given dataset X
    """
    
    m = X.shape[1]
    n = len(parameters) // 4 # number of layers in the neural network
    p = np.zeros((1,m))
    
    # Forward propagation
    probas, caches = L_model_forward(X, parameters)

    
    # convert probas to 0/1 predictions
    for i in range(0, probas.shape[1]):
        if probas[0,i] > 0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0

    print("Accuracy: "  + str(np.sum((p == y)/m)))
        
    return p


