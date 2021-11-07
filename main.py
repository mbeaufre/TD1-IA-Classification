import numpy as np
from kppv import *
from functions import *

def initialize_parameters_deep(layer_dims):
    np.random.seed(1)
    parameters = {}
    L = len(layer_dims)  # number of layers in the network
    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * 0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

        assert (parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l - 1]))
        assert (parameters['b' + str(l)].shape == (layer_dims[l], 1))
    return parameters


def linear_forward(A, W, b):
    Z = np.dot(W,A)+b
    cache = (A, W, b)
    return Z, cache

def linear_activation_forward(A_prev, W, b, activation):
    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    elif activation == "softmax":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = softmax(Z)
    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
    cache = (linear_cache, activation_cache)
    return A, cache

def L_model_forward(X, parameters):
    caches = []
    A = X
    L = len(parameters) // 2  # number of layers in the neural network
    for l in range(1, L + 1): # L+1 is pretty important ATTENTION
        A_prev = A
        Wl = parameters['W' + str(l)]
        bl = parameters['b' + str(l)]
        if l < L:
            A, cache = linear_activation_forward(A_prev, Wl, bl, activation="relu")
        else:
            # AL, cache = linear_activation_forward(A_prev, Wl, bl, activation="sigmoid")
            AL, cache = linear_activation_forward(A_prev, Wl, bl, activation="softmax")
        caches.append(cache)
    return AL, caches

def compute_cost(AL, Y, activation_out):
    m = Y.shape[1]
    if activation_out == "sigmoid":
        cost = (-1/m)*np.sum(Y*np.log(AL)+(1-Y)*np.log(1-AL))
    elif activation_out == "softmax":
        cost = (-1 / m) * np.sum(Y * np.log(AL))
    return cost

def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]
    dW = (1/m)*np.dot(dZ,A_prev.T)
    db = (1/m)*np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)
    return dA_prev, dW, db

def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache

    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
    elif activation == "softmax":
        dZ = softmax_backward(dA, activation_cache)
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)

    dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db

def L_model_backward(AL, Y, caches):
    grads = {}
    L = len(caches) # the number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL)) # derivative of cost with respect to AL
    grads['dA' + str(L-1)], grads['dW' + str(L)], grads['db' + str(L)] = linear_activation_backward(dAL, caches[L-1], activation="softmax")  # Softmax or sigmoid, in function of the need
    for l in reversed(range(L-1)):
        grads['dA' + str(l)], grads['dW' + str(l+1)], grads['db' + str(l+1)] = linear_activation_backward(grads['dA' + str(l+1)], caches[l], activation="relu")
    return grads

def update_parameters(params, grads, learning_rate):
    parameters = params.copy()
    L = len(parameters) // 2 # number of layers in the neural network
    for l in range(1,L+1):
        parameters['W' + str(l)] = parameters['W' + str(l)] -learning_rate*grads['dW' + str(l)]
        parameters['b' + str(l)] = parameters['b' + str(l)] -learning_rate*grads['db' + str(l)]
    return parameters

# L_layer_model
def L_layer_model(X, Y, layers_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False):
    np.random.seed(1)
    costs = []  # keep track of cost
    parameters = initialize_parameters_deep(layers_dims)
    costn_1 = np.inf

    for i in range(0, num_iterations):
        AL, caches = L_model_forward(X, parameters)
        cost = compute_cost(AL, Y, activation_out='softmax')
        grads = L_model_backward(AL, Y, caches)
        parameters = update_parameters(parameters, grads, learning_rate)

        if costn_1 < cost:
            print("Error, Alpha parameter to lessen")
            break
        costn_1 = cost
        # Print the cost every 100 iterations
        if i < 10:
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
        elif print_cost and i % 100 == 0 or i == num_iterations - 1:
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
            predictions = np.argmax(AL, axis=0)
            Y_reversed_one_hot = np.argmax(Y, axis=0)
            assert Y_reversed_one_hot.size == predictions.size
            print("Accuracy after iteration {}: {}".format(i, np.sum(Y_reversed_one_hot == predictions) / Y_reversed_one_hot.size))
        if i % 100 == 0 or i == num_iterations:
            costs.append(cost)

    return parameters, costs

def evaluate_prediction(Ytest, Ypred):
    predictions = np.argmax(Ypred, axis=0)
    assert Ytest.size == predictions.size
    return np.sum(Ytest == Ypred) / Ytest.size


if __name__=='main':
    # Training the model
    np.random.seed(1)  # pour que l'exécution soit déterministe

    data, labels = read_cifar(short=True)
    Xapp, Yapp, Xtest, Ytest = split_data(data, labels)
    X_train = Xapp.T
    Y_train = np.array([Yapp])
    X_test_clean = Xtest.T
    Y_test_clean = np.array([Ytest])
    n_x, m = X_train.shape[0], X_train.shape[1]
    uniques = np.unique(Yapp)
    n_y = len(uniques)
    Y_train_final = np.zeros((n_y, Y_train.shape[1]))
    for i in range(n_y):
        Y_train_final[i,:] = (Yapp == uniques[i])
    layers_dims = [n_x, 50, 30, n_y] # 0.0025 good for [50,25]  # Try 0.0001 bcs 0.0005 not working at 1500 it.
    parameters, costs = L_layer_model(X_train, Y_train_final, layers_dims, learning_rate=0.0004, num_iterations=3000, print_cost=True)

    pred_train = evaluate_prediction(X_train, Y_train, parameters)
    pred_test = evaluate_prediction(X_test_clean, Y_test_clean, parameters)
    print(f'NN accuracy on training set \033[1m{evaluate_prediction(X_train, Y_train, parameters):.3%}\033[0m')
    print(f'NN accuracy on test set \033[1m{evaluate_prediction(X_test_clean, Y_test_clean, parameters):.3%}\033[0m')