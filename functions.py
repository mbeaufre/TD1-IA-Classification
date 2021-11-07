import numpy as np

def softmax(Z):
    cache = Z
    Z -= np.max(Z)
    sm = (np.exp(Z) / np.sum(np.exp(Z), axis=0))
    return sm, cache

def sigmoid(Z):
    A = 1 / (1 + np.exp(-Z))
    cache = Z
    return A, cache

def relu(Z):
    A = np.maximum(0, Z)
    assert (A.shape == Z.shape)
    cache = Z
    return A, cache

def softmax_backward(dA, cache):
    z = cache
    z -= np.max(z)
    s = (np.exp(z) / np.sum(np.exp(z), axis=0))
    dZ = dA * s * (1 - s)
    return dZ

def sigmoid_backward(dA, cache):
    Z = cache
    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)
    assert (dZ.shape == Z.shape)
    return dZ

def relu_backward(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True)  # just converting dz to a correct object.
    # When z <= 0, you should set dz to 0 as well.
    dZ[Z <= 0] = 0
    assert (dZ.shape == Z.shape)
    return dZ