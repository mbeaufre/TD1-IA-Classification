import numpy as np
np.random.seed(1) # pour que l'exécution soit déterministe

def sigmoid(value):
    return 1. / (1. + np.exp(-value))

def sigmoid_derivative(value):
    return value * (1. - value)

def init(D_in, D_h, D_out):
    W1 = 2 * np.random.random((D_in, D_h)) - 1
    b1 = np.zeros((1,D_h))
    W2 = 2 * np.random.random((D_h, D_out)) - 1
    b2 = np.zeros((1,D_out))
    return W1, b1, W2, b2

def forward(X, W1, b1, W2, b2):
    I1 = X.dot(W1) + b1 # Potentiel d'entrée de la couche cachée
    O1 = sigmoid(I1) # Sortie de la couche cachée (fonction d'activation de type sigmoïde)
    I2 = O1.dot(W2) + b2 # Potentiel d'entrée de la couche de sortie
    O2 = sigmoid(I2)# Sortie de la couche de sortie (fonction d'activation de type sigmoïde)
    return O1, O2 # Les valeurs prédites sont les sorties de la couche de sortie

def backward(X, Y, O1, O2):
    DO2 = Y - O2
    DI2 = sigmoid_derivative(O2) * DO2

    DW2 = O1.T.dot(DI2)
    Db2 = np.sum(DI2, axis=0)

    DO1 = DI2.dot(W2.T)
    DI1 = sigmoid_derivative(DO1) * DO1

    DW1 = X.T.dot(DI1)
    Db1 = np.sum(DI1, axis=0)

    return DW1, Db1, DW2, Db2

def loss(Y_pred, Y):
    return np.square(Y_pred - Y).sum() / 2

if __name__  == '__main__':
    # Génération des données
    # N est le nombre de données d'entrée
    # D_in est la dimension des données d'entrée
    # D_h le nombre de neurones de la couche cachée

    # D_out est la dimension de sortie (nombre de neurones de la couche de sortie)
    N, D_in, D_h, D_out = 30, 2, 10, 3

    # Création d'une matrice d'entrée X et de sortie Y avec des valeurs aléatoires
    X = np.random.random((N, D_in))
    Y = np.random.random((N, D_out))

    # Initialisation aléatoire des poids du réseau
    W1, b1, W2, b2 = init(D_in, D_h, D_out)

    a = 0.005
    for i in range(1000):
        # Passe avant : calcul de la sortie prédite Y_pred
        O1, O2 = forward(X, W1, b1, W2, b2)
        # Calcul et affichage de la fonction perte de type MSE
        print(i, loss(O2, Y))
        DW1, Db1, DW2, Db2 = backward(X, Y, O1, O2)
        W1, b1, W2, b2 = W1 + DW1 * a, b1 + Db1 * a, W2 + DW2 * a, b2 + Db2 * a
