import numpy as np
from skimage.feature import local_binary_pattern, hog
import matplotlib.pyplot as plt
rng = np.random.default_rng()

FILES = [
    '../cifar-10-batches-py/data_batch_1',
    '../cifar-10-batches-py/data_batch_2',
    '../cifar-10-batches-py/data_batch_3',
    '../cifar-10-batches-py/data_batch_4',
    '../cifar-10-batches-py/data_batch_5',
    '../cifar-10-batches-py/test_batch',
]

def unpickle(filename):
    import pickle
    with open(filename, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def read_cifar(short=False):
    data, labels = None, None
    for filename in FILES:
        extracted_file = unpickle(filename)
        local_data, local_labels = np.array(extracted_file[b'data'], dtype='float32'), np.array(extracted_file[b'labels'], dtype='float32')
        if type(data) == np.ndarray and type(labels) == np.ndarray:
            data, labels = np.vstack((data, local_data)), np.hstack((labels, local_labels))
        else:
            data, labels = local_data, local_labels
            if short:
                return data, labels
    return data, labels


def transform_data_to_LBP(data):
    return np.apply_along_axis(transform_image_to_LBP, 1, data)

def transform_image_to_LBP(image_raw):
    # Settings for LBP
    METHOD = 'uniform'
    radius = 3
    n_points = 8 * radius

    # Extract colors
    size = image_raw.size // 3
    image_raw_red = image_raw[:size]
    image_raw_green = image_raw[size: 2 * size]
    image_raw_blue = image_raw[2 * size: 3 * size]

    # grayscale = 0.299red + 0.587green + 0.114blue
    image_raw_gray = 0.299 * image_raw_red + 0.587 * image_raw_green + 0.114 * image_raw_blue

    # Reshape image
    image = image_raw_gray.reshape(32,32)

    # LBP
    lbp = local_binary_pattern(image, n_points, radius, METHOD)

    return lbp.ravel()

def transform_data_to_HOG(data):
    return np.apply_along_axis(transform_image_to_HOG, 1, data)

def transform_image_to_HOG(image_raw):
    # Reshape
    image = image_raw.reshape(32, 32, 3)

    # HOG
    fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True, multichannel=True)

    return hog_image.ravel()


def split_data(data, labels, split=0.8):
    N, D = data.shape
    hstack = np.hstack((data, labels.reshape(-1, 1)))
    rng.shuffle(hstack)
    hstack_app, hstack_test = hstack[:int(split * N), ...], hstack[int(split * N):, ...]
    return hstack_app[:, :D], hstack_app[:, -1], hstack_test[:, :D], hstack_test[:, -1]

def multiple_split_data(data, labels, cv=5):
    N, D = data.shape
    hstack = np.hstack((data, labels.reshape(-1, 1)))
    rng.shuffle(hstack)
    split = int(N / cv)
    hstacks = list()
    for i in range(cv):
        local_hstack = hstack[int(split * i):int(split * (i + 1)), ...]
        local_hstack_data, local_hstack_label = local_hstack[:, :D], local_hstack[:, -1]
        hstacks.append((local_hstack_data, local_hstack_label))
    return hstacks

def kppv_distance(A, B, squared=False):
    M = A.shape[0]
    N = B.shape[0]

    assert A.shape[1] == B.shape[1], f"The number of components for vectors in A \
        {A.shape[1]} does not match that of B {B.shape[1]}!"

    A_dots = (A*A).sum(axis=1).reshape((M,1))*np.ones(shape=(1,N))
    B_dots = (B*B).sum(axis=1)*np.ones(shape=(M,1))
    D_squared =  A_dots + B_dots -2*A.dot(B.T)

    if squared == False: # Remove negative value to avoid computation errors
        zero_mask = np.less(D_squared, 0.0)
        D_squared[zero_mask] = 0.0
        return np.sqrt(D_squared)

    return D_squared

def kppv_predict(dist, Yapp, k=1000):
    Napp, Ntest = dist.shape
    k_nearest = np.array(Yapp[np.argsort(dist, axis=0)[:k, ...]], dtype='int64')
    y_test = list()
    for i in range(Ntest):
        y_test.append(np.argmax(np.bincount(k_nearest[:, i], minlength=10)))
    return np.array(y_test, dtype='int32')

def evaluate_prediction(Ytest, Ypred):
    assert Ytest.size == Ypred.size
    return np.sum(Ytest == Ypred) / Ytest.size # Accuracy

def get_accuracy(data, labels, k, verbose=False):
    Xapp, Yapp, Xtest, Ytest = split_data(data, labels)
    dist = kppv_distance(Xapp, Xtest)
    print(dist.shape, Yapp.shape)
    Ypred = kppv_predict(dist, Yapp, k)
    accuracy = evaluate_prediction(Ytest, Ypred)
    if verbose:
        print(f'KPPV accuracy \033[1m{accuracy:.3%}\033[0m')
    return accuracy

def get_cross_accuracy(data, labels, k, cv=5, verbose=False):
    splits = multiple_split_data(data, labels, cv) # (X, Y)

    # Create all distance matrix for each fold
    # One fold is: all "app" but one "test"
    accuracies = list()
    for i in range(cv):
        xys = [(s[0], s[1]) for j, s in enumerate(splits) if j != i]
        Xapp, Yapp = None, None
        for x, y in xys:
            if type(Xapp) == np.ndarray and type(Yapp) == np.ndarray:
                Xapp, Yapp = np.vstack((Xapp, x)), np.hstack((Yapp, y))
            else:
                Xapp, Yapp = x, y
        # test -> splits[i] and app -> rest
        dist = kppv_distance(Xapp, splits[i][0], k)
        pred = kppv_predict(dist, Yapp, k)
        accuracy = evaluate_prediction(splits[i][1], pred)
        if verbose:
            print(f'\tFold {i + 1} -> accuracy \033[1m{accuracy:.3%}\033[0m')
        accuracies.append(accuracy)
    mean_accuracy = sum(accuracies) / len(accuracies)
    if verbose:
        print(f'Accuracy \033[1m{mean_accuracy:.3%}\033[0m')
    return mean_accuracy

def accuracy_vs_k(data, labels, from_k=1, to_k=50, step=5, verbose=False, use_cross=False, cv=5):
    accuracies = list()
    ks = list(range(from_k, to_k + 1, step))
    for k in ks:
        print(f'### k:{k:3d}')
        if use_cross:
            accuracies.append(get_cross_accuracy(data, labels, k, cv, verbose))
        else:
            accuracies.append(get_accuracy(data, labels, k, verbose))
    plt.plot(ks, accuracies)
    plt.show()


if __name__ == '__main__':
    data, labels = read_cifar(True)
    data = transform_data_to_HOG(data)
    accuracy_vs_k(data, labels, from_k=1, to_k=100, step=10, verbose=True, use_cross=True)
