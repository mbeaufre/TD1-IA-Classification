import numpy as np
rng = np.random.default_rng()

FILES = [
    './cifar-10-batches-py/data_batch_1',
    './cifar-10-batches-py/data_batch_2',
    './cifar-10-batches-py/data_batch_3',
    './cifar-10-batches-py/data_batch_4',
    './cifar-10-batches-py/data_batch_5',
    './cifar-10-batches-py/test_batch',
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

def split_data(data, labels, split=0.8):
    N, D = data.shape
    hstack = np.hstack((data, labels.reshape(-1, 1)))
    rng.shuffle(hstack)
    hstack_app, hstack_test = hstack[:int(split * N), ...], hstack[int(split * N):, ...]
    return hstack_app[:, :D], hstack_app[:, -1], hstack_test[:, :D], hstack_test[:, -1]

def kppv_distance(A, B, squared=False):
    M = Xapp.shape[0]
    N = Xtest.shape[0]

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

def kppv_predict(dist, Yapp, K=1000):
    Napp, Ntest = dist.shape
    k_nearest = np.array(Yapp[np.argsort(dist, axis=0)[:K, ...]], dtype='int64')
    y_test = list()
    for i in range(Ntest):
        y_test.append(np.argmax(np.bincount(k_nearest[:, i], minlength=10)))
    return np.array(y_test, dtype='int32')

def evaluate_prediction(Ytest, Ypred):
    assert Ytest.size == Ypred.size
    return np.sum(Ytest == Ypred) / Ytest.size

if __name__ == '__main__':
    data, labels = read_cifar()
    Xapp, Yapp, Xtest, Ytest = split_data(data, labels)
    dist = kppv_distance(Xapp, Xtest)
    Ypred = kppv_predict(dist, Yapp, 100)
    print(f'KPPV accuracy \033[1m{evaluate_prediction(Ytest, Ypred):.3%}\033[0m')
