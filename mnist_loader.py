from torchvision import datasets
import numpy as np
from transform import oneHot

def mnistLoader():
    mnist_train = datasets.MNIST('./data_mnist',
                   train=True,
                   download=True)
    mnist_test = datasets.MNIST('./data_mnist',
                   train=False,
                   download=True)
    train_set_array = mnist_train.data.numpy()
    test_set_array = mnist_test.data.numpy()
    train_set_array_targets = mnist_train.targets.numpy()
    test_set_array_targets = mnist_test.targets.numpy()

    X_train = ((train_set_array / 255) - 0.1307) / 0.3081
    Y_train = train_set_array_targets
    X_test = ((test_set_array / 255) - 0.1307) / 0.3081
    Y_test = test_set_array_targets

    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)
    Y_train = oneHot(Y_train.reshape(-1, 1))
    Y_test = oneHot(Y_test.reshape(-1, 1))

    return X_train, Y_train, X_test, Y_test