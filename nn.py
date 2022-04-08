import numpy as np
from chxtorch import *
import matplotlib.pyplot as plt


class Net:
    def __init__(self, layerList, learning_rate=.001, momentum=0.0, penalty=0.0, criterion=negLogLihood, activation='Sigmoid', dropout=False, silent=False, need_plot=False):
        if activation == 'Sigmoid':
            activation = Sigmoid
        if activation == 'ReLU':
            activation = ReLU
        self.fc = LinearSequence(layerList, learning_rate, momentum, penalty, dropout, activation=activation)
        self.criterion = criterion()
        self.silent = silent
        self.need_plot = need_plot
        if need_plot:
            self.train_loss = []
            self.train_acc = []
            self.test_loss = []
            self.test_acc = []

    def forward(self, x, training=True):
        x = self.fc(x, training)
        return x

    def load_params(self, params):
        i = 0
        for layer in self.fc.layers:
            if isinstance(layer, Linear):
                layer.W = params[i]['W']
                layer.b = params[i]['b']
                i = i + 1

    def backward(self, g):
        g = self.fc.backward(g)
        return g

    def params(self):
        params = []
        for layer in self.fc.layers:
            if isinstance(layer, Linear):
                params.append({'W':layer.W, 'b':layer.b})
        return params
                
    def train(self, X_train, y_train, X_test, y_test, epoch=4):
        length = len(X_train)
        iterations = int(epoch * length)
        self.iterations = iterations

        for i in range(iterations):
            # 抽样
            sample = np.random.randint(length)
            x, y = X_train[sample].reshape(1, -1), y_train[sample].reshape(1, -1)
            pred = self.forward(x, training=True)
            loss = self.criterion.forloss(pred, y)

            g = self.criterion.backloss()

            self.backward(g)

            # learning rate decay
            for layer in self.fc.layers:
                if isinstance(layer, Linear):
                    layer.learning_rate *= 0.9 ** 0.0001

            if ((i % (iterations // 10) == 0)):
                accuracy = self.acc(X_test, y_test)
                      
            if not self.silent and (i % (iterations // 10) == 0):               
                print('iteration', i, ': test acc:', accuracy)

            if self.need_plot and (i % (iterations // 100) == 0):
                self.train_loss.append(self.loss(X_train, y_train))
                self.train_acc.append(self.acc(X_train, y_train))
                self.test_loss.append(self.loss(X_test, y_test))
                self.test_acc.append(self.acc(X_test, y_test))

        accuracy = self.acc(X_test, y_test)
                      
        if not self.silent:               
            print('iteration', iterations, ': test acc:', accuracy)

        if self.need_plot:
            self.train_loss.append(self.loss(X_train, y_train))
            self.train_acc.append(self.acc(X_train, y_train))
            self.test_loss.append(self.loss(X_test, y_test))
            self.test_acc.append(self.acc(X_test, y_test))

        return accuracy


    def acc(self, x, y):
        prob = self.forward(x, training=False)
        pred = np.argmax(prob, axis=1)
        truth = np.argmax(y, axis=1)
        correct = np.equal(pred, truth).sum()

        accuracy = correct / len(x)
        return accuracy

    def loss(self, x, y):
        pred = self.forward(x, training=False)
        loss = self.criterion.forloss(pred, y)

        loss /= len(x)
        return loss

    def show_loss_plot(self, path):
        iterations = self.iterations
        train_loss = self.train_loss
        test_loss = self.test_loss
        i = np.arange(0, 101, 1) * (iterations / 100)
        plt.plot(i, train_loss, i, test_loss)
        plt.grid()
        plt.title("Loss")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.legend(["Training set", "Test set"])
        plt.savefig(path)

    def show_acc_plot(self, path):
        iterations = self.iterations
        train_acc = self.train_acc
        test_acc = self.test_acc
        i = np.arange(0, 101, 1) * (iterations / 100)
        plt.plot(i, train_acc, i, test_acc)
        plt.grid()
        plt.title("acc")
        plt.xlabel("Iteration")
        plt.ylabel("acc")
        plt.legend(["Training set", "Test set"])
        plt.savefig(path)