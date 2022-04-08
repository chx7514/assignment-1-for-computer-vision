import scipy.io
import numpy as np
from scipy import signal


class Module:
    """模板"""
    def __init__(self):
        self.x = 0
        self.y = 0

    def __call__(self, x, training=True):
        return self.forward(x, training)

    def forward(self, x, training=True):
        pass

    def backward(self, g):
        pass


class Sigmoid(Module):
    def forward(self, x, training=True):
        self.x = x
        self.y = 1 / (1 + np.exp(-x))
        return self.y

    def backward(self, g):
        return g * self.y * (1 - self.y)


class ReLU(Module):
    def forward(self, x, training=True):
        self.x = x
        self.y = np.maximum(0, x)
        return self.y

    def backward(self, g):
        g[self.y < 0] = 0
        return g


class Tanh(Module):
    def forward(self, x, training=True):
        self.x = x
        self.y = np.tanh(x)
        return self.y

    def backward(self, g):
        return g * (1 - self.y ** 2)


class Linear(Module):
    def __init__(self, num_in, num_out, learning_rate=.01, momentum=0.0, penalty=0.0):
        super().__init__()
        self.W = np.random.randn(num_in, num_out) / np.sqrt(num_in + num_out)
        self.b = np.random.randn(1, num_out) / np.sqrt(num_out)
        self.v = np.zeros(self.W.shape)
        self.u = np.zeros(self.b.shape)

        self.learning_rate = learning_rate
        self.momentum = momentum
        self.penalty = penalty

    def forward(self, x, training=True):
        self.x = x
        self.y = np.matmul(x, self.W) + self.b
        return self.y

    def backward(self, g):
        dx = np.matmul(g, self.W.T)
        dW = np.matmul(self.x.T, g)
        db = g

        dW += self.penalty * self.W / np.linalg.norm(self.W, 'fro')
        db += self.penalty * self.b / np.linalg.norm(self.b, 'fro')

        self.v = self.momentum * self.v - self.learning_rate * dW
        self.u = self.momentum * self.u - self.learning_rate * db

        self.W = self.W + self.v
        self.b = self.b + self.u

        return dx


class LinearWithoutBias(Module):
    def __init__(self, num_in, num_out, learning_rate=.01, momentum=0.0, penalty=0.0):
        super().__init__()
        self.W = np.random.rand(num_in, num_out)
        self.v = np.zeros(self.W.shape)
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.penalty = penalty

    def forward(self, x, training=True):
        self.x = x
        self.y = np.matmul(x, self.W)
        return self.y

    def backward(self, g):
        dx = np.matmul(g, self.W.T)
        dW = np.matmul(self.x.T, g)

        dW += self.penalty * self.W / np.linalg.norm(self.W, 'fro')

        self.v = self.momentum * self.v - self.learning_rate * dW
        self.W = self.W + self.v

        return dx


class LinearSequence(Module):
    def __init__(self, layerList, learning_rate=.001, momentum=0.0, penalty=0.0, dropout=False, activation=Sigmoid):
        super().__init__()
        self.layerList = layerList
        self.layers = []

        num_in = layerList[0]
        for i in range(1, len(layerList) - 1):
            num_out = layerList[i]
            li = Linear(num_in, num_out, learning_rate, momentum, penalty)
            if dropout:
                di = Dropout()
                self.layers.append(di)
            ri = activation()
            self.layers.append(li)
            self.layers.append(ri)
            num_in = num_out

        l_last = Linear(num_in, layerList[len(layerList) - 1], learning_rate, momentum, penalty)
        self.layers.append(l_last)

    def forward(self, x, training=True):
        for layer in self.layers:
            x = layer.forward(x, training)
        self.y = x
        return x

    def backward(self, g):
        for layer in self.layers[::-1]:
            g = layer.backward(g)
        return g


def padding(x, P):
    y = np.pad(x, ((0, 0), (P, P), (P, P)), "constant")
    return y


def point_conv(a, W, b):
    z = np.sum(np.sum(a * W) + b)
    return z


class Conv(Module):
    def __init__(self, Cin, Cout, K, P, lr=0.001, momentum=0, penalty=0):
        super().__init__()
        self.Cin = Cin
        self.Cout = Cout
        self.K = K
        self.P = P
        self.lr = lr
        self.momentum = momentum
        self.penalty = penalty
        self.W = np.random.rand(Cin, Cout, K, K)
        self.b = np.random.rand(Cout)
        self.v = np.zeros(self.W.shape)
        self.u = np.zeros(self.b.shape)

    def forward(self, x, training=True):
        self.x = x
        Cin, H, W = x.shape
        K, Cin, Cout, P, K = self.K, self.Cin, self.Cout, self.P, self.K
        x_pad = padding(x, P)
        self.x_pad = x_pad
        Hout = H - K + 2 * P + 1
        Wout = W - K + 2 * P + 1

        y = np.zeros((Cout, Hout, Wout))
        for cout in range(Cout):
            for cin in range(Cin):
                y[cout, :, :] = y[cout, :, :] + signal.convolve2d(x_pad[cin, :, :], self.W[cin, cout, :, :], 'valid')
            y[cout, :, :] = y[cout, :, :] + self.b[cout] * np.ones((1, Hout, Wout))
        self.y = y
        return y

    def backward(self, g):
        _, H, W = g.shape
        K, Cin, Cout, P, K = self.K, self.Cin, self.Cout, self.P, self.K
        g_pad = padding(g, K - 1)
        dx = np.zeros(self.x.shape)

        for cin in range(Cin):
            for cout in range(Cout):
                dx[cin, :, :] += signal.convolve2d(g_pad[cout, :, :], np.rot90(np.rot90(self.W[cin, cout, :, :])).T, 'valid')

        dx = dx[:, P:-P, P:-P]

        dW = np.zeros(self.W.shape)
        for ih in range(K):
            for iw in range(K):
                dW[:, :, ih, iw] = np.dot(self.x_pad[:, ih:ih+H, iw:iw+W].transpose([0, 1, 2]).reshape(self.x_pad.shape[0], -1), g.transpose([1,2,0]).reshape(-1, g.shape[0]))

        db = np.einsum('ijk->i', g)

        dW = dW / (H * W * Cin)
        db = db / (H * W * Cin)
        dx = dx / (K * K)

        self.v = self.momentum * self.v - self.lr * dW
        self.u = self.momentum * self.u - self.lr * db

        self.W = self.W + self.v
        self.b = self.b + self.u

        return dx


class Flatten(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, training=True):
        self.x = x
        self.y = x.reshape(1, -1)
        return self.y

    def backward(self, g):
        g = g.reshape(self.x.shape)
        return g


class LossModule:
    def __init__(self):
        super().__init__()
        self.l = 0
        self.y_hat = 0
        self.y = 0

    def forloss(self, y_hat, y):
        pass

    def backloss(self):
        pass


class MSELoss(LossModule):
    def forloss(self, y_hat, y):
        self.y_hat = y_hat
        self.y = y
        self.l = ((y_hat - y) ** 2).sum() / 2
        return self.l

    def backloss(self):
        return self.y_hat - self.y


class negLogLihood(LossModule):
    def __init__(self):
        super().__init__()
        self.p = 0

    def forloss(self, y_hat, y):
        self.y_hat = y_hat
        self.y = y
        p = np.exp(y_hat)
        p = p / np.sum(p, axis=1).reshape(-1, 1)
        l = np.multiply(-np.log(p), y).sum()
        self.p, self.l = p, l
        return self.l

    def backloss(self):
        return self.p - self.y


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.mask = 0
        self.p = p

    def forward(self, x, training=True):
        if training:
            self.mask = (np.random.random(x.shape) < self.p) / self.p
            self.y = x * self.mask
        else:
            self.mask = np.ones(x.shape)
            self.y = x
        return self.y

    def backward(self, g):
        return self.mask * g


if __name__ == '__main__':
    pass
