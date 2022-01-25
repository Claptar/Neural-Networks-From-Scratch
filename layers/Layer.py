import numpy as np
from datasets import spiral_data
import matplotlib.pyplot as plt


class Sequential:
    def __init__(self):
        pass

    def forward(self):
        pass

    def backward(self):
        pass


class Dense:
    def __init__(self, in_features, out_features, bias=True):
        self.weights = 0.1 * np.random.randn(in_features, out_features)
        if bias:
            self.bias = np.zeros((1, out_features))
        self.inputs = None
        self.grad = None

    def forward(self, inputs):
        if isinstance(self.bias, np.ndarray):
            self.inputs = np.hstack([inputs, np.full(shape=(1, self.bias.shape[1]), fill_value=1)])
        self.inputs = inputs
        return np.dot(inputs, self.weights) + self.bias

    def backward(self):
        self.grad = inputs.T
        return self.grad


if __name__ == '__main__':
    X, y = spiral_data(points=100, classes=3)
    print(X.shape)


def _show_data():
    X, y = spiral_data(points=100, classes=3)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='brg')
    plt.show()
