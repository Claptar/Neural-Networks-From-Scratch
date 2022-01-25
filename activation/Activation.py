import numpy as np


class ReLU:
    def __init__(self):
        self.output = None

    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

    def backward(self):
        pass


class Sigmoid:
    def __init__(self):
        self.output = None

    def forward(self, inputs):
        self.output = 1 / (1 + np.exp(-inputs))

    def backward(self):
        pass


class Softmax:
    def __init__(self):
        self.output = None

    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)

    def backward(self):
        pass


if __name__ == '__main__':
    layer_outputs = [4.8, 1.21, 2.385]
    softmax = Softmax()
    softmax.forward([[1, 2, 3]])
    print(softmax.output)

