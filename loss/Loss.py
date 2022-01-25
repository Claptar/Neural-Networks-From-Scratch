import numpy as np


class CrossEntropy:
    def __init__(self):
        pass

    def forward(self, inputs, y_true):
        return - np.sum(y_true * np.log(inputs), axis=1)


class MSE:
    def __init__(self):
        self.output = None

    def forward(self, input, y_true):
        self.output = (input - y_true).sum() / y_true.size()

    def backward(self):
        pass


if __name__ == '__main__':
    crossentropy = CrossEntropy()
    print(crossentropy.forward([[0.7, 0.1, 0.2]], [[1, 0, 0]]))
