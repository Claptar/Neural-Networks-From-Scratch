import numpy as np


class SGD:
    def __init__(self, learning_rate=1e-1):
        self.learning_rate = learning_rate

    def update_params(self, layer):
        layer.weights += - self.learning_rate * layer.gradW
        layer.bias += - self.learning_rate * layer.gradb
