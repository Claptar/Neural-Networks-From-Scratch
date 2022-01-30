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
    def __init__(self, in_features, out_features):
        stdv = 1. / np.sqrt(in_features)
        self.weights = np.random.uniform(-stdv, stdv, size=(in_features, out_features))
        self.bias = np.random.uniform(-stdv, stdv, size=out_features)
        self.inputs = None
        self.gradW = np.zeros_like(self.weights)
        self.gradb = np.zeros_like(self.bias)
        self.gradInput = None
        self.output = None

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.bias

    def backward(self, gradOutput):
        self.gradW = self.inputs.T @ gradOutput
        self.gradb = np.sum(gradOutput, axis=0)
        self.gradInput = gradOutput @ self.weights.T



