import numpy as np


class ReLU:
    def __init__(self):
        self.output = None
        self.gradInput = None
        self.inputs = None

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, gradOutput):
        self.gradInput = gradOutput.copy()
        self.gradInput[self.gradInput <= 0] = 0


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
        self.gradInput = None

    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)

    def backward(self, gradOutput):
        self.gradInput = np.einsum("bi, ij -> bij",
                                   self.output,
                                   np.eye(self.output.shape[1]))
        self.gradInput -= np.einsum("bi, bj -> bij",
                                    self.output,
                                    self.output)
        self.gradInput = np.einsum("bij, bi -> bj", self.gradInput, gradOutput)
