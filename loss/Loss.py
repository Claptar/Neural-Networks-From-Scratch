import numpy as np
from activation.Activation import Softmax


class Loss:
    def forward(self, y_pred, y_true):
        pass

    def calculate(self, output, y_true):
        sample_losses = self.forward(output, y_true)
        data_loss = np.mean(sample_losses)
        return data_loss

    def backward(self, gradOutput, y_true):
        pass


class CrossEntropy(Loss):
    EPS = 1e-15

    def __init__(self, onehot=True):
        self.gradInput = None
        self.onehot = onehot

    def forward(self, y_pred, y_true):
        y_pred_clipped = np.clip(y_pred, self.EPS, 1 - self.EPS)
        if self.onehot:
            confidence = np.sum(y_pred_clipped * y_true, axis=1)
        else:
            n_samples = len(y_true)
            confidence = y_pred_clipped[range(n_samples), y_true]
        return - np.log(confidence)

    def backward(self, y_pred, y_true):
        normalized_input = np.clip(y_pred, self.EPS, 1 - self.EPS)
        self.gradInput = - (y_true / normalized_input) / y_pred.shape[0]


class SoftmaxCrossEntropy:
    def __init__(self, onehot=True):
        self.activation = Softmax()
        self.loss = CrossEntropy(onehot)
        self.output = None
        self.gradInput = None

    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        self.output = self.activation.output
        return self.loss.calculate(self.output, y_true)

    def backward(self, y_pred, y_true):
        n_samples = y_pred.shape[0]

        if self.loss.onehot:
            y_true = np.argmax(y_true, axis=1)

        self.gradInput = y_pred.copy()
        self.gradInput[range(n_samples), y_true] -= 1
        self.gradInput = self.gradInput / n_samples


class MSE:
    def __init__(self):
        self.output = None

    def __call__(self, inputs, y_true):
        return self.forward(inputs, y_true)

    def forward(self, inputs, y_true):
        self.output = (inputs - y_true).sum() / y_true.size()

    def backward(self):
        pass
