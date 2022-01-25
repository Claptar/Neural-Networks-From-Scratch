import numpy as np


class Loss:
    def forward(self, y_pred, y_true):
        pass

    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss


class CrossEntropy(Loss):
    def __init__(self):
        pass

    def forward(self, y_pred, y_true, onehot=True):
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        if onehot:
            confidence = np.sum(y_pred_clipped * y_true, axis=1)
        else:
            n_samples = len(y_true)
            confidence = y_pred_clipped[range(n_samples), y_true]
        return - np.log(confidence)


class MSE:
    def __init__(self):
        self.output = None

    def __call__(self, inputs, y_true):
        return self.forward(inputs, y_true)

    def forward(self, inputs, y_true):
        self.output = (inputs - y_true).sum() / y_true.size()

    def backward(self):
        pass


if __name__ == '__main__':
    crossentropy = CrossEntropy()
    print(crossentropy.forward([[0.7, 0.1, 0.2]], [[1, 0, 0]]))
