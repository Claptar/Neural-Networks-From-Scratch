import numpy as np
import matplotlib.pyplot as plt
from activation import ReLU
from datasets import spiral_data
from layers import Dense
from loss import SoftmaxCrossEntropy
from metrics import accuracy
from optimizers import SGD

if __name__ == '__main__':
    X, y = spiral_data(points=100, classes=3)

# Net initialization
    dense1 = Dense(2, 64)
    activation1 = ReLU()
    dense2 = Dense(64, 3)
# Loss-func initialization
    loss_activation = SoftmaxCrossEntropy(onehot=False)
# Optimizer initialization
    optimizer = SGD()

    for epoch in range(10001):
        # forward pass
        dense1.forward(X)
        activation1.forward(dense1.output)
        dense2.forward(activation1.output)

        # Compute loss
        loss_output = loss_activation.forward(dense2.output, y)

        # Computing accuracy
        predictions = np.argmax(loss_activation.output, axis=1)
        acc = accuracy(predictions, y)

        # Backpropagation
        loss_activation.backward(loss_activation.output, y)
        dense2.backward(loss_activation.gradInput)
        activation1.backward(dense2.gradInput)
        dense1.backward(activation1.gradInput)

        # Update params
        optimizer.update_params(dense1)
        optimizer.update_params(dense2)

        if not epoch % 100:
            print(f'epoch: {epoch}')
            print(f'loss: {loss_output}')
            print(f'acc: {acc}')


def _show_data():
    X, y = spiral_data(points=100, classes=3)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='brg')
    plt.show()
