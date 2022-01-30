from dataset_generation import spiral_data
import matplotlib.pyplot as plt


def _show_data():
    X, y = spiral_data(points=100, classes=3)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='brg')
    plt.show()
