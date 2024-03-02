import torch
import numpy as np

from data import load_data

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def main():
    x_train, x_test, y_train, y_test = load_data("twomoons")
    X = torch.cat([x_train, x_test])

    if y_train is not None:
        y = torch.cat([y_train, y_test])
    else:
        y = None

    fig, ax = plt.subplots()
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="tab10", s=2)
    ax.grid(True)
    ax.set_title("Visualization of the twomoons dataset")
    fig.savefig("twomoons.png", dpi=400)
    plt.close(fig)


if __name__ == "__main__":
    main()