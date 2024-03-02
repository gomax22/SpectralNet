import torch
import numpy as np

from data import load_data

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def main():
    x_train, x_test, y_train, y_test = load_data("mnist")

    X = torch.cat([x_train, x_test])

    if y_train is not None:
        y = torch.cat([y_train, y_test])
    else:
        y = None
    
    X = X.view(X.size(0), -1)
    X_embedded = TSNE(n_components=2).fit_transform(X)

    fig, ax = plt.subplots()
    ax.grid(True)
    sc = ax.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y, cmap="tab10", s=2)
    ax.set_title("t-SNE visualization of MNIST dataset")
    ax.legend(*sc.legend_elements(), fontsize='small', loc='upper right')
    fig.savefig("mnist_tsne.png", dpi=400)
    plt.close(fig)

if __name__ == "__main__":
    main()
    