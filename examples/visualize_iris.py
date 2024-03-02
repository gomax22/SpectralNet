import torch
import numpy as np

from data import load_data

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def main():
    x_train, x_test, y_train, y_test = load_data("iris")

    X = torch.cat([x_train, x_test])

    if y_train is not None:
        y = torch.cat([y_train, y_test])
    else:
        y = None

    cluster_labels = {
        '$\\mathdefault{0}$':  "setosa",
        '$\\mathdefault{1}$': "versicolor",
        '$\\mathdefault{2}$': "virginica"
    }

    
    X_embedded = TSNE(n_components=2).fit_transform(X)

    fig, ax = plt.subplots()
    ax.grid(True)
    sc = ax.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y, cmap="tab10", s=6)
    L = ax.legend(*sc.legend_elements(), fontsize='small', loc='upper right')
    L.get_texts()[0].set_text(cluster_labels['$\\mathdefault{0}$'])
    L.get_texts()[1].set_text(cluster_labels['$\\mathdefault{1}$'])
    L.get_texts()[2].set_text(cluster_labels['$\\mathdefault{2}$'])
    ax.set_title("t-SNE visualization of iris dataset")
    fig.savefig("../figures/iris_tsne.png", dpi=400)
    plt.close(fig)


if __name__ == "__main__":
    main()