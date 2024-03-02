import torch
import numpy as np

from data import load_data

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING
def main():
    x_train, x_test, y_train, y_test = load_data("1478")

    X = torch.cat([x_train, x_test])

    if y_train is not None:
        y = torch.cat([y_train, y_test])
    else:
        y = None
    
    X = X.view(X.size(0), -1)

    # create balanced dataset where each class has 1000 samples
    for class_idx in range(6):
        class_indices = torch.where(y == class_idx)[0]
        class_indices = np.random.choice(class_indices, 1000, replace=False)

        if class_idx == 0:
            X_balanced = X[class_indices]
            y_balanced = y[class_indices]
        else:
            X_balanced = torch.cat([X_balanced, X[class_indices]])
            y_balanced = torch.cat([y_balanced, y[class_indices]])

    X = X_balanced
    y = y_balanced


    cluster_labels = {
        '$\\mathdefault{0}$':  "WALKING",    
        '$\\mathdefault{1}$': "WALKING_UPSTAIRS",
        '$\\mathdefault{2}$': "WALKING_DOWNSTAIRS",
        '$\\mathdefault{3}$': "SITTING",
        '$\\mathdefault{4}$': "STANDING", 
        '$\\mathdefault{5}$': "LAYING"
    }

    X_embedded = TSNE(n_components=2).fit_transform(X)

    fig, ax = plt.subplots()
    ax.grid(True)
    sc = ax.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y, cmap='viridis', s=2)
    ax.set_title("t-SNE visualization of HAR dataset")
    L = ax.legend(*sc.legend_elements(), fontsize='small', loc='upper right')
    L.get_texts()[0].set_text(cluster_labels['$\\mathdefault{0}$'])
    L.get_texts()[1].set_text(cluster_labels['$\\mathdefault{1}$'])
    L.get_texts()[2].set_text(cluster_labels['$\\mathdefault{2}$'])
    L.get_texts()[3].set_text(cluster_labels['$\\mathdefault{3}$'])
    L.get_texts()[4].set_text(cluster_labels['$\\mathdefault{4}$'])
    L.get_texts()[5].set_text(cluster_labels['$\\mathdefault{5}$'])
    fig.savefig("har_tsne.png", dpi=400)

    plt.close(fig)

if __name__ == "__main__":
    main()
    