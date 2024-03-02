import torch
import numpy as np
from examples.data import load_data
from spectralnet import SpectralNet
from SpectralClustering import NgJordanWeissClustering, ShiMalikClustering
from sklearn.cluster import KMeans
from spectralnet import Metrics

def main():
    x_train, x_test, y_train, y_test = load_data("1478") # har dataset

    X = torch.cat([x_train, x_test])
    y = torch.cat([y_train, y_test])

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

    spectralnet = SpectralNet(
        n_clusters=6,
        should_use_ae=False,
        should_use_siamese=True,
        siamese_hiddens=[512, 512, 256],
        siamese_n_nbg=10,
        siamese_use_approx=True,
        spectral_hiddens=[512, 512, 256, 6],
        spectral_lr=1e-4,
    )

    spectralnet.fit(X, y)
    cluster_assignments_spectralnet = spectralnet.predict(X)
    print("Cluster assignments computed by SpectralNet")

    shimalik = ShiMalikClustering(n_clusters=6, n_neighbors=10)
    shimalik.fit(X.detach().cpu().numpy())
    cluster_assignments_shimalik = shimalik.get_cluster_assignments()
    print("Cluster assignments computed by Shi-Malik algorithm")

    ngjordanweiss = NgJordanWeissClustering(n_clusters=6, n_neighbors=10)
    ngjordanweiss.fit(X.detach().cpu().numpy())
    cluster_assignments_ngjordanweiss = ngjordanweiss.get_cluster_assignments()
    print("Cluster assignments computed by Ng-Jordan-Weiss algorithm")

    kmeans = KMeans(n_clusters=6, n_init=10, max_iter=300).fit(X.detach().cpu().numpy())
    cluster_assignments_kmeans = kmeans.labels_
    print("Cluster assignments computed by KMeans")


    report = {
        "SpectralNet": cluster_assignments_spectralnet,
        "ShiMalikClustering": cluster_assignments_shimalik,
        "NgJordanWeissClustering": cluster_assignments_ngjordanweiss,
        "KMeans": cluster_assignments_kmeans,
    }

    y = y.detach().cpu().numpy()

    # print clustering report
    Metrics.clustering_report(report, y, 6)


if __name__ == "__main__":
    main()