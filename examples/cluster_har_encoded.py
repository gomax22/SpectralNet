import torch
import numpy as np

from data import load_data
from spectralnet import SpectralNet
from spectralnet import Metrics 
from SpectralClustering import NgJordanWeissClustering, ShiMalikClustering
from sklearn.cluster import KMeans


def main():
    x_train, x_test, y_train, y_test = load_data("1478")

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

    # perform spectral clustering using SpectralNet
    spectralnet = SpectralNet(
        n_clusters=6,
        should_use_ae=True,
        should_use_siamese=True,
        ae_hiddens=[512, 512, 256, 6],
        ae_epochs=80,
        siamese_hiddens=[512, 512, 256, 6],
        siamese_n_nbg=25,
        siamese_use_approx=True,
        spectral_hiddens=[512, 512, 256, 6],
        spectral_lr=1e-4,
        spectral_is_normalized=True
    )
    
    spectralnet.fit(X, y)
    cluster_assignments_spectralnet = spectralnet.predict(X)
    print("Cluster assignments computed by SpectralNet")

    # get encoded representation of reduced HAR dataset
    X_encoded = spectralnet.ae_trainer.embed(X)

    # perform spectral clustering using ShiMalikClustering on the encoded representation
    shi_malik = ShiMalikClustering(n_clusters=6, n_neighbors=25)
    shi_malik.fit(X_encoded.detach().cpu().numpy())
    cluster_assignments_shimalik = shi_malik.get_cluster_assignments()
    print("Cluster assignments computed by Shi-Malik algorithm")


    # perform spectral clustering using NgJordanWeissClustering on the encoded representation
    ngjordanweiss = NgJordanWeissClustering(n_clusters=6, n_neighbors=25)
    ngjordanweiss.fit(X_encoded.detach().cpu().numpy())
    cluster_assignments_ngjordanweiss = ngjordanweiss.get_cluster_assignments()
    print("Cluster assignments computed by Ng-Jordan-Weiss algorithm")

    # perform KMeans clustering on the encoded representation
    kmeans = KMeans(n_clusters=6, n_init=10, max_iter=300).fit(X_encoded.detach().cpu().numpy())
    cluster_assignments_kmeans = kmeans.labels_

    report = {
        "SpectralNet": cluster_assignments_spectralnet,
        "ShiMalikClustering": cluster_assignments_shimalik,
        "NgJordanWeissClustering": cluster_assignments_ngjordanweiss,
        "KMeans": cluster_assignments_kmeans
    }

    # print clustering report
    Metrics.clustering_report(report, y.detach().cpu().numpy(), 6)

if __name__ == "__main__":
    main()