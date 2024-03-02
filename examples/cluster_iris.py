import torch
import numpy as np

from data import load_data

from spectralnet import Metrics
from spectralnet import SpectralNet
from SpectralClustering import ShiMalikClustering, NgJordanWeissClustering
from sklearn.cluster import KMeans

def main():
    x_train, x_test, y_train, y_test = load_data("iris")

    X = torch.cat([x_train, x_test])

    if y_train is not None:
        y = torch.cat([y_train, y_test])
    else:
        y = None
    
    ### SpectralNet
    spectralnet = SpectralNet(
        n_clusters=3,
        should_use_ae=False,
        should_use_siamese=True,
        siamese_batch_size=256,
        siamese_hiddens=[128, 128, 3],
        siamese_n_nbg = 25,
        siamese_use_approx=True,
        siamese_epochs=50,
        spectral_hiddens=[128, 128, 3],
        spectral_lr=5e-5,
        spectral_patience=10,
        spectral_epochs=60,
        spectral_is_local_scale=False,
    )

    spectralnet.fit(X, y)
    cluster_assignments_spectralnet = spectralnet.predict(X)
    print("Cluster assignments computed by SpectralNet")

    # perform spectral clustering by means of Shi and Malik algorithm
    shimalik = ShiMalikClustering(n_clusters=3, n_neighbors=25)
    shimalik.fit(X.detach().cpu().numpy())
    cluster_assignments_shimalik = shimalik.get_cluster_assignments()
    print("Cluster assignments computed by Shi-Malik algorithm")

    # perform spectral clustering by means of Ng, Jordan, and Weiss algorithm
    ngjordanweiss = NgJordanWeissClustering(n_clusters=3, n_neighbors=25)
    ngjordanweiss.fit(X.detach().cpu().numpy())
    cluster_assignments_ngjordanweiss = ngjordanweiss.get_cluster_assignments()
    print("Cluster assignments computed by Ng-Jordan-Weiss algorithm")


    # perform KMeans clustering
    kmeans = KMeans(n_clusters=3, n_init=10, max_iter=300).fit(X.detach().cpu().numpy())
    cluster_assignments_kmeans = kmeans.labels_
    print("Cluster assignments computed by KMeans")

    report = {
        "SpectralNet": cluster_assignments_spectralnet,
        "ShiMalikClustering": cluster_assignments_shimalik,
        "NgJordanWeissClustering": cluster_assignments_ngjordanweiss,
        "KMeans": cluster_assignments_kmeans,
    }


    # print clustering report
    Metrics.clustering_report(report, y.detach().cpu().numpy(), 3)

if __name__ == "__main__":
    main()
