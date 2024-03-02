import torch
import numpy as np

from data import load_data

from spectralnet import SpectralNet
from SpectralClustering import ShiMalikClustering, NgJordanWeissClustering
from sklearn.cluster import KMeans
from spectralnet import Metrics
import matplotlib.pyplot as plt

def main():
    x_train, x_test, y_train, y_test = load_data("twomoons")
    X = torch.cat([x_train, x_test])

    if y_train is not None:
        y = torch.cat([y_train, y_test])
    else:
        y = None

    n_clusters = 2
    spectralnet = SpectralNet(
        n_clusters=n_clusters,
        should_use_ae=False,
        should_use_siamese=False,
        spectral_batch_size=712,
        spectral_epochs=40,
        spectral_is_local_scale=False,
        spectral_n_nbg=8,
        spectral_scale_k=2,
        spectral_lr=1e-2,
        spectral_hiddens=[128, 128, 2],
    )
    
    spectralnet.fit(X, y)
    cluster_assignments_spectralnet = spectralnet.predict(X)
    print("Cluster assignments computed by SpectralNet")


    # perform spectral clustering by means of Shi and Malik algorithm
    shi_malik = ShiMalikClustering(n_clusters=n_clusters, n_neighbors=8)
    shi_malik.fit(X.detach().cpu().numpy())
    cluster_assignments_shi_malik = shi_malik.get_cluster_assignments()
    print("Cluster assignments computed by Shi-Malik algorithm")

    # perform spectral clustering by means of Ng, Jordan, and Weiss algorithm
    ng_jordan_weiss = NgJordanWeissClustering(n_clusters=n_clusters, n_neighbors=8)
    ng_jordan_weiss.fit(X.detach().cpu().numpy())
    cluster_assignments_ng_jordan_weiss = ng_jordan_weiss.get_cluster_assignments()
    print("Cluster assignments computed by Ng-Jordan-Weiss algorithm")
    
    # perform KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, n_init=10).fit(X.detach().cpu().numpy())
    cluster_assignments_kmeans = kmeans.predict(X.detach().cpu().numpy())
    print("Cluster assignments computed by KMeans")


    report = {
        "SpectralNet": cluster_assignments_spectralnet,
        "ShiMalikClustering": cluster_assignments_shi_malik,
        "NgJordanWeissClustering": cluster_assignments_ng_jordan_weiss,
        "KMeans": cluster_assignments_kmeans,
    }

    # print clustering report
    Metrics.clustering_report(report, y.detach().cpu().numpy(), n_clusters)


if __name__ == "__main__":
    main()
