import torch
import numpy as np

from data import load_data

from spectralnet import SpectralNet
from spectralnet import Metrics
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering


def main():
    x_train, x_test, y_train, y_test = load_data("mnist")

    X = torch.cat([x_train, x_test])

    if y_train is not None:
        y = torch.cat([y_train, y_test])
    else:
        y = None
    
    ### SpectralNet
    spectralnet = SpectralNet(
        n_clusters=10,
        should_use_ae=True,
        should_use_siamese=True,
    )

    spectralnet.fit(X, y)
    cluster_assignments_spectralnet = spectralnet.predict(X)
    X_encoded = spectralnet.ae_trainer.embed(X)

    kmeans = KMeans(n_clusters=10, n_init=10, max_iter=300).fit(X_encoded.detach().cpu().numpy())
    cluster_assignments_kmeans = kmeans.predict(X_encoded.detach().cpu().numpy())

    spectral_clustering = SpectralClustering(n_clusters=10, affinity='nearest_neighbors', n_neighbors=25)
    cluster_assignments_spectral = spectral_clustering.fit_predict(X_encoded.detach().cpu().numpy())

    report = {
        "SpectralNet": cluster_assignments_spectralnet,
        "KMeans": cluster_assignments_kmeans,
        "SpectralClustering": cluster_assignments_spectral,
    }

    # print clustering report
    Metrics.clustering_report(report, y.detach().cpu().numpy(), n_clusters=10)


if __name__ == "__main__":
    embeddings, assignments = main()
