import numpy as np
import sklearn.metrics as metrics

from munkres import Munkres
from sklearn.metrics import normalized_mutual_info_score as nmi

from spectralnet._utils import *


class Metrics:
    @staticmethod
    def acc_score(
        cluster_assignments: np.ndarray, y: np.ndarray, n_clusters: int, fname: str = None
    ) -> float:
        """
        Compute the accuracy score of the clustering algorithm.

        Parameters
        ----------
        cluster_assignments : np.ndarray
            Cluster assignments for each data point.
        y : np.ndarray
            Ground truth labels.
        n_clusters : int
            Number of clusters.
        fname : str, optional
            File name for saving the confusion matrix plot.

        Returns
        -------
        float
            The computed accuracy score.

        Notes
        -----
        This function takes the `cluster_assignments` which represent the assigned clusters for each data point,
        the ground truth labels `y`, and the number of clusters `n_clusters`. It computes the accuracy score of the
        clustering algorithm by comparing the cluster assignments with the ground truth labels. The accuracy score
        is returned as a floating-point value.

        If `fname` is provided, the confusion matrix plot will be saved with the given file name. Otherwise, the
        confusion matrix will be printed to the console.

        Examples
        --------
        >>> cluster_assignments = np.array([0, 1, 1, 0, 2, 2])
        >>> y = np.array([0, 1, 1, 0, 2, 2])
        >>> n_clusters = 3
        >>> acc_score(cluster_assignments, y, n_clusters)
        1.0
        """
        confusion_matrix = metrics.confusion_matrix(y, cluster_assignments, labels=None)
        cost_matrix = calculate_cost_matrix(confusion_matrix, n_clusters=n_clusters)
        indices = Munkres().compute(cost_matrix)
        kmeans_to_true_cluster_labels = get_cluster_labels_from_indices(indices)
        y_pred = kmeans_to_true_cluster_labels[cluster_assignments]
        cm = metrics.confusion_matrix(y, y_pred)
        if fname:
            plot_confusion_matrix(cm, np.unique(y), fname)
        else:
            print(cm)
        accuracy = np.mean(y_pred == y)
        return accuracy

    @staticmethod
    def nmi_score(cluster_assignments: np.ndarray, y: np.ndarray) -> float:
        """
        Compute the normalized mutual information score of the clustering algorithm.

        Parameters
        ----------
        cluster_assignments : np.ndarray
            Cluster assignments for each data point.
        y : np.ndarray
            Ground truth labels.

        Returns
        -------
        float
            The computed normalized mutual information score.

        Notes
        -----
        This function takes the `cluster_assignments` which represent the assigned clusters for each data point
        and the ground truth labels `y`. It computes the normalized mutual information (NMI) score of the clustering
        algorithm. NMI measures the mutual dependence between the cluster assignments and the ground truth labels,
        normalized by the entropy of both variables. The NMI score ranges between 0 and 1, where a higher score
        indicates a better clustering performance. The computed NMI score is returned as a floating-point value.
        """
        return nmi(cluster_assignments, y)
    
    @staticmethod
    def ari_score(cluster_assignments: np.ndarray, y: np.ndarray) -> float:
        """
        Compute the adjusted rand index score of the clustering algorithm.

        Parameters
        ----------
        cluster_assignments : np.ndarray
            Cluster assignments for each data point.
        y : np.ndarray
            Ground truth labels.

        Returns
        -------
        float
            The computed adjusted rand index score.

        Notes
        -----
        This function takes the `cluster_assignments` which represent the assigned clusters for each data point
        and the ground truth labels `y`. It computes the adjusted rand index (ARI) score of the clustering
        algorithm. ARI measures the similarity between the cluster assignments and the ground truth labels, adjusted
        for chance. The ARI score ranges between -1 and 1, where a higher score indicates a better clustering performance.
        The computed ARI score is returned as a floating-point value.
        """
        return metrics.adjusted_rand_score(y, cluster_assignments)
    
    @staticmethod
    def clustering_report(report: dict, y: np.ndarray, n_clusters: int, savefig: bool = False) -> None:
        for method_name, cluster_assignments in report.items():
            print(f"\nClustering report for {method_name}")
            acc_score = Metrics.acc_score(cluster_assignments, y, n_clusters, fname=method_name if savefig else None)
            nmi_score = Metrics.nmi_score(cluster_assignments, y)
            ari_score = Metrics.ari_score(cluster_assignments, y)
            print(f"ACC ({method_name}): {np.round(acc_score, 3)}")
            print(f"NMI ({method_name}): {np.round(nmi_score, 3)}")
            print(f"ARI ({method_name}): {np.round(ari_score, 3)}\n")


