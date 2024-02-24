from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
import numpy
from decorators import debug

class ShiMalikClustering:
    def __init__(self, 
                 n_clusters: int = 8,
                 n_neighbors: int = 30,
                 knn_algorithm: str = 'auto',
                 knn_metric: str = 'euclidean',
                 kmeans_n_init: str = 'auto') -> None:
        self._n_clusters = n_clusters
        self._n_neighbors = n_neighbors
        self._knn_algorithm = knn_algorithm
        self._knn_metric = knn_metric
        self._kmeans_n_init = kmeans_n_init
        self._kmeans = None
        self._nn = None
        self._knn_graph = None
        self._X = None
        self._W = None
        self._D = None
        self._L = None
        self._eigvals = None
        self._eigvecs = None
        self._U = None


    @debug
    def _get_knn_graph(self, X: numpy.ndarray) -> numpy.ndarray:
        nn = NearestNeighbors(n_neighbors=self._n_neighbors, 
                                algorithm=self._knn_algorithm, 
                                metric=self._knn_metric).fit(X)
        knn_graph = nn.kneighbors_graph(X)
        return nn, knn_graph
    
    @debug
    def _get_affinity_matrix(self, X: numpy.ndarray) -> numpy.ndarray:
        self._nn, self._knn_graph = self._get_knn_graph(X)

        W = self._knn_graph.toarray()
        W = self._make_symmetric_affinity_matrix(W)

        return W

    @debug
    def _make_symmetric_affinity_matrix(self, W: numpy.ndarray) -> None:
        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                if W[i, j]:
                    W[j, i] = 1
        return W
    
    @debug
    def _get_degree_matrix(self, W: numpy.ndarray) -> numpy.ndarray:
        return numpy.diag(numpy.sum(W, axis=1))
    
    @debug
    def _get_normalized_laplacian_matrix(self, D: numpy.ndarray, W: numpy.ndarray) -> numpy.ndarray:
        return numpy.eye(W.shape[0]) - numpy.dot(numpy.linalg.inv(D), W)
    
    @debug
    def _solve_generalized_eigenproblem(self, L: numpy.ndarray) -> tuple[numpy.ndarray, numpy.ndarray]:
        eigvals, eigvecs = numpy.linalg.eigh(L)
        return eigvals, eigvecs

    @debug
    def _sort_eigenvectors(self) -> None:
        idx = numpy.argsort(self._eigvals)
        self._eigvals = self._eigvals[idx]
        self._eigvecs = self._eigvecs[:, idx]

    @debug
    def fit(self, X: numpy.ndarray) -> None:
        # Store the input data
        self._X = X.reshape(X.shape[0], -1)

        # Compute the affinity matrix    
        self._W = self._get_affinity_matrix(self._X)
    
        # compute the degree matrix
        self._D = self._get_degree_matrix(self._W)
        
        # Compute the normalized Laplacian matrix
        self._L = self._get_normalized_laplacian_matrix(self._D, self._W)

        # Solve the generalized eigenvalue problem
        self._eigvals, self._eigvecs = self._solve_generalized_eigenproblem(self._L)
        
        # Sort the eigenvectors in ascending order of eigenvalues
        self._sort_eigenvectors()

        # Select the first n_clusters eigenvectors
        self._U = self._eigvecs[:, :self._n_clusters]

        # Perform k-means clustering on the selected eigenvectors
        self._kmeans = KMeans(n_clusters=self._n_clusters, n_init=self._kmeans_n_init).fit(self._U)
    
    @debug
    def get_cluster_assignments(self) -> numpy.ndarray | None:
        return self._kmeans.labels_ if self._kmeans else None

class NgJordanWeissClustering:
    def __init__(self, 
                 n_clusters: int = 8,
                 n_neighbors: int = 30,
                 knn_algorithm: str = 'auto',
                 knn_metric: str = 'euclidean',
                 kmeans_n_init: str = 'auto',
                 precomputed_laplacian: numpy.ndarray | None = None) -> None:
        self._n_clusters = n_clusters
        self._n_neighbors = n_neighbors
        self._knn_algorithm = knn_algorithm
        self._knn_metric = knn_metric
        self._kmeans_n_init = kmeans_n_init
        self._kmeans = None
        self._nn = None
        self._knn_graph = None
        self._X = None
        self._W = None
        self._D = None
        self._L = None
        self._eigvals = None
        self._eigvecs = None
        self._U = precomputed_laplacian
        self._T = None

    @debug
    def _get_knn_graph(self, X: numpy.ndarray) -> numpy.ndarray:
        nn = NearestNeighbors(n_neighbors=self._n_neighbors, 
                                algorithm=self._knn_algorithm, 
                                metric=self._knn_metric).fit(X)
        knn_graph = nn.kneighbors_graph(X)
        return nn, knn_graph
    
    @debug
    def _get_affinity_matrix(self, X: numpy.ndarray) -> numpy.ndarray:
        self._nn, self._knn_graph = self._get_knn_graph(X)

        W = self._knn_graph.toarray()
        W = self._make_symmetric_affinity_matrix(W)

        return W

    @debug
    def _make_symmetric_affinity_matrix(self, W: numpy.ndarray) -> numpy.ndarray:
        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                if W[i, j]:
                    W[j, i] = 1
        return W
    
    @debug
    def _get_degree_matrix(self, W: numpy.ndarray) -> numpy.ndarray:
        return numpy.diag(numpy.sum(W, axis=1))
    
    @debug
    def _get_normalized_laplacian_matrix(self, D: numpy.ndarray, W: numpy.ndarray) -> numpy.ndarray:
        D_inv_sqrt = numpy.linalg.inv(numpy.sqrt(D))
        return numpy.eye(W.shape[0]) - numpy.dot(numpy.dot(D_inv_sqrt, W), D_inv_sqrt)
    
    @debug
    def _solve_generalized_eigenproblem(self, L: numpy.ndarray) -> tuple[numpy.ndarray, numpy.ndarray]:
        eigvals, eigvecs = numpy.linalg.eig(L)
        eigvals = eigvals.real
        eigvecs = eigvecs.real
        return eigvals, eigvecs
    
    @debug
    def _sort_eigenvectors(self) -> None:
        idx = numpy.argsort(self._eigvals)
        self._eigvals = self._eigvals[idx]
        self._eigvecs = self._eigvecs[:, idx]

    @debug
    def _normalize_rows(self, X: numpy.ndarray) -> numpy.ndarray:
        return X / numpy.linalg.norm(X, axis=1)[:, numpy.newaxis]
    
    @debug
    def fit(self, X: numpy.ndarray) -> None:

        # Store the input data
        self._X = X.reshape(X.shape[0], -1)

        if self._U is None:
            self._W = self._get_affinity_matrix(self._X)
            self._D = self._get_degree_matrix(self._W)
            self._L = self._get_normalized_laplacian_matrix(self._D, self._W)

            # Solve the generalized eigenvalue problem
            self._eigvals, self._eigvecs = self._solve_generalized_eigenproblem(self._L)

            # Sort the eigenvectors based on eigenvalues
            self._sort_eigenvectors()

            # Extract the first n_clusters eigenvectors
            self._U = self._eigvecs[:, :self._n_clusters]

        # Normalize the rows of U
        self._T = self._normalize_rows(self._U)

        # Perform K-means clustering on T
        self._kmeans = KMeans(n_clusters=self._n_clusters, n_init=self._kmeans_n_init).fit(self._T)
    
    @debug
    def get_cluster_assignments(self) -> numpy.ndarray | None:
        return self._kmeans.labels_ if self._kmeans else None