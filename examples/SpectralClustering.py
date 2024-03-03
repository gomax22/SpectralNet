from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
import numpy

class BaseSpectralClustering:
    """
    Base class for spectral clustering algorithms.

    Parameters:
    - n_clusters (int): The number of clusters to find.
    - n_neighbors (int): The number of nearest neighbors to consider when constructing the affinity matrix.
    - precomputed (bool): Whether the affinity matrix is precomputed or not.

    Attributes:
    - n_clusters (int): The number of clusters to find.
    - n_neighbors (int): The number of nearest neighbors to consider when constructing the affinity matrix.
    - X (numpy.ndarray): The input data.
    - kmeans (KMeans): The KMeans clustering model.
    - affinity_matrix (numpy.ndarray): The affinity matrix.
    - degree_matrix (numpy.ndarray): The degree matrix.
    - laplacian_matrix (numpy.ndarray): The Laplacian matrix.
    - laplacian_eigenvalues (numpy.ndarray): The eigenvalues of the Laplacian matrix.
    - laplacian_eigenvectors (numpy.ndarray): The eigenvectors of the Laplacian matrix.
    """

    def __init__(self, n_clusters: int = 8, n_neighbors: int = 30, precomputed: bool = False) -> None:
        """
        Initialize the SpectralClustering object.

        Args:
            n_clusters (int): The number of clusters to form.
            n_neighbors (int): The number of nearest neighbors to consider when constructing the affinity matrix.
            precomputed (bool): Whether the affinity matrix is precomputed or not.

        Returns:
            None
        """
        self.n_clusters: int = n_clusters
        self.n_neighbors: int = n_neighbors
        self.precomputed: bool = precomputed
        self.X: numpy.ndarray = None
        self.kmeans: KMeans = None
        self.affinity_matrix: numpy.ndarray = None
        self.degree_matrix: numpy.ndarray = None
        self.laplacian_matrix: numpy.ndarray = None
        self.laplacian_eigenvalues: numpy.ndarray = None
        self.laplacian_eigenvectors: numpy.ndarray = None
    
    def _get_affinity_matrix(self, X: numpy.ndarray) -> numpy.ndarray:
        """
        Compute the affinity matrix for Spectral Clustering.

        This method computes the affinity matrix based on the k-nearest neighbor graph.
        The affinity matrix represents the similarity between data points in the input dataset.

        Parameters:
            X (numpy.ndarray): The input dataset of shape (n_samples, n_features).

        Returns:
            numpy.ndarray: The affinity matrix of shape (n_samples, n_samples).

        References:
            - A Tutorial on Spectral Clustering by Ulrike von Luxburg, 2007 (Section 2.2)

        Notes:
            The affinity matrix is computed as follows:
            1. Compute the k-nearest neighbors graph using the input dataset.
            2. Compute the affinity matrix by setting the weights of the edges based on the similarity of their endpoints.
            3. Return the symmetric affinity matrix.

        """
        # compute k-nearest neighbors graph
        self.estimator = NearestNeighbors(n_neighbors=self.n_neighbors).fit(X)
        
        # compute the affinity matrix
        affinity_matrix = self.estimator.kneighbors_graph(mode="distance").toarray()

        # return the symmetric affinity matrix
        return numpy.maximum(affinity_matrix, affinity_matrix.T)
    
    def _get_degree_matrix(self, W: numpy.ndarray) -> numpy.ndarray:
        """
        Compute the degree matrix of the affinity matrix.

        Parameters:
        - W (numpy.ndarray): Affinity matrix.

        Returns:
        - numpy.ndarray: Degree matrix.
        """
        assert numpy.allclose(W, W.T), "Affinity matrix must be symmetric"
        assert numpy.all(W >= 0), "Affinity matrix must be non-negative"

        return numpy.diag(numpy.sum(W, axis=1))
    
    def _get_normalized_laplacian_matrix(self, D: numpy.ndarray, W: numpy.ndarray) -> numpy.ndarray:
        """
        Abstract method for computing the normalized Laplacian matrix.

        Parameters:
        - D (numpy.ndarray): The degree matrix.
        - W (numpy.ndarray): The adjacency matrix.

        Returns:
        - numpy.ndarray: The normalized Laplacian matrix.
        """
        raise NotImplementedError
    
    def _solve_eigenproblem(self, L: numpy.ndarray) -> tuple[numpy.ndarray, numpy.ndarray]:
        """
        Abstract method for solving the eigenproblem for the given Laplacian matrix.

        Parameters:
        - L (numpy.ndarray): The Laplacian matrix.

        Returns:
        - tuple[numpy.ndarray, numpy.ndarray]: A tuple containing the eigenvalues and eigenvectors.
        """
        raise NotImplementedError
    
    def _sort_laplacian_eigenvectors(self) -> None:
        """
        Sorts the computed laplacian eigenvectors and eigenvalues in ascending order.

        Raises:
            AssertionError: If eigenvalues or eigenvectors are not computed.
        """
        assert self.laplacian_eigenvalues is not None, "Eigenvalues not computed"
        assert self.laplacian_eigenvectors is not None, "Eigenvectors not computed"

        idx = numpy.argsort(self.laplacian_eigenvalues)
        self.laplacian_eigenvalues = self.laplacian_eigenvalues[idx]
        self.laplacian_eigenvectors = self.laplacian_eigenvectors[:, idx]

    def fit(self, X: numpy.ndarray) -> None:
        """
        Abstract method for fitting the SpectralClustering model to the given data.

        Parameters:
        X (numpy.ndarray): The input data array of shape (n_samples, n_features).

        Returns:
        None
        """
        raise NotImplementedError
    
    def get_cluster_assignments(self) -> numpy.ndarray:
            """
            Returns the cluster assignments computed by KMeans.

            Raises:
                AssertionError: If KMeans has not been computed.

            Returns:
                numpy.ndarray: The cluster assignments.
            """
            assert self.kmeans is not None, "KMeans not computed"
            return self.kmeans.labels_



class ShiMalikClustering(BaseSpectralClustering):
    """
    Spectral clustering algorithm based on the Shi-Malik method.

    Parameters:
    - n_clusters (int): The number of clusters to form.
    - n_neighbors (int): The number of nearest neighbors to consider when constructing the affinity matrix.
    - precomputed (bool): Whether the affinity matrix is precomputed or not.

    Attributes:
    - X (numpy.ndarray): The input data.
    - affinity_matrix (numpy.ndarray): The affinity matrix.
    - degree_matrix (numpy.ndarray): The degree matrix.
    - laplacian_matrix (numpy.ndarray): The normalized Laplacian matrix.
    - laplacian_eigenvalues (numpy.ndarray): The eigenvalues of the Laplacian matrix.
    - laplacian_eigenvectors (numpy.ndarray): The eigenvectors of the Laplacian matrix.
    - kmeans (KMeans): The k-means clustering model.

    Methods:
    - fit(X: numpy.ndarray) -> None: Fit the model to the input data.

    """

    def __init__(self, n_clusters: int = 8, n_neighbors: int = 30, precomputed: bool = False) -> None:
        """
        Initialize the SpectralClustering object.

        Parameters:
        - n_clusters (int): The number of clusters to form.
        - n_neighbors (int): The number of nearest neighbors to consider for each sample.
        - precomputed (bool): Whether the affinity matrix is precomputed or not.

        Returns:
        None
        """
        super().__init__(n_clusters, n_neighbors, precomputed)

    def _get_normalized_laplacian_matrix(self, D: numpy.ndarray, W: numpy.ndarray) -> numpy.ndarray:
        """
        Compute the normalized Laplacian matrix.

        Parameters:
        - D (numpy.ndarray): The degree matrix.
        - W (numpy.ndarray): The affinity matrix.

        Returns:
        - numpy.ndarray: The normalized Laplacian matrix.

        """
        assert D is not None, "Degree matrix not computed"
        assert W is not None, "Affinity matrix not computed"
        
        return numpy.eye(W.shape[0]) - numpy.dot(numpy.linalg.inv(D), W)
    
    def _solve_eigenproblem(self, L: numpy.ndarray) -> tuple[numpy.ndarray, numpy.ndarray]:
        """
        Solve the generalized eigenvalue problem.

        Parameters:
        - L (numpy.ndarray): The Laplacian matrix.

        Returns:
        - tuple[numpy.ndarray, numpy.ndarray]: The eigenvalues and eigenvectors.

        """
        assert L is not None, "Laplacian matrix not computed"

        eigvals, eigvecs = numpy.linalg.eig(L)
        eigvals = eigvals.real
        eigvecs = eigvecs.real

        return eigvals, eigvecs

    def fit(self, X: numpy.ndarray) -> None:
        """
        Fit the model to the input data. 

        Parameters:
        - X (numpy.ndarray): The input data.

        Notes:
        - If precomputed == True, the input data is assumed to be the affinity matrix, 
        otherwise, the affinity matrix is computed from the input data.

        Returns:
        - None

        """

        if self.precomputed:
            # Store the input data
            self.X = X.reshape(X.shape[0], -1)

            # Compute the affinity matrix    
            self.affinity_matrix = self._get_affinity_matrix(self.X)
        else:
            self.affinity_matrix = X
    
        # compute the degree matrix
        self.degree_matrix = self._get_degree_matrix(self.affinity_matrix)
        
        # Compute the normalized Laplacian matrix
        self.laplacian_matrix = self._get_normalized_laplacian_matrix(self.degree_matrix, self.affinity_matrix)

        # Solve the generalized eigenvalue problem
        self.laplacian_eigenvalues, self.laplacian_eigenvectors = self._solve_eigenproblem(self.laplacian_matrix)
        
        # Sort the eigenvectors in ascending order of eigenvalues
        self._sort_laplacian_eigenvectors()

        # Perform k-means clustering on the first n_clusters eigenvectors
        self.kmeans = KMeans(n_clusters=self.n_clusters, n_init=10).fit(self.laplacian_eigenvectors[:, :self.n_clusters])
    

class NgJordanWeissClustering(BaseSpectralClustering):
    """
    Spectral clustering algorithm based on the Ng, Jordan, and Weiss method.
    
    Parameters:
    - n_clusters (int): The number of clusters to form.
    - n_neighbors (int): The number of nearest neighbors to consider when constructing the affinity matrix.
    - precomputed (bool): Whether the affinity matrix is precomputed or not.
    
    Attributes:
    - X (numpy.ndarray): The input data.
    - affinity_matrix (numpy.ndarray): The affinity matrix.
    - degree_matrix (numpy.ndarray): The degree matrix.
    - laplacian_matrix (numpy.ndarray): The normalized Laplacian matrix.
    - laplacian_eigenvalues (numpy.ndarray): The eigenvalues of the Laplacian matrix.
    - laplacian_eigenvectors (numpy.ndarray): The eigenvectors of the Laplacian matrix.
    - normalized_laplacian_eigenspace (numpy.ndarray): The normalized reduced Laplacian eigenspace.
    - kmeans (KMeans): The k-means clustering model.

    Methods:
    - fit(X: numpy.ndarray) -> None: Fit the model to the input data.
    
    """
    def __init__(self, n_clusters: int = 8, n_neighbors: int = 30, precomputed: bool = False) -> None:
        """
        Initialize a SpectralClustering object.

        Parameters:
        - n_clusters (int): The number of clusters to form.
        - n_neighbors (int): The number of nearest neighbors to consider when constructing the affinity matrix.

        Returns:
        None
        """
        super().__init__(n_clusters, n_neighbors, precomputed)
        self.normalized_laplacian_eigenspace = None
    
    def _get_normalized_laplacian_matrix(self, D: numpy.ndarray, W: numpy.ndarray) -> numpy.ndarray:
        """
        Compute the normalized Laplacian matrix.

        Parameters:
            D (numpy.ndarray): Degree matrix.
            W (numpy.ndarray): Affinity matrix.

        Returns:
            numpy.ndarray: Normalized Laplacian matrix.
        """
        assert D is not None, "Degree matrix not computed"
        assert W is not None, "Affinity matrix not computed"
        
        D_inv_sqrt = numpy.linalg.inv(numpy.sqrt(D))
        return numpy.eye(W.shape[0]) - numpy.dot(numpy.dot(D_inv_sqrt, W), D_inv_sqrt)
    
    def _solve_generalized_eigenproblem(self, L: numpy.ndarray) -> tuple[numpy.ndarray, numpy.ndarray]:
        """
        Solves the generalized eigenvalue problem for the given Laplacian matrix.

        Args:
            L (numpy.ndarray): The Laplacian matrix.

        Returns:
            tuple[numpy.ndarray, numpy.ndarray]: A tuple containing the eigenvalues and eigenvectors.

        Raises:
            AssertionError: If the Laplacian matrix is not computed.
        """
        assert L is not None, "Laplacian matrix not computed"
        
        eigvals, eigvecs = numpy.linalg.eigh(L)
        return eigvals, eigvecs
    
    def _normalize_laplacian_eigenspace(self, E: numpy.ndarray) -> numpy.ndarray:
        """
        Normalize the given Laplacian eigenspace.

        Args:
            E (numpy.ndarray): The Laplacian eigenspace to be normalized.

        Returns:
            numpy.ndarray: The normalized Laplacian eigenspace.

        Raises:
            AssertionError: If the Laplacian eigenspace is not computed.
        """
        assert E is not None, "Laplacian eigenspace not computed"
        return E / numpy.linalg.norm(E, axis=1)[:, numpy.newaxis]
    
    def fit(self, X: numpy.ndarray) -> None:
        """
        Fit the spectral clustering model to the input data.
        
        Parameters:
            X (numpy.ndarray): The input data matrix of shape (n_samples, n_features).

        Notes:
            If precomputed == True, the input data is assumed to be the affinity matrix,
                otherwise, the affinity matrix is computed from the input data.

        Returns:
            None
        """

        if self.precomputed:
            # Store the input data
            self.X = X.reshape(X.shape[0], -1)

            # compute the affinity matrix
            self.affinity_matrix = self._get_affinity_matrix(self.X)
        else:
            self.affinity_matrix = X
        
        # compute the degree matrix
        self.degree_matrix = self._get_degree_matrix(self.affinity_matrix)
        
        # Compute the normalized Laplacian matrix
        self.laplacian_matrix = self._get_normalized_laplacian_matrix(self.degree_matrix, self.affinity_matrix)

        # Solve the generalized eigenvalue problem
        self.laplacian_eigenvalues, self.laplacian_eigenvectors = self._solve_generalized_eigenproblem(self.laplacian_matrix)

        # Sort the eigenvectors based on eigenvalues
        self._sort_laplacian_eigenvectors()

        # Normalize the rows of the reduced laplacian eigenspace
        self.normalized_laplacian_eigenspace = self._normalize_laplacian_eigenspace(self.laplacian_eigenvectors[:, :self.n_clusters])

        # Perform K-means clustering on the normalized reduced laplacian eigenspace
        self.kmeans = KMeans(n_clusters=self.n_clusters, n_init=10).fit(self.normalized_laplacian_eigenspace)
    