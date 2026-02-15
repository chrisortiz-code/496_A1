import numpy as np

class PCA:
    """
    Principal Component Analysis (from scratch).

    Parameters
    ----------
    k : int
        Number of principal components to retain.
    """

    def __init__(self, k):
        self.k = k
        self.mean_ = None
        self.components_ = None
        self.eigenvalues_ = None

    def fit(self, X):
        """
        Fit PCA on dataset X.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        """
        n, d = X.shape
        if self.k > d:
            raise ValueError("k cannot exceed number of features")

        # 1. Center the data
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        # 2. Compute covariance matrix
        cov = (1 / n) * X_centered.T @ X_centered

        # 3. Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # 4. Sort in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # 5. Keep top-k
        self.eigenvalues_ = eigenvalues[:self.k]
        self.components_ = eigenvectors[:, :self.k]

        return self

    def transform(self, X):
        """
        Project X onto top-k principal components.

        Returns
        -------
        Z : ndarray of shape (n_samples, k)
        """
        if self.components_ is None:
            raise RuntimeError("PCA not fitted yet.")

        X_centered = X - self.mean_
        Z = X_centered @ self.components_
        return Z

    def fit_transform(self, X):
        """
        Fit PCA and return projected data.
        """
        self.fit(X)
        return self.transform(X)
