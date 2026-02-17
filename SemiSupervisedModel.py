import numpy as np
from scipy.sparse import lil_matrix, diags


class LabelPropagation:
    """
    Zhu & Ghahramani (2002) label propagation.

    Builds an RBF-kernel similarity graph over all data points (labeled +
    unlabeled), then iteratively spreads known labels through the graph.
    After convergence the unlabeled points receive soft pseudo-labels
    (probability distributions over classes).

    Steps:
      1. Compute pairwise affinity matrix W using RBF kernel:
         W_ij = exp(-||x_i - x_j||^2 / (2 * sigma^2))
      2. Build transition matrix T = D^{-1} W  (row-normalized)
      3. Iterate: Y <- T @ Y, then clamp labeled rows back to ground truth
      4. Return soft labels for unlabeled points

    Why the closed-form solution is infeasible:
    -----------------------------------------------
    The closed-form solution for label propagation is:
        Y_U = (I - T_UU)^{-1} @ T_UL @ Y_L

    where T_UU is the (n_u x n_u) submatrix of the transition matrix for
    unlabeled-to-unlabeled transitions, and T_UL is the (n_u x n_l) submatrix
    for unlabeled-to-labeled transitions.

    For Fashion-MNIST with n_u ~ 44,800 unlabeled samples:
      - T_UU is a 44,800 x 44,800 matrix
      - Dense storage: 44,800^2 * 4 bytes (float32) = ~8 GB
      - (I - T_UU)^{-1} requires inverting this matrix, which is O(n^3) in
        time and O(n^2) in memory — both completely infeasible
      - Even using sparse solvers (scipy.sparse.linalg.spsolve), the result
        matrix Y_U = (I - T_UU)^{-1} @ T_UL @ Y_L is dense (n_u x C), so
        the solve would need to factorize the sparse (I - T_UU) which fills
        in during LU decomposition, often exceeding available memory

    The iterative approach (Y <- T @ Y, clamp labeled) converges to the same
    fixed point but only requires O(k * n) memory for the sparse T matrix
    and O(n * C) for the label matrix Y, making it practical for large datasets.
    """

    def __init__(self, sigma=1.0, max_iter=50, k_neighbors=10, num_classes=10):
        self.sigma = sigma
        self.max_iter = max_iter
        self.k_neighbors = k_neighbors
        self.num_classes = num_classes

    def generate_pseudo_labels(self, X_labeled, y_labeled, X_unlabeled, threshold):
        """
        Run label propagation and return filtered pseudo-labels above threshold.

        Args:
            X_labeled:   (n_l, d) labeled features
            y_labeled:   (n_l,) integer labels
            X_unlabeled: (n_u, d) unlabeled features
            threshold:   confidence threshold (e.g. 0.95)

        Returns:
            X_pseudo: (n_p, d) pseudo-labeled features above threshold
            y_pseudo: (n_p,) hard pseudo-labels
            n_pseudo: number of pseudo-labels generated
        """
        pseudo_soft = self.propagate(X_labeled, y_labeled, X_unlabeled)

        confidence = pseudo_soft.max(axis=1)
        pseudo_hard = pseudo_soft.argmax(axis=1)
        mask = confidence >= threshold
        n_pseudo = int(mask.sum())

        print(f"  Pseudo-labels above {threshold}: {n_pseudo}/{len(X_unlabeled)} "
              f"({100*n_pseudo/len(X_unlabeled):.1f}%)")

        return X_unlabeled[mask], pseudo_hard[mask], n_pseudo

    def propagate(self, X_labeled, y_labeled, X_unlabeled):
        """
        Args:
            X_labeled:   (n_l, d) numpy array of labeled features
            y_labeled:   (n_l,) numpy array of integer labels
            X_unlabeled: (n_u, d) numpy array of unlabeled features

        Returns:
            pseudo_labels: (n_u, num_classes) soft label distributions
        """
        n_l = len(X_labeled)
        n_u = len(X_unlabeled)
        n = n_l + n_u

        # Combine all data: labeled first, then unlabeled
        X_all = np.concatenate([X_labeled, X_unlabeled], axis=0)

        # 1. Build sparse affinity matrix using k-nearest neighbors
        print(f"   Label propagation: building kNN graph (n={n}, k={self.k_neighbors})...")
        W = self._build_knn_affinity(X_all)

        # 2. Row-normalize: T = D^{-1} W (keep sparse!)
        row_sums = np.array(W.sum(axis=1)).flatten()
        row_sums[row_sums == 0] = 1.0
        D_inv = diags(1.0 / row_sums)  # Sparse diagonal matrix
        T = D_inv @ W  # Sparse @ sparse = sparse

        # 3. Initialize label matrix Y_L for labeled data only
        Y_L = np.zeros((n_l, self.num_classes))
        for i, label in enumerate(y_labeled.astype(int)):
            Y_L[i, label] = 1.0

        # 4. Use iterative method (closed-form requires dense n_u x n_u inverse — see docstring)
        print(f"   Label propagation: using iterative method...")
        return self._propagate_iterative(T, Y_L, n_l, n_u)

    def _propagate_iterative(self, T, Y_L, n_l, n_u):
        """Iterative label propagation (memory-efficient for large sparse graphs)."""
        n = n_l + n_u
        Y = np.full((n, self.num_classes), 1.0 / self.num_classes, dtype=np.float32)
        Y[:n_l] = Y_L

        print(f"      Iterating (max {self.max_iter} iterations)...")
        for it in range(self.max_iter):
            Y_new = T @ Y  # Sparse matrix @ dense array
            Y_new[:n_l] = Y_L  # Clamp labeled data
            delta = np.abs(Y_new - Y).max()
            Y = Y_new

            if (it + 1) % 10 == 0:
                print(f"      Iteration {it + 1}/{self.max_iter}: delta={delta:.2e}")

            if delta < 1e-6:
                print(f"      Converged at iteration {it + 1} (delta={delta:.2e})")
                break

        return Y[n_l:]

    def _build_knn_affinity(self, X):
        """Build a symmetric kNN affinity matrix using RBF kernel (sparse)."""
        n = len(X)
        W = lil_matrix((n, n), dtype=np.float32)

        chunk_size = 1000
        for i_start in range(0, n, chunk_size):
            i_end = min(i_start + chunk_size, n)
            X_chunk = X[i_start:i_end]

            chunk_norm = np.sum(X_chunk ** 2, axis=1, keepdims=True)
            X_norm = np.sum(X ** 2, axis=1, keepdims=True).T
            dists = chunk_norm + X_norm - 2 * np.dot(X_chunk, X.T)
            dists = np.maximum(dists, 0)

            for local_i, global_i in enumerate(range(i_start, i_end)):
                row_dists = dists[local_i]
                row_dists[global_i] = np.inf
                knn_idx = np.argpartition(row_dists, self.k_neighbors)[:self.k_neighbors]
                knn_dists = row_dists[knn_idx]
                affinities = np.exp(-knn_dists / (2 * self.sigma ** 2))
                W[global_i, knn_idx] = affinities

        W = W.tocsr()
        W = (W + W.T) / 2.0
        return W
