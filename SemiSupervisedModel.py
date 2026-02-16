import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.sparse import lil_matrix, diags
from scipy.sparse.linalg import spsolve

TORCH_INIT_STRATEGIES = {
    "he": lambda w: w.data.normal_(0, np.sqrt(2.0 / w.shape[1])),
    "uniform": lambda w: nn.init.uniform_(w, -1 / np.sqrt(w.shape[1]), 1 / np.sqrt(w.shape[1])),
    "normal": lambda w: nn.init.normal_(w, mean=0.0, std=0.01),
}


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

            if (i_end) % 5000 == 0 or i_end == n:
                print(f"      Processed {i_end}/{n} samples...")

        W = W.tocsr()
        W = (W + W.T) / 2.0
        return W


class CoTraining:
    """
    Co-training with two views (Blum & Mitchell, 1998).

    For images: View A = left half (columns 0-13), View B = right half (columns 14-27).
    Each view has 28 * 14 = 392 features.

    Algorithm:
      1. Train model_A on View A of labeled data, model_B on View B of labeled data
      2. Each round:
         a. Both models predict on unlabeled data
         b. If model A is confident but B is not → add to B's training set
         c. If model B is confident but A is not → add to A's training set
      3. Final prediction: average softmax of both models

    Uses Stage2Model-style architecture (with dual regularization) for each view.
    """

    IMG_SIZE = 28
    HALF_WIDTH = 14
    VIEW_DIM = IMG_SIZE * HALF_WIDTH  # 392

    def __init__(self, hidden_size=512, init_strategy="he", lr=0.01, momentum=0.04,
                 lambda1=1e-5, lambda2=1e-5, num_classes=10):
        self.hidden_size = hidden_size
        self.init_strategy = init_strategy
        self.lr = lr
        self.momentum = momentum
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.num_classes = num_classes

    @staticmethod
    def split_views(X_flat):
        """
        Split flattened 28x28 images into left and right halves.

        Args:
            X_flat: (n, 784) numpy array or torch tensor

        Returns:
            view_a: (n, 392) left half
            view_b: (n, 392) right half
        """
        if isinstance(X_flat, torch.Tensor):
            imgs = X_flat.view(-1, 28, 28)
            view_a = imgs[:, :, :14].reshape(-1, 392)
            view_b = imgs[:, :, 14:].reshape(-1, 392)
        else:
            imgs = X_flat.reshape(-1, 28, 28)
            view_a = imgs[:, :, :14].reshape(-1, 392)
            view_b = imgs[:, :, 14:].reshape(-1, 392)
        return view_a, view_b

    def _make_model(self, device):
        """Create a Stage2Model-style network for a single view (392 inputs)."""
        from Stage2Model import Stage2Model
        return Stage2Model(
            input_size=self.VIEW_DIM, hidden_size=self.hidden_size,
            output_size=self.num_classes, init_strategy=self.init_strategy,
            lr=self.lr, momentum=self.momentum,
            lambda1=self.lambda1, lambda2=self.lambda2
        ).to(device)

    def train_cotrain(self, X_labeled, y_labeled, X_unlabeled,
                      val_loader_full, test_loader_full, device,
                      num_rounds=5, train_epochs_per_round=30,
                      confidence_threshold=0.90, max_add_per_round=500):
        """
        Run the co-training algorithm.

        Args:
            X_labeled:   (n_l, 784) numpy array
            y_labeled:   (n_l,) integer labels
            X_unlabeled: (n_u, 784) numpy array
            val_loader_full: DataLoader for validation (full 784 features)
            test_loader_full: DataLoader for test (full 784 features)
            device: torch device
            num_rounds: number of co-training iterations
            train_epochs_per_round: epochs to train each model per round
            confidence_threshold: min softmax prob to count as "confident"
            max_add_per_round: max samples to add per model per round

        Returns:
            dict with test_acc, per-round stats
        """
        from torch.utils.data import TensorDataset, DataLoader as TDL
        import platform
        nw = 0 if platform.system() == 'Windows' else 2

        # Split into views
        X_l_a, X_l_b = self.split_views(X_labeled)
        X_u_a, X_u_b = self.split_views(X_unlabeled)

        # Mutable copies of labeled sets for each model
        labeled_a_X = X_l_a.copy()
        labeled_a_y = y_labeled.copy()
        labeled_b_X = X_l_b.copy()
        labeled_b_y = y_labeled.copy()

        unlabeled_a = X_u_a.copy()
        unlabeled_b = X_u_b.copy()
        # Track which unlabeled indices are still available
        available_mask = np.ones(len(X_unlabeled), dtype=bool)

        model_a = self._make_model(device)
        model_b = self._make_model(device)

        round_stats = []

        for rnd in range(num_rounds):
            print(f"   --- Co-training round {rnd+1}/{num_rounds} ---")
            print(f"      Model A labeled: {len(labeled_a_y)} | Model B labeled: {len(labeled_b_y)}")

            # Train model A on its labeled set
            ds_a = TensorDataset(torch.FloatTensor(labeled_a_X), torch.LongTensor(labeled_a_y))
            loader_a = TDL(ds_a, batch_size=64, shuffle=True, num_workers=nw)
            for ep in range(train_epochs_per_round):
                model_a.train_epoch(loader_a, device)

            # Train model B on its labeled set
            ds_b = TensorDataset(torch.FloatTensor(labeled_b_X), torch.LongTensor(labeled_b_y))
            loader_b = TDL(ds_b, batch_size=64, shuffle=True, num_workers=nw)
            for ep in range(train_epochs_per_round):
                model_b.train_epoch(loader_b, device)

            # Predict on remaining unlabeled data
            avail_idx = np.where(available_mask)[0]
            if len(avail_idx) == 0:
                print("      No unlabeled data left.")
                break

            with torch.no_grad():
                model_a.eval()
                model_b.eval()

                u_a_tensor = torch.FloatTensor(unlabeled_a[avail_idx]).to(device)
                u_b_tensor = torch.FloatTensor(unlabeled_b[avail_idx]).to(device)

                probs_a = torch.softmax(model_a(u_a_tensor), dim=1).cpu().numpy()
                probs_b = torch.softmax(model_b(u_b_tensor), dim=1).cpu().numpy()

            conf_a = probs_a.max(axis=1)
            conf_b = probs_b.max(axis=1)
            pred_a = probs_a.argmax(axis=1)
            pred_b = probs_b.argmax(axis=1)

            # A confident, B not → add to B's training set
            a_confident = (conf_a >= confidence_threshold) & (conf_b < confidence_threshold)
            a_to_b_idx = avail_idx[a_confident][:max_add_per_round]

            # B confident, A not → add to A's training set
            b_confident = (conf_b >= confidence_threshold) & (conf_a < confidence_threshold)
            b_to_a_idx = avail_idx[b_confident][:max_add_per_round]

            n_a_to_b = len(a_to_b_idx)
            n_b_to_a = len(b_to_a_idx)

            if n_a_to_b > 0:
                labeled_b_X = np.concatenate([labeled_b_X, unlabeled_b[a_to_b_idx]])
                labeled_b_y = np.concatenate([labeled_b_y, pred_a[a_confident][:max_add_per_round]])
                available_mask[a_to_b_idx] = False

            if n_b_to_a > 0:
                labeled_a_X = np.concatenate([labeled_a_X, unlabeled_a[b_to_a_idx]])
                labeled_a_y = np.concatenate([labeled_a_y, pred_b[b_confident][:max_add_per_round]])
                available_mask[b_to_a_idx] = False

            print(f"      A→B: {n_a_to_b} | B→A: {n_b_to_a} | Remaining unlabeled: {available_mask.sum()}")

            # Evaluate combined model on validation
            val_acc = self._evaluate_combined(model_a, model_b, val_loader_full, device)
            print(f"      Combined val acc: {val_acc:.4f}")
            round_stats.append({"round": rnd+1, "a_to_b": n_a_to_b, "b_to_a": n_b_to_a, "val_acc": val_acc})

        # Final test accuracy
        test_acc = self._evaluate_combined(model_a, model_b, test_loader_full, device)
        self.model_a = model_a
        self.model_b = model_b

        return {"test_acc": test_acc, "rounds": round_stats}

    def _evaluate_combined(self, model_a, model_b, loader_full, device):
        """Evaluate by averaging softmax outputs of both models on their respective views."""
        model_a.eval()
        model_b.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_X, batch_y in loader_full:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)

                view_a, view_b = self.split_views(batch_X)
                probs_a = torch.softmax(model_a(view_a), dim=1)
                probs_b = torch.softmax(model_b(view_b), dim=1)

                # Average confidence
                combined = (probs_a + probs_b) / 2.0
                _, predicted = torch.max(combined, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()

        return correct / total


class SemiSupervisedModel(nn.Module):
    """
    Neural network trained on labeled data + pseudo-labeled data from
    Zhu label propagation.

    Usage:
      1. Run LabelPropagation to get pseudo_labels for unlabeled data
      2. Combine labeled + pseudo-labeled into a single training set
      3. Train with a weighted loss: full weight on labeled, reduced on pseudo

    Same train_epoch / evaluate API as VanillaModel.
    """

    def __init__(self, input_size=784, hidden_size=1028, output_size=10,
                 init_strategy="he", lr=0.01, momentum=0.04):
        super(SemiSupervisedModel, self).__init__()
        if init_strategy not in TORCH_INIT_STRATEGIES:
            raise ValueError(f"Strategy must be one of {list(TORCH_INIT_STRATEGIES.keys())}")

        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

        weight_fn = TORCH_INIT_STRATEGIES[init_strategy]
        for m in self.modules():
            if isinstance(m, nn.Linear):
                weight_fn(m.weight)
                nn.init.constant_(m.bias, 0)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.parameters(), lr=lr, momentum=momentum)

    def forward(self, x):
        return self.network(x)

    def train_epoch(self, train_loader, device):
        self.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = self(batch_X)
            loss = self.criterion(outputs, batch_y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

        return total_loss / len(train_loader), correct / total

    @torch.no_grad()
    def evaluate(self, loader, device):
        self.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_X, batch_y in loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = self(batch_X)
            loss = self.criterion(outputs, batch_y)

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

        return total_loss / len(loader), correct / total
