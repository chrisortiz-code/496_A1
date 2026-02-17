import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
from VanillaModel import VanillaModel


class CoTrainingSSL:
    """
    Co-Training Semi-Supervised Learning (Blum & Mitchell, 1998).

    Uses two views of the data (left half and right half of image) to train
    two separate classifiers. Each classifier labels unlabeled data, and
    high-confidence predictions are added to the other classifier's training set.

    Views:
        - View 1 (Left):  pixels from columns 0-13  (14 * 28 = 392 features)
        - View 2 (Right): pixels from columns 14-27 (14 * 28 = 392 features)
    """

    def __init__(self, hidden_size=512, lr=0.01, momentum=0.04,
                 confidence_threshold=0.90, max_add_per_round=100,
                 num_classes=10):
        self.hidden_size = hidden_size
        self.lr = lr
        self.momentum = momentum
        self.confidence_threshold = confidence_threshold
        self.max_add_per_round = max_add_per_round
        self.num_classes = num_classes

        self.model_left = None
        self.model_right = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _split_views(self, X_flat):
        """Split flattened 784-dim images into left and right halves."""
        X_2d = X_flat.reshape(-1, 28, 28)
        X_left = X_2d[:, :, :14].reshape(-1, 14 * 28)
        X_right = X_2d[:, :, 14:].reshape(-1, 14 * 28)
        return X_left.astype(np.float32), X_right.astype(np.float32)

    def _create_model(self, input_size=392):
        """Create a VanillaModel for one view."""
        return VanillaModel(
            input_size=input_size, hidden_size=self.hidden_size,
            output_size=self.num_classes, lr=self.lr, momentum=self.momentum
        ).to(self.device)

    def _train_model(self, model, X, y, epochs=50, batch_size=64):
        """Train a single view model."""
        dataset = _SimpleDataset(X, y)
        loader = TorchDataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            model.train_epoch(loader, self.device)

    @torch.no_grad()
    def _predict_proba(self, model, X):
        """Get softmax probabilities from a model."""
        model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        logits = model(X_tensor)
        probs = torch.softmax(logits, dim=1)
        return probs.cpu().numpy()

    def fit(self, X_labeled, y_labeled, X_unlabeled,
            co_training_rounds=5, epochs_per_round=20, verbose=True):
        """
        Train co-training classifiers.

        Args:
            X_labeled: (N_l, 784) labeled data features
            y_labeled: (N_l,) integer labels
            X_unlabeled: (N_u, 784) unlabeled data features
            co_training_rounds: number of co-training iterations
            epochs_per_round: training epochs per round
            verbose: print progress

        Returns:
            self
        """
        # Split into views
        X_left_labeled, X_right_labeled = self._split_views(X_labeled)
        X_left_unlabeled, X_right_unlabeled = self._split_views(X_unlabeled)

        # Initialize models
        self.model_left = self._create_model(input_size=392)
        self.model_right = self._create_model(input_size=392)

        # Initial training on labeled data
        if verbose:
            print(f"   Initial training on {len(y_labeled)} labeled samples...")
        self._train_model(self.model_left, X_left_labeled, y_labeled, epochs=50)
        self._train_model(self.model_right, X_right_labeled, y_labeled, epochs=50)

        # Separate training sets per view — each model gets its own labels
        X_left_train = X_left_labeled.copy()
        X_right_train = X_right_labeled.copy()
        y_left_train = y_labeled.copy()
        y_right_train = y_labeled.copy()

        # Unlabeled pool
        X_left_pool = X_left_unlabeled.copy()
        X_right_pool = X_right_unlabeled.copy()
        pool_mask = np.ones(len(X_left_pool), dtype=bool)  # True = still unlabeled

        # Co-training rounds
        for round_idx in range(co_training_rounds):
            if pool_mask.sum() == 0:
                if verbose:
                    print(f"   Round {round_idx + 1}: No unlabeled samples left")
                break

            # Get predictions from both models on remaining unlabeled data
            pool_indices = np.where(pool_mask)[0]
            X_left_pool_active = X_left_pool[pool_indices]
            X_right_pool_active = X_right_pool[pool_indices]

            probs_left = self._predict_proba(self.model_left, X_left_pool_active)
            probs_right = self._predict_proba(self.model_right, X_right_pool_active)

            # Find high-confidence predictions from left model
            conf_left = probs_left.max(axis=1)
            pred_left = probs_left.argmax(axis=1)
            confident_left = conf_left >= self.confidence_threshold

            # Find high-confidence predictions from right model
            conf_right = probs_right.max(axis=1)
            pred_right = probs_right.argmax(axis=1)
            confident_right = conf_right >= self.confidence_threshold

            n_added_left = 0
            n_added_right = 0

            # Left model is confident -> teach the RIGHT model only
            if confident_left.sum() > 0:
                add_indices_left = np.where(confident_left)[0][:self.max_add_per_round]
                pool_idx_left = pool_indices[add_indices_left]

                X_right_train = np.vstack([X_right_train, X_right_pool[pool_idx_left]])
                y_right_train = np.concatenate([y_right_train, pred_left[add_indices_left]])

                pool_mask[pool_idx_left] = False
                n_added_left = len(add_indices_left)

            # Right model is confident -> teach the LEFT model only
            remaining_pool = np.where(pool_mask)[0]
            if len(remaining_pool) > 0:
                remaining_local = np.isin(pool_indices, remaining_pool)
                confident_right_remaining = confident_right & remaining_local

                if confident_right_remaining.sum() > 0:
                    add_indices_right = np.where(confident_right_remaining)[0][:self.max_add_per_round]
                    pool_idx_right = pool_indices[add_indices_right]

                    X_left_train = np.vstack([X_left_train, X_left_pool[pool_idx_right]])
                    y_left_train = np.concatenate([y_left_train, pred_right[add_indices_right]])

                    pool_mask[pool_idx_right] = False
                    n_added_right = len(add_indices_right)

            if verbose:
                print(f"   Round {round_idx + 1}: Added {n_added_left} from left, "
                      f"{n_added_right} from right | Pool remaining: {pool_mask.sum()}")

            # Retrain each model on its own expanded dataset
            self._train_model(self.model_left, X_left_train, y_left_train, epochs=epochs_per_round)
            self._train_model(self.model_right, X_right_train, y_right_train, epochs=epochs_per_round)

        if verbose:
            print(f"   Co-training complete. Left set: {len(y_left_train)}, Right set: {len(y_right_train)}")

        return self

    def fit_views(self, X_left_labeled, X_right_labeled, y_labeled,
                  X_left_unlabeled, X_right_unlabeled,
                  co_training_rounds=5, epochs_per_round=20, verbose=True):
        """
        Train co-training classifiers on pre-split views (e.g. PCA-reduced).

        Same algorithm as fit() but skips the internal 784->left/right split.
        Caller is responsible for splitting and any per-view transformations.
        """
        view_dim = X_left_labeled.shape[1]

        self.model_left = self._create_model(input_size=view_dim)
        self.model_right = self._create_model(input_size=view_dim)

        if verbose:
            print(f"   Initial training on {len(y_labeled)} labeled samples (view_dim={view_dim})...")
        self._train_model(self.model_left, X_left_labeled, y_labeled, epochs=50)
        self._train_model(self.model_right, X_right_labeled, y_labeled, epochs=50)

        X_left_train = X_left_labeled.copy()
        X_right_train = X_right_labeled.copy()
        y_left_train = y_labeled.copy()
        y_right_train = y_labeled.copy()

        X_left_pool = X_left_unlabeled.copy()
        X_right_pool = X_right_unlabeled.copy()
        pool_mask = np.ones(len(X_left_pool), dtype=bool)

        for round_idx in range(co_training_rounds):
            if pool_mask.sum() == 0:
                if verbose:
                    print(f"   Round {round_idx + 1}: No unlabeled samples left")
                break

            pool_indices = np.where(pool_mask)[0]
            probs_left = self._predict_proba(self.model_left, X_left_pool[pool_indices])
            probs_right = self._predict_proba(self.model_right, X_right_pool[pool_indices])

            conf_left = probs_left.max(axis=1)
            pred_left = probs_left.argmax(axis=1)
            confident_left = conf_left >= self.confidence_threshold

            conf_right = probs_right.max(axis=1)
            pred_right = probs_right.argmax(axis=1)
            confident_right = conf_right >= self.confidence_threshold

            n_added_left = 0
            n_added_right = 0

            if confident_left.sum() > 0:
                add_idx = np.where(confident_left)[0][:self.max_add_per_round]
                pool_idx = pool_indices[add_idx]
                X_right_train = np.vstack([X_right_train, X_right_pool[pool_idx]])
                y_right_train = np.concatenate([y_right_train, pred_left[add_idx]])
                pool_mask[pool_idx] = False
                n_added_left = len(add_idx)

            remaining_pool = np.where(pool_mask)[0]
            if len(remaining_pool) > 0:
                remaining_local = np.isin(pool_indices, remaining_pool)
                confident_right_remaining = confident_right & remaining_local
                if confident_right_remaining.sum() > 0:
                    add_idx = np.where(confident_right_remaining)[0][:self.max_add_per_round]
                    pool_idx = pool_indices[add_idx]
                    X_left_train = np.vstack([X_left_train, X_left_pool[pool_idx]])
                    y_left_train = np.concatenate([y_left_train, pred_right[add_idx]])
                    pool_mask[pool_idx] = False
                    n_added_right = len(add_idx)

            if verbose:
                print(f"   Round {round_idx + 1}: Added {n_added_left} from left, "
                      f"{n_added_right} from right | Pool remaining: {pool_mask.sum()}")

            self._train_model(self.model_left, X_left_train, y_left_train, epochs=epochs_per_round)
            self._train_model(self.model_right, X_right_train, y_right_train, epochs=epochs_per_round)

        if verbose:
            print(f"   Co-training complete. Left set: {len(y_left_train)}, Right set: {len(y_right_train)}")
        return self

    def score_views(self, X_left, X_right, y):
        """Score on pre-split views."""
        probs_left = self._predict_proba(self.model_left, X_left)
        probs_right = self._predict_proba(self.model_right, X_right)
        preds = ((probs_left + probs_right) / 2).argmax(axis=1)
        return (preds == y).mean()

    def predict(self, X, decision_mode="avg"):
        """Predict labels using both views."""
        X_left, X_right = self._split_views(X)

        probs_left = self._predict_proba(self.model_left, X_left)
        probs_right = self._predict_proba(self.model_right, X_right)

        if decision_mode == "avg":
            combined = (probs_left + probs_right) / 2
            return combined.argmax(axis=1)
        else:  # confident
            conf_left = probs_left.max(axis=1)
            conf_right = probs_right.max(axis=1)

            pred_left = probs_left.argmax(axis=1)
            pred_right = probs_right.argmax(axis=1)

            return np.where(conf_left > conf_right, pred_left, pred_right)

    def predict_proba(self, X, decision_mode="avg"):
        """Get combined probability estimates."""
        X_left, X_right = self._split_views(X)

        probs_left = self._predict_proba(self.model_left, X_left)
        probs_right = self._predict_proba(self.model_right, X_right)

        if decision_mode == "avg":
            return (probs_left + probs_right) / 2
        else:
            conf_left = probs_left.max(axis=1, keepdims=True)
            conf_right = probs_right.max(axis=1, keepdims=True)
            return np.where(conf_left > conf_right, probs_left, probs_right)

    def score(self, X, y):
        """Compute accuracy."""
        preds = self.predict(X)
        return (preds == y).mean()


class _SimpleDataset(Dataset):
    """Simple PyTorch dataset wrapper."""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
