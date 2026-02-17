import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
from tensorflow.keras.datasets import fashion_mnist


class DataLoader:
    def __init__(self):
        # 1. Load Data
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

        # 2. Combine and Flatten
        x_full = np.concatenate([x_train, x_test], axis=0)  # (70000, 28, 28)
        y_full = np.concatenate([y_train, y_test], axis=0)  # (70000,)

        # Flatten: (70000, 784)
        self.IMG_SIZE = 28
        X_flat = x_full.reshape(x_full.shape[0], -1)

        # 3. Create Internal Pandas Structures
        X_df = pd.DataFrame(X_flat)
        y_df = pd.Series(y_full, name="label")

        # Master DataFrame
        self.df = pd.concat([X_df, y_df], axis=1)
        # Keep track of feature columns (all columns except label)
        self.feature_cols = X_df.columns

    def get_standard_df(self):
        """Returns the raw dataframe."""
        return self.df.copy()

    def get_z_score_df(self):
        """Returns Z-score normalized DataFrame."""
        df_features = self.df[self.feature_cols]
        y = self.df["label"]

        # Calculate statistics
        mu = df_features.mean()
        sigma = df_features.std()

        # Safety: avoid division by zero
        sigma[sigma == 0] = 1

        # Normalize
        X_zscore = (df_features - mu) / sigma

        return pd.concat([X_zscore, y], axis=1)

    def get_min_max_df(self):
        """Returns Min-Max normalized DataFrame."""
        df_features = self.df[self.feature_cols]
        y = self.df["label"]

        x_min = df_features.min()
        x_max = df_features.max()

        # Safety: avoid division by zero
        denom = x_max - x_min
        denom[denom == 0] = 1

        # Normalize
        X_minmax = (df_features - x_min) / denom

        return pd.concat([X_minmax, y], axis=1)

    # ---------------------------------------------------------
    # Image Processing Helpers (Static / Internal)
    # ---------------------------------------------------------
    @staticmethod
    def _color_jitter_3x3(flat_image, scale=1.0):
        # Reshape to 2D
        img = flat_image.reshape(28, 28)
        jittered = img.copy().astype(float)

        rows, cols = img.shape

        # Note: This loop is slow in Python; usually we use vectorized operations
        # or libraries like cv2/albumentations. Keeping user logic here:
        for i in range(rows):
            for j in range(cols):
                # neighborhood bounds
                i0, i1 = max(0, i - 1), min(rows, i + 2)
                j0, j1 = max(0, j - 1), min(cols, j + 2)

                neighborhood = img[i0:i1, j0:j1]
                sigma = np.std(neighborhood)

                noise = np.random.normal(0, scale * sigma)
                jittered[i, j] += noise

        return jittered.flatten()

    @staticmethod
    def _horizontal_flip(flat_image):
        # Reshape to 2D
        img = flat_image.reshape(28, 28)
        # Flip Left-Right (Horizontal)
        # logic is: flipped[:, c] = image[:, IMG_SIZE - 1 - c]
        flipped = np.fliplr(img)
        return flipped.flatten()

    # ---------------------------------------------------------
    # Augmentation Methods
    # ---------------------------------------------------------
    def augment_33_jitter(self, df, scale=1.0):
        """Applies jitter to the features of the provided dataframe."""
        print("Applying 3x3 Jitter (this may take time)...")
        df_aug = df.copy()

        # Apply transformation only to feature columns
        # using lambda to apply row-wise
        features = df_aug[self.feature_cols].values

        # We use a list comprehension for slight speedup over pd.apply
        aug_features = [self._color_jitter_3x3(img, scale) for img in features]

        df_aug[self.feature_cols] = pd.DataFrame(aug_features, index=df.index)
        return df_aug

    def augment_horizontal_flip(self, df, prob=1.0):
        """Apply horizontal flip to prob fraction of images."""
        df_aug = df.copy()
        features = df_aug[self.feature_cols].to_numpy().copy()
        mask = np.random.rand(len(features)) < prob
        if mask.any():
            features[mask] = np.array([self._horizontal_flip(row) for row in features[mask]])
        df_aug[self.feature_cols] = features
        return df_aug

    def augment_combined(self, df, flip_prob=0.5, jitter_prob=0.5, jitter_scale=1.0):
        """
        Apply augmentation: each image has flip_prob chance of flip,
        jitter_prob chance of 3x3 jitter. Both can apply to same image.
        """
        df_aug = df.copy()
        features = df_aug[self.feature_cols].to_numpy().copy()
        n = len(features)

        # Flip mask
        flip_mask = np.random.rand(n) < flip_prob
        if flip_mask.any():
            features[flip_mask] = np.array([self._horizontal_flip(row) for row in features[flip_mask]])

        # Jitter mask
        jitter_mask = np.random.rand(n) < jitter_prob
        if jitter_mask.any():
            features[jitter_mask] = np.array([self._color_jitter_3x3(row, jitter_scale) for row in features[jitter_mask]])

        df_aug[self.feature_cols] = features
        return df_aug
    # ---------------------------------------------------------
    # Splitting Methods
    # ---------------------------------------------------------
    def get_supervised_split(
            self,
            df,
            dataset_pct=0.10,
            test_size=0.10,
            val_size=0.10,
            seed=42
    ):
        """
        Splits df into Train, Validation, and Test
        using only dataset_pct of the data.
        Returns a dictionary.
        """

        np.random.seed(seed)
        n = len(df)

        # --- 1. Subsample dataset ---
        n_used = int(dataset_pct * n)
        indices = np.random.permutation(n)[:n_used]

        # --- 2. Compute split sizes ---
        n_test = int(test_size * n_used)
        n_val = int(val_size * n_used)

        test_idx = indices[:n_test]
        val_idx = indices[n_test: n_test + n_val]
        train_idx = indices[n_test + n_val:]

        return {
            "train": df.iloc[train_idx].reset_index(drop=True),
            "validation": df.iloc[val_idx].reset_index(drop=True),
            "test": df.iloc[test_idx].reset_index(drop=True)
        }

    def get_semi_supervised_split(self, df, test_size=0.10, val_size=0.10, labeled_ratio=0.10, seed=42):
        """
        Splits df into Labeled Train, Unlabeled Train, Validation, and Test.
        Unlabeled train has 'label' column set to NaN.
        Returns a dictionary.
        """
        # 1. Get standard supervised split first (use ALL data, then split)
        splits = self.get_supervised_split(df, dataset_pct=1.0, test_size=test_size, val_size=val_size, seed=seed)
        train_pool = splits['train']

        # 2. Split Train Pool into Labeled and Unlabeled
        n_train = len(train_pool)
        indices = np.random.permutation(n_train)

        n_labeled = int(labeled_ratio * n_train)

        labeled_idx = indices[:n_labeled]
        unlabeled_idx = indices[n_labeled:]

        labeled_data = train_pool.iloc[labeled_idx].reset_index(drop=True)
        unlabeled_data = train_pool.iloc[unlabeled_idx].reset_index(drop=True)

        # 3. Mask labels in unlabeled data
        unlabeled_data = unlabeled_data.copy()
        unlabeled_data["label"] = np.nan

        return {
            "labeled_train": labeled_data,
            "unlabeled_train": unlabeled_data,
            "validation": splits['validation'],
            "test": splits['test']
        }

    def prepare_data(self, test_size=0.10, val_size=0.10, labeled_ratio=0.10,
                     seed=42, preprocess=True, normalize="z_score"):
        """
        Proper ML pipeline: split FIRST, then preprocess (normalize + augment train only).

        Args:
            preprocess: If True, applies normalization (train stats) + augmentation (train only).
                        If False, returns raw splits with no normalization or augmentation.
            normalize: "z_score" or "min_max" (only used if preprocess=True)

        Returns:
            dict with labeled_train, unlabeled_train, validation, test
        """
        np.random.seed(seed)

        # 1. Split RAW data first (no preprocessing yet)
        raw_splits = self.get_semi_supervised_split(
            self.df, test_size=test_size, val_size=val_size,
            labeled_ratio=labeled_ratio, seed=seed
        )

        if not preprocess:
            # Return raw splits, no normalization or augmentation
            return raw_splits

        # 2-3. Normalize all splits using train stats
        normalized = self.normalize_splits(raw_splits, normalize=normalize)

        # 4. Augment TRAINING data only (50% flip + 50% jitter)
        labeled_aug = self.augment_combined(
            normalized["labeled_train"], flip_prob=0.5, jitter_prob=0.5
        )
        normalized["labeled_train"] = pd.concat(
            [normalized["labeled_train"], labeled_aug], axis=0
        ).reset_index(drop=True)

        return normalized

    def normalize_splits(self, splits, normalize="z_score"):
        """
        Normalize all splits using stats computed from labeled_train only.
        Does NOT augment. Use this when you need normalization without augmentation
        (e.g. for PCA experiments).

        Args:
            splits: dict from get_semi_supervised_split()
            normalize: "z_score" or "min_max"

        Returns:
            dict with same keys, features normalized
        """
        train_features = splits["labeled_train"][self.feature_cols]

        if normalize == "z_score":
            mu = train_features.mean()
            sigma = train_features.std()
            sigma[sigma == 0] = 1
            stats = {"type": "z_score", "mu": mu, "sigma": sigma}
        else:  # min_max
            x_min = train_features.min()
            x_max = train_features.max()
            denom = x_max - x_min
            denom[denom == 0] = 1
            stats = {"type": "min_max", "min": x_min, "denom": denom}

        def apply_norm(df):
            df_out = df.copy()
            if stats["type"] == "z_score":
                df_out[self.feature_cols] = (df[self.feature_cols] - stats["mu"]) / stats["sigma"]
            else:
                df_out[self.feature_cols] = (df[self.feature_cols] - stats["min"]) / stats["denom"]
            return df_out

        return {k: apply_norm(v) for k, v in splits.items()}


    def to_numpy(self, df, num_classes=10, one_hot=True, drop_na_labels=True):
        """
        Converts a DataFrame to (X, y) numpy arrays.
        If one_hot=False, returns integer labels.
        """
        X = df[self.feature_cols].values.astype(np.float32)

        if "label" not in df.columns:
            return X, None

        y_raw = df["label"].values

        if drop_na_labels:
            mask = ~pd.isna(y_raw)
            X = X[mask]
            y_raw = y_raw[mask]

        y_raw = y_raw.astype(int)

        if one_hot:
            y = np.eye(num_classes)[y_raw]
        else:
            y = y_raw

        return X, y

    def to_torch_loaders(self, data_dict, batch_size=64, label_keys=None):
        """
        Converts a split dict (from get_supervised_split or get_semi_supervised_split)
        into a dict of TorchDataLoaders with the same keys.

        label_keys: which keys contain labeled data. Defaults to all keys
                    whose 'label' column has no NaNs.
        """
        import platform
        num_workers = 0 if platform.system() == 'Windows' else 2
        pin = torch.cuda.is_available()

        loaders = {}
        for key, df in data_dict.items():
            X, y = self.to_numpy(df)
            if y is None:
                continue
            dataset = _DatasetToTorch(X, y)
            shuffle = ("train" in key)
            loaders[key] = TorchDataLoader(
                dataset, batch_size=batch_size, shuffle=shuffle,
                num_workers=num_workers, pin_memory=pin
            )
        return loaders


class _DatasetToTorch(Dataset):
    """Wraps numpy X, y (one-hot) into a PyTorch Dataset with integer labels."""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(np.argmax(y, axis=1))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ==========================================
# Usage Example
# ==========================================
if __name__ == "__main__":
    loader = DataLoader()

    # 1. Get different normalized versions
    df_minmax = loader.get_min_max_df()
    df_zscore = loader.get_z_score_df()

    # 2. Supervised Split Example
    print("\n--- Supervised Split (MinMax) ---")
    sup_data = loader.get_supervised_split(df_minmax, test_size=0.1, val_size=0.1)
    print(f"Train shape: {sup_data['train'].shape}")
    print(f"Test shape:  {sup_data['test'].shape}")

    # 3. Semi-Supervised Split Example
    print("\n--- Semi-Supervised Split (Z-Score) ---")
    semi_data = loader.get_semi_supervised_split(df_zscore, labeled_ratio=0.2)
    print(f"Labeled Train shape:   {semi_data['labeled_train'].shape}")
    print(f"Unlabeled Train shape: {semi_data['unlabeled_train'].shape}")

    # Check NaN labels
    na_count = semi_data['unlabeled_train']['label'].isna().sum()
    print(f"Labels hidden (NaN) in unlabeled set: {na_count}")

    # 4. Augmentation Example (on a small subset for speed)
    print("\n--- Augmentation Example ---")
    small_sample = sup_data['test'].head(5)  # Take 5 images
    aug_df = loader.augment_horizontal_flip(small_sample)
    print("Augmentation complete.")