import numpy as np
import pandas as pd
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

        print(f"Data Loaded. Shape: {self.df.shape}")

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

    def augment_horizontal_flip(self, df):
        """Applies horizontal flip to the features of the provided dataframe."""
        print("Applying Horizontal Flip...")
        df_aug = df.copy()

        features = df_aug[self.feature_cols].values
        #10% augmented ratio
        aug_features = [self._horizontal_flip(row) if np.random.rand()<0.1 else row for row in features ]

        df_aug[self.feature_cols] = pd.DataFrame(aug_features, index=df.index)
        return df_aug

    # ---------------------------------------------------------
    # Splitting Methods
    # ---------------------------------------------------------
    def get_supervised_split(self, df, test_size=0.10, val_size=0.10, seed=42):
        """
        Splits df into Train, Validation, and Test.
        Returns a dictionary.
        """
        np.random.seed(seed)
        n = len(df)
        indices = np.random.permutation(n)

        n_test = int(test_size * n)
        n_val = int(val_size * n)

        test_idx = indices[:n_test]
        val_idx = indices[n_test: n_test + n_val]
        train_idx = indices[n_test + n_val:]

        return {
            "train": df.iloc[train_idx].reset_index(drop=True),
            "validation": df.iloc[val_idx].reset_index(drop=True),
            "test": df.iloc[test_idx].reset_index(drop=True)
        }

    def get_semi_supervised_split(self, df, test_size=0.10, val_size=0.10, labeled_ratio=0.20, seed=42):
        """
        Splits df into Labeled Train, Unlabeled Train, Validation, and Test.
        Unlabeled train has 'label' column set to NaN.
        Returns a dictionary.
        """
        # 1. Get standard supervised split first
        splits = self.get_supervised_split(df, test_size, val_size, seed)
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