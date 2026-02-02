import pandas as pd
import numpy as np
import os


class Results:
    """
    Pipeline for logging per-epoch training metrics to a CSV file.

    Each row represents one epoch of one experiment run, tagged with
    metadata columns (seed, weight_init, normalization, augmentations).

    Extra keyword arguments passed to log_epoch() are stored as
    additional columns (e.g. weight norms, pruning ratios).
    """

    CORE_COLUMNS = [
        "seed", "weight_init", "normalization", "augmentations",
        "epoch", "train_loss", "train_acc", "val_loss", "val_acc",
        "test_acc", "converged_epoch"
    ]

    def __init__(self, filepath="results.csv"):
        self.filepath = filepath
        self._run_buffer = []
        self._meta = {}

        if os.path.exists(filepath):
            self.df = pd.read_csv(filepath)
        else:
            self.df = pd.DataFrame(columns=self.CORE_COLUMNS)

    def begin_run(self, seed, weight_init, normalization, augmentations):
        """Start a new experiment run. Clears the epoch buffer."""
        self._meta = {
            "seed": seed,
            "weight_init": weight_init,
            "normalization": normalization,
            "augmentations": augmentations,
        }
        self._run_buffer = []

    def log_epoch(self, epoch, train_loss, train_acc, val_loss, val_acc, **kwargs):
        """Record metrics for a single epoch."""
        row = {
            **self._meta,
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "test_acc": np.nan,
            "converged_epoch": np.nan,
        }
        row.update(kwargs)
        self._run_buffer.append(row)

    def end_run(self, test_acc, converged_epoch):
        """Stamp final metrics on the last epoch row and flush to CSV."""
        if self._run_buffer:
            self._run_buffer[-1]["test_acc"] = test_acc
            self._run_buffer[-1]["converged_epoch"] = converged_epoch

        run_df = pd.DataFrame(self._run_buffer)
        self.df = pd.concat([self.df, run_df], ignore_index=True)
        self._flush()
        self._run_buffer = []

    def get_run(self, seed, weight_init):
        """Return rows for a specific (seed, weight_init) combination."""
        mask = (self.df["seed"] == seed) & (self.df["weight_init"] == weight_init)
        return self.df[mask].reset_index(drop=True)

    def summary(self):
        """Print mean +/- std of test_acc grouped by (weight_init, normalization)."""
        final = self.df.dropna(subset=["test_acc"])
        if final.empty:
            print("No completed runs found.")
            return

        grouped = final.groupby(["weight_init", "normalization"])
        for name, group in grouped:
            accs = group["test_acc"].values
            epochs = group["converged_epoch"].values
            print(f"  {name[0]:>10} | {name[1]:>10} | "
                  f"Test Acc: {np.mean(accs):.4f} +/- {np.std(accs):.4f} | "
                  f"Epochs: {np.mean(epochs):.1f} +/- {np.std(epochs):.1f} | "
                  f"n={len(accs)}")

    def _flush(self):
        """Write the full dataframe to CSV."""
        self.df.to_csv(self.filepath, index=False)
