import pandas as pd
import numpy as np
import os
from scipy import stats


class Results:
    """
    Pipeline for logging per-epoch training metrics to CSV files.

    Directory Structure: base_dir/{config}/{seed}/results_lr{lr}_momentum{momentum}.csv
    """

    CORE_COLUMNS = [
        "config", "seed", "weight_init", "normalization", "augmentations",
        "epoch", "train_loss", "train_acc", "val_loss", "val_acc",
        "test_acc", "converged_epoch", "lr", "momentum"
    ]

    def __init__(self, base_dir="results"):
        self.base_dir = base_dir
        self.filepath = None
        self._run_buffer = []
        self._meta = {}
        # Master df for aggregating across all runs (for summary/ablation)
        self._all_runs = []

    def begin_run(self, seed, weight_init, normalization, augmentations, config,
                  lr=0.01, momentum=0.04):
        """Start a new experiment run. Sets up file path and clears buffer."""
        self._meta = {
            "config": config,
            "seed": seed,
            "weight_init": weight_init,
            "normalization": normalization,
            "augmentations": augmentations,
            "lr": lr,
            "momentum": momentum,
        }
        self._run_buffer = []

        # Create directory: base_dir/config/seed/
        run_dir = os.path.join(self.base_dir, config, str(seed))
        os.makedirs(run_dir, exist_ok=True)

        # Filename: results_lr0.01_momentum0.04.csv
        filename = f"results_lr{lr}_momentum{momentum}.csv"
        self.filepath = os.path.join(run_dir, filename)

    def log_epoch(self, epoch, train_loss, train_acc, val_loss, val_acc, **kwargs):
        """Record metrics for a single epoch."""
        row = {
            **self._meta,
            "epoch": epoch,
            "train_loss": float(train_loss),
            "train_acc": float(train_acc),
            "val_loss": float(val_loss),
            "val_acc": float(val_acc),
            "test_acc": np.nan,
            "converged_epoch": np.nan,
        }
        row.update(kwargs)
        self._run_buffer.append(row)

    def end_run(self, test_acc, converged_epoch):
        """Stamp final metrics on the last epoch row and flush to CSV."""
        if self._run_buffer:
            self._run_buffer[-1]["test_acc"] = float(test_acc)
            self._run_buffer[-1]["converged_epoch"] = int(converged_epoch)

        run_df = pd.DataFrame(self._run_buffer)
        run_df.to_csv(self.filepath, index=False)

        # Store final row for aggregation
        if self._run_buffer:
            self._all_runs.append(self._run_buffer[-1])

        self._run_buffer = []

    def get_config_accs(self, config_name):
        """Return array of test accuracies for a given config."""
        accs = [r["test_acc"] for r in self._all_runs if r["config"] == config_name]
        return np.array(accs)

    def summary(self):
        """Print mean +/- std of test_acc grouped by config."""
        if not self._all_runs:
            print("No completed runs found.")
            return

        df = pd.DataFrame(self._all_runs)
        grouped = df.groupby("config")
        for name, group in grouped:
            accs = group["test_acc"].values
            epochs = group["converged_epoch"].values
            print(f"  {name:>20} | "
                  f"Test Acc: {np.mean(accs):.4f} +/- {np.std(accs):.4f} | "
                  f"Epochs: {np.mean(epochs):.1f} +/- {np.std(epochs):.1f} | "
                  f"n={len(accs)}")

    def ablation_table(self, config_order):
        """
        Build ablation table: config, mean+/-std, improvement vs baseline.
        Reads from CSV files on disk.
        """
        rows = []
        baseline_acc = None
        prev_accs = None

        for cfg in config_order:
            accs = self.load_config_accs(cfg)
            if len(accs) == 0:
                rows.append({"config": cfg, "mean_acc": np.nan, "std_acc": np.nan,
                             "improvement_pct": np.nan, "p_value": np.nan})
                continue

            mean_acc = np.mean(accs)
            std_acc = np.std(accs)

            if baseline_acc is None:
                baseline_acc = mean_acc
                improvement = 0.0
                p_val = np.nan
            else:
                improvement = ((mean_acc - baseline_acc) / baseline_acc) * 100
                if prev_accs is not None and len(accs) == len(prev_accs):
                    _, p_val = stats.ttest_rel(accs, prev_accs)
                else:
                    p_val = np.nan

            prev_accs = accs
            rows.append({
                "config": cfg,
                "mean_acc": mean_acc,
                "std_acc": std_acc,
                "improvement_pct": improvement,
                "p_value": p_val
            })

        return pd.DataFrame(rows)

    def confidence_interval(self, config_name, confidence=0.90):
        """Compute confidence interval using t-distribution. Reads from CSV."""
        accs = self.load_config_accs(config_name)
        n = len(accs)
        if n < 2:
            return np.mean(accs) if n > 0 else np.nan, np.nan, np.nan, np.nan, n

        mean = np.mean(accs)
        se = stats.sem(accs)
        alpha = 1 - confidence
        t_val = stats.t.ppf(1 - alpha / 2, df=n - 1)
        margin = t_val * se

        return mean, mean - margin, mean + margin, t_val, n

    def load_from_disk(self, config_name=None):
        """
        Load results from CSV files on disk into _all_runs.

        Args:
            config_name: If provided, only load that config. Otherwise load all.
        """
        from glob import glob

        if config_name:
            config_dirs = [os.path.join(self.base_dir, config_name)]
        else:
            config_dirs = [os.path.join(self.base_dir, d)
                          for d in os.listdir(self.base_dir)
                          if os.path.isdir(os.path.join(self.base_dir, d))]

        for config_dir in config_dirs:
            if not os.path.exists(config_dir):
                continue

            # Iterate through seed folders
            for seed_dir in glob(os.path.join(config_dir, "*")):
                if not os.path.isdir(seed_dir):
                    continue

                # Read ALL csv files in seed folder (regardless of name)
                for csv_file in glob(os.path.join(seed_dir, "*.csv")):
                    try:
                        df = pd.read_csv(csv_file)
                        if len(df) == 0:
                            continue
                        # Get last row (where test_acc is filled)
                        last_row = df.iloc[-1].to_dict()
                        if pd.notna(last_row.get("test_acc")):
                            self._all_runs.append(last_row)
                    except Exception as e:
                        print(f"Error reading {csv_file}: {e}")

    def load_config_accs(self, config_name):
        """
        Load test accuracies for a config directly from disk.
        Returns array of test_acc values.
        """
        from glob import glob

        config_path = os.path.join(self.base_dir, config_name)
        if not os.path.exists(config_path):
            return np.array([])

        accs = []
        for seed_dir in glob(os.path.join(config_path, "*")):
            if not os.path.isdir(seed_dir):
                continue

            for csv_file in glob(os.path.join(seed_dir, "*.csv")):
                try:
                    df = pd.read_csv(csv_file)
                    if len(df) == 0:
                        continue
                    last_row = df.iloc[-1]
                    if pd.notna(last_row.get("test_acc")):
                        accs.append(float(last_row["test_acc"]))
                except Exception:
                    pass

        return np.array(accs)

    def list_configs(self):
        """List all config folders in base_dir with their results."""
        configs = []
        for folder in sorted(os.listdir(self.base_dir)):
            folder_path = os.path.join(self.base_dir, folder)
            if os.path.isdir(folder_path):
                accs = self.load_config_accs(folder)
                if len(accs) > 0:
                    configs.append({
                        "config": folder,
                        "mean_acc": accs.mean(),
                        "std_acc": accs.std(),
                        "n": len(accs)
                    })
        return configs
