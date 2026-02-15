import numpy as np
from math import sqrt

import numpy as np
from math import sqrt

class OverfitDetector:
    """
    Moving-average-based overfitting detector.

    Detects overfitting when:

        E_V(t) > mean_window + scale * std_window * sqrt(first_check / epoch)

    - Uses rolling window (local behavior)
    - Strict early, gradually relaxes
    """

    def __init__(self, max_epochs, window=5, scale=2.0):
        self.scale = scale
        self.window = window
        self.first_check = 10
        self.val_errors = []
        self.max_epochs = max_epochs

    def check(self, val_error, epoch):
        self.val_errors.append(val_error)

        # Not enough data yet
        if len(self.val_errors) < max(self.first_check, self.window):
            return False, val_error, 0.0

        # Use last `window` errors
        recent = self.val_errors[-self.window:]

        mean_ev = np.mean(recent)
        std_ev = np.std(recent)

        # Strict early, relaxed later
        decay = sqrt(self.first_check / (epoch + 1))

        threshold = mean_ev + self.scale * std_ev * decay

        is_overfitting = val_error > threshold

        return is_overfitting, mean_ev, std_ev


if __name__ == "__main__":
    

    # Simulate: error decreases then spikes
    fake_errors = [0.30, 0.25, 0.20, 0.18, 0.16, 0.15, 0.14, 0.13, 0.13, 0.30]

    # --- Quick sanity test ---
    detector = OverfitDetector(len(fake_errors))

    print("Epoch | E_V   | E_V_bar | sigma  | Threshold | Overfit?")
    print("------+-------+---------+--------+-----------+---------")
    for i, ev in enumerate(fake_errors):
        overfit, mean_ev, std_ev = detector.check(ev,i)
        threshold = mean_ev + std_ev
        print(f"  {i+1:02d}  | {ev:.3f} |  {mean_ev:.3f}  | {std_ev:.3f} |   {threshold:.3f}   |  {overfit}")