import numpy as np

class OverfitDetector:
    """
    Detects overfitting using:  E_V(t) > mean(E_V) + std(E_V)
    
    Tracks all validation errors and checks if the current one
    exceeds the running mean by more than one standard deviation.
    """
    def __init__(self):
        self.val_errors = []
    
    def check(self, val_error):
        """
        Args:
            val_error: validation error (1 - val_accuracy) at current epoch
        Returns:
            is_overfitting (bool), mean_ev, std_ev
        """
        self.val_errors.append(val_error)
        
        # Need at least 2 points to compute a meaningful std
        if len(self.val_errors) < 2:
            return False, val_error, 0.0
        
        # E_V_bar = (1/t) * sum(E_V)
        mean_ev = np.mean(self.val_errors)
        
        # sigma_EV = sqrt( (1/t) * sum( (E_V - E_V_bar)^2 ) )
        std_ev = np.std(self.val_errors)
        
        threshold = mean_ev + std_ev
        is_overfitting = val_error > threshold
        
        return is_overfitting, mean_ev, std_ev


if __name__ == "__main__":
    # --- Quick sanity test ---
    detector = OverfitDetector()

    # Simulate: error decreases then spikes
    fake_errors = [0.30, 0.25, 0.20, 0.18, 0.16, 0.15, 0.14, 0.13, 0.13, 0.30]

    print("Epoch | E_V   | E_V_bar | sigma  | Threshold | Overfit?")
    print("------+-------+---------+--------+-----------+---------")
    for i, ev in enumerate(fake_errors):
        overfit, mean_ev, std_ev = detector.check(ev)
        threshold = mean_ev + std_ev
        print(f"  {i+1:02d}  | {ev:.3f} |  {mean_ev:.3f}  | {std_ev:.3f} |   {threshold:.3f}   |  {overfit}")