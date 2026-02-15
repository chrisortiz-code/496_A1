import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

TORCH_INIT_STRATEGIES = {
    "he": lambda w: w.data.normal_(0, np.sqrt(2.0 / w.shape[1])),
    "uniform": lambda w: nn.init.uniform_(w, -1 / np.sqrt(w.shape[1]), 1 / np.sqrt(w.shape[1])),
    "normal": lambda w: nn.init.normal_(w, mean=0.0, std=0.01),
}


class Stage2Model(nn.Module):
    """
    Starts as a larger-than-necessary network (2x hidden units by default).
    Supports magnitude-based weight pruning: zeros out the smallest |w| values
    globally across all Linear layers, then enforces the mask during fine-tuning
    so pruned weights stay at zero.

    Custom Loss Function (Level 2 vs Vanilla):
        L(w) = CrossEntropy + lambda1 * |w| + lambda2 * |1 / (w + epsilon)|

    The two regularization terms create a "sweet spot" for weight magnitudes:
      - lambda1 * |w|           penalizes weights that grow too large (L1-style)
      - lambda2 * |1/(w + eps)| penalizes weights that shrink toward zero (inverse penalty)

    Together they encourage weights to stay at moderate magnitudes, which
    complements magnitude-based pruning by discouraging near-zero weights
    that would otherwise be pruned.

    Gradient (derived via chain rule, used automatically by autograd):
        dL/dw = (y_hat - y)x + lambda1 * sgn(w) - lambda2 * sgn(w+eps) / (w+eps)^2
    """

    def __init__(self, input_size=784, hidden_size=2056, output_size=10,
                 init_strategy="he", lr=0.01, momentum=0.04,
                 lambda1=1e-5, lambda2=1e-5, epsilon=1e-8):
        super(Stage2Model, self).__init__()
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

        # Dual-regularization hyperparameters
        self.lambda1 = lambda1   # large-weight penalty coefficient
        self.lambda2 = lambda2   # small-weight penalty coefficient
        self.epsilon = epsilon   # numerical stability for inverse term

        # Pruning masks: 1 = keep, 0 = pruned. None until prune() is called.
        self.masks = {}

    def forward(self, x):
        return self.network(x)

    # ------------------------------------------------------------------
    # Dual Regularization: large-w penalty + small-w penalty
    # ------------------------------------------------------------------
    def _regularization_loss(self):
        """
        Computes the two regularization terms over all Linear layer weights:
            lambda1 * sum(|w|)  +  lambda2 * sum(|1 / (w + epsilon)|)

        Both terms are differentiable and autograd computes the gradients:
            d/dw [lambda1 * |w|]             = lambda1 * sgn(w)
            d/dw [lambda2 * |1/(w+eps)|]     = -lambda2 * sgn(w+eps) / (w+eps)^2
        """
        large_penalty = 0.0
        small_penalty = 0.0

        for module in self.modules():
            if isinstance(module, nn.Linear):
                w = module.weight
                large_penalty = large_penalty + w.abs().sum()
                small_penalty = small_penalty + (1.0 / (w + self.epsilon)).abs().sum()

        return self.lambda1 * large_penalty + self.lambda2 * small_penalty

    # ------------------------------------------------------------------
    # Pruning
    # ------------------------------------------------------------------
    def prune(self, rate):
        """
        Global magnitude-based pruning.
        Removes `rate` fraction of weights (by smallest |w|) across all
        Linear layers. Stores binary masks and zeros out pruned weights.

        Args:
            rate: float in (0, 1), e.g. 0.25 means prune 25% of all weights
        """
        # 1. Collect all weight magnitudes into a single flat tensor
        all_magnitudes = []
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                all_magnitudes.append(module.weight.data.abs().flatten())

        all_magnitudes = torch.cat(all_magnitudes)

        # 2. Find the threshold: the value below which `rate` fraction of weights fall
        k = int(rate * all_magnitudes.numel())
        if k == 0:
            return
        threshold = torch.kthvalue(all_magnitudes, k).values.item()

        # 3. Build masks and apply
        total_pruned = 0
        total_weights = 0
        self.masks = {}

        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                mask = (module.weight.data.abs() > threshold).float()
                self.masks[name] = mask
                module.weight.data *= mask
                total_pruned += (mask == 0).sum().item()
                total_weights += mask.numel()

        print(f"   Pruned {total_pruned}/{total_weights} weights "
              f"({100 * total_pruned / total_weights:.1f}%) at threshold={threshold:.6f}")

    def _apply_masks(self):
        """Zero out pruned weights after each optimizer step."""
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear) and name in self.masks:
                module.weight.data *= self.masks[name]

    def sparsity(self):
        """Returns the fraction of weights that are exactly zero."""
        total = 0
        zeros = 0
        for module in self.modules():
            if isinstance(module, nn.Linear):
                total += module.weight.numel()
                zeros += (module.weight.data == 0).sum().item()
        return zeros / total if total > 0 else 0.0

    # ------------------------------------------------------------------
    # Training / Evaluation (same API as VanillaModel)
    # ------------------------------------------------------------------
    def train_epoch(self, train_loader, device):
        self.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = self(batch_X)
            loss = self.criterion(outputs, batch_y) + self._regularization_loss()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Re-zero pruned weights so they never come back
            self._apply_masks()

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
