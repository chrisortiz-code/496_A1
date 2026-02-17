import torch
import torch.nn as nn
from VanillaModel import VanillaModel


class Stage2Model(VanillaModel):
    """
    Extends VanillaModel with L1 regularization and magnitude-based pruning.

    Loss Function:
        L(w) = CrossEntropy + lambda1 * |w|

    L1 regularization encourages sparsity, complementing magnitude-based pruning.
    """

    def __init__(self, input_size=784, hidden_size=2056, output_size=10,
                 init_strategy="he", lr=0.01, momentum=0.04, lambda1=1e-4,
                 pruned=False):
        super().__init__(input_size=input_size, hidden_size=hidden_size,
                         output_size=output_size, init_strategy=init_strategy,
                         lr=lr, momentum=momentum)
        self.lambda1 = lambda1
        self.masks = {}
        self.pruned = pruned

    # ------------------------------------------------------------------
    # L1 Regularization
    # ------------------------------------------------------------------
    def _regularization_loss(self):
        """L1 penalty: lambda1 * sum(|w|) - encourages sparsity."""
        l1_penalty = 0.0
        for module in self.modules():
            if isinstance(module, nn.Linear):
                l1_penalty = l1_penalty + module.weight.abs().sum()
        return self.lambda1 * l1_penalty

    # ------------------------------------------------------------------
    # Pruning
    # ------------------------------------------------------------------
    def prune(self, rate):
        """
        Global magnitude-based pruning.
        Removes `rate` fraction of weights (by smallest |w|) across all
        Linear layers. Stores binary masks and zeros out pruned weights.
        """
        all_magnitudes = []
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                all_magnitudes.append(module.weight.data.abs().flatten())

        all_magnitudes = torch.cat(all_magnitudes)

        k = int(rate * all_magnitudes.numel())
        if k == 0:
            return
        threshold = torch.kthvalue(all_magnitudes, k).values.item()

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

        self.pruned = True
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
    # Training (overrides VanillaModel to add L1 + mask enforcement)
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

            if self.pruned:
                self._apply_masks()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

        return total_loss / len(train_loader), correct / total
