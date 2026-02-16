import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

TORCH_INIT_STRATEGIES = {
    "he": lambda w: w.data.normal_(0, np.sqrt(2.0 / w.shape[1])),
    "uniform": lambda w: nn.init.uniform_(w, -1 / np.sqrt(w.shape[1]), 1 / np.sqrt(w.shape[1])),
    "normal": lambda w: nn.init.normal_(w, mean=0.0, std=0.01),
}


class RigLModel(nn.Module):
    """
    Implements RigL-style sparse-to-sparse training (Evci et al., 2020).

    The network starts dense, gets pruned to `sparsity_target`, then on every
    `update_interval` training steps the topology is updated:
      1. DROP: remove the `drop_fraction` of active weights with smallest |w|
      2. GROW: activate the same number of currently-dead weights that have the
         largest gradient magnitude (the gradient signal tells us which missing
         connections would help the most)

    Total sparsity stays constant after the initial prune. Only the *which*
    weights are active changes over time.
    """

    def __init__(self, input_size=784, hidden_size=2056, output_size=10,
                 init_strategy="he", lr=0.01, momentum=0.04,
                 sparsity_target=0.50, update_interval=100, drop_fraction=0.20,
                 lambda1=1e-4):
        super(RigLModel, self).__init__()
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

        self.sparsity_target = sparsity_target
        self.update_interval = update_interval
        self.drop_fraction = drop_fraction

        # L1 regularization
        self.lambda1 = lambda1
        self.masks = {}
        self._step_count = 0
        self._topology_updates = 0

    def forward(self, x):
        return self.network(x)

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
    # Initial pruning (sets the starting sparse topology)
    # ------------------------------------------------------------------
    def init_sparse(self):
        """Prune to sparsity_target using magnitude, establishing initial masks."""
        all_mag = []
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                all_mag.append(m.weight.data.abs().flatten())
        all_mag = torch.cat(all_mag)

        k = int(self.sparsity_target * all_mag.numel())
        if k == 0:
            return
        threshold = torch.kthvalue(all_mag, k).values.item()

        self.masks = {}
        total_pruned = 0
        total_weights = 0
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                mask = (m.weight.data.abs() > threshold).float()
                self.masks[name] = mask
                m.weight.data *= mask
                total_pruned += (mask == 0).sum().item()
                total_weights += mask.numel()

        print(f"   RigL init: pruned {total_pruned}/{total_weights} "
              f"({100 * total_pruned / total_weights:.1f}%) to start")

    # ------------------------------------------------------------------
    # Topology update: drop smallest active, grow largest-gradient dead
    # ------------------------------------------------------------------
    def _update_topology(self):
        """
        For each Linear layer:
          1. Among active weights (mask=1), drop the `drop_fraction` with smallest |w|
          2. Among dead weights (mask=0), grow the same count with largest |grad|
          3. Newly grown weights are re-initialized from He distribution
        """
        for name, m in self.named_modules():
            if not isinstance(m, nn.Linear) or name not in self.masks:
                continue

            mask = self.masks[name]
            w = m.weight.data
            g = m.weight.grad

            if g is None:
                continue

            # --- DROP: smallest |w| among active ---
            active = (mask == 1)
            n_active = active.sum().item()
            n_drop = int(self.drop_fraction * n_active)
            if n_drop == 0:
                continue

            active_mag = w.abs() * active.float()
            # Set inactive positions to inf so they aren't selected as "smallest"
            active_mag[~active] = float('inf')
            drop_threshold = torch.kthvalue(active_mag.flatten(), n_drop).values.item()
            drop_mask = (active_mag <= drop_threshold) & active

            # --- GROW: largest |grad| among dead ---
            dead = (mask == 0)
            dead_grad = g.abs() * dead.float()
            n_grow = min(n_drop, dead.sum().item())
            if n_grow == 0:
                continue

            # Set active positions to -1 so they aren't selected
            dead_grad[~dead] = -1.0
            grow_threshold = torch.kthvalue(dead_grad.flatten(),
                                            dead_grad.numel() - n_grow + 1).values.item()
            grow_mask = (dead_grad >= grow_threshold) & dead

            # Apply: drop
            mask[drop_mask] = 0
            w[drop_mask] = 0

            # Apply: grow with He re-init
            fan_in = w.shape[1]
            n_grown = grow_mask.sum().item()
            w[grow_mask] = torch.randn(n_grown, device=w.device) * np.sqrt(2.0 / fan_in)
            mask[grow_mask] = 1

            self.masks[name] = mask

        self._topology_updates += 1

    def _apply_masks(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear) and name in self.masks:
                m.weight.data *= self.masks[name]

    def sparsity(self):
        total = 0
        zeros = 0
        for m in self.modules():
            if isinstance(m, nn.Linear):
                total += m.weight.numel()
                zeros += (m.weight.data == 0).sum().item()
        return zeros / total if total > 0 else 0.0

    # ------------------------------------------------------------------
    # Training / Evaluation (same API as VanillaModel / Stage2Model)
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

            # Topology update before step (need gradients on dead weights)
            self._step_count += 1
            if self.masks and self._step_count % self.update_interval == 0:
                self._update_topology()

            self.optimizer.step()
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
