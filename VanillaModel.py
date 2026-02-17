import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from Overfitting import OverfitDetector

TORCH_INIT_STRATEGIES = {
    "he": lambda w: w.data.normal_(0, np.sqrt(2.0 / w.shape[1])),
    "uniform": lambda w: nn.init.uniform_(w, -1 / np.sqrt(w.shape[1]), 1 / np.sqrt(w.shape[1])),
    "normal": lambda w: nn.init.normal_(w, mean=0.0, std=0.01),
}


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_loop(model, loaders, device, results=None, epochs=200, print_every=20,
               collect_curves=False):
    """
    Standard training loop with OverfitDetector and optional Results logging.

    Returns:
        (test_acc, converged_epoch) or (test_acc, converged_epoch, train_accs, val_accs) if collect_curves=True
    """
    overfit_detector = OverfitDetector(max_epochs=epochs)
    converged_epoch = epochs
    train_accs, val_accs = [], []

    for epoch in range(epochs):
        train_loss, train_acc = model.train_epoch(loaders["train"], device)
        val_loss, val_acc = model.evaluate(loaders["validation"], device)
        if results:
            results.log_epoch(epoch + 1, train_loss, train_acc, val_loss, val_acc)
        if collect_curves:
            train_accs.append(train_acc)
            val_accs.append(val_acc)

        val_error = 1.0 - val_acc
        is_overfitting, mean_ev, std_ev = overfit_detector.check(val_error, epoch + 1)

        if print_every and (epoch + 1) % print_every == 0:
            status = " ** OVERFIT **" if is_overfitting else ""
            print(f"      Epoch {epoch+1:03d} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}{status}")

        if is_overfitting:
            converged_epoch = epoch + 1
            print(f"      >>> Overfitting at epoch {converged_epoch}")
            break

    _, test_acc = model.evaluate(loaders["test"], device)
    if results:
        results.end_run(test_acc=test_acc, converged_epoch=converged_epoch)
    if collect_curves:
        return test_acc, converged_epoch, train_accs, val_accs
    return test_acc, converged_epoch


class VanillaModel(nn.Module):
    def __init__(self, input_size=784, hidden_size=1028, output_size=10,
                 init_strategy="he", lr=0.01, momentum=0.04):
        super(VanillaModel, self).__init__()
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

    def forward(self, x):
        return self.network(x)

    def train_epoch(self, train_loader, device):
        self.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = self(batch_X)
            loss = self.criterion(outputs, batch_y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

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
