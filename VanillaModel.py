import torch
import torch.nn as nn
import numpy as np

TORCH_INIT_STRATEGIES = {
    "he": lambda w: nn.init.kaiming_normal_(w, mode='fan_in', nonlinearity='relu'),
    "uniform": lambda w: nn.init.uniform_(w, -1 / np.sqrt(w.shape[1]), 1 / np.sqrt(w.shape[1])),
    "normal": lambda w: nn.init.normal_(w, mean=0.0, std=0.01),
}


class VanillaModel(nn.Module):
    def __init__(self, input_size=784, hidden_size=1028, output_size=10, init_strategy="he"):
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

    def forward(self, x):
        return self.network(x)
