# a1 - Fashion-MNIST Semi-Supervised Learning

Machine learning experiments on Fashion-MNIST covering preprocessing, ablation studies, L1 regularization, pruning, and semi-supervised learning (Label Propagation + Co-Training).

---

## Project Structure

```
a1/
├── results/                ← CSV outputs written here during training
├── a1.ipynb                ← main notebook
├── DataLoader.py           ← data loading, preprocessing, augmentation
├── VanillaModel.py         ← base MLP, train_loop, set_seed
├── Stage2Model.py          ← MLP with L1 regularization + pruning
├── SemiSupervisedModel.py  ← Label Propagation
├── CoTrainingSSL.py        ← Co-Training SSL
├── Results.py              ← CSV logging and summary helpers
├── requirements.txt        ← pip dependencies (excludes torch)
└── README.md
```

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Install PyTorch with GPU (CUDA) support

First check your CUDA version:
```bash
nvidia-smi
```

Then install the matching PyTorch build:

| CUDA Version | Command |
|---|---|
| 12.4 | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124` |
| 12.1 | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121` |
| 11.8 | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118` |

etc.
### 3. Verify GPU is working
```python
import torch
print(torch.cuda.is_available())       # Should print: True
print(torch.cuda.get_device_name(0))   # Should print your GPU name
```

If `False` is returned, your CUDA version and PyTorch build don't match — recheck `nvidia-smi` and reinstall.


### 4. Launch the notebook
```bash
jupyter notebook a1.ipynb
```

---

## Notebook Stages

| Stage | What it does |
|---|---|
| Stage 1 | Data loading, Z-score vs Min-Max preprocessing, augmentation demo |
| Stage 2 | Ablation study — baseline → preprocessing → He init |
| Stage 2 cont. | Weight init comparison (uniform / normal / He) |
| Stage 2 cont. | Normalization comparison (Z-score vs Min-Max) |
| Stage 2 cont. | Hyperparameter grid search (LR x Momentum) |
| Stage 3a | L1 regularization lambda tuning + weight sparsity |
| Stage 3a cont. | L1 regularization full run across 5 seeds |
| Stage 3a cont. | Magnitude-based pruning (10% / 25% / 50%) |
| Stage 3b | Fully supervised upper bound (100% labels) |
| Stage 4 | kNN tuning for Label Propagation |
| Stage 4 cont. | Confidence threshold sweep (GSSL + Co-Training) |
| Stage 4 cont. | Extra seeds for 95% threshold |
| Stage 5 | SSL vs upper bound comparison + bar chart |
| Stage 6 | PCA dimensionality reduction experiment |
| Final | Ablation table, confidence intervals, paired t-tests |

---


Run top-to-bottom and confirm no import errors and the results/ folder is being populated.

---


Report_Chris_Ortiz.pdf holds the report

## Notes

- All results are saved as CSVs inside results/ automatically by the Results class
- The final analysis cell requires all prior result CSVs to exist — run stages in order
- Training is slow on CPU; GPU is strongly recommended for full runs
- A few cells I recommend to skip are:
- - Hyperparameter tuning: it took me about an hour just to figure out the best lr and momentum (take my word for it, .001 and 0.9 are the best)
- - Finding the upper bound: The full dataset takes very long. In test.ipynb you can see 89.35 was acheived, then in the main notebook, we acheived 89.45%. Unless you are changing seeds, this will be the result.
