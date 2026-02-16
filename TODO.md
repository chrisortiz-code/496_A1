# COSC 4P96 Assignment 1 - TODO List

## Project Overview
Neural Network implementation with pruning and regularization techniques on Fashion-MNIST dataset.

---

## Stage 1: Vanilla Neural Network ✓ (Mostly Complete)

### Implementation
- [x] Create `VanillaModel.py` with 3-layer feedforward architecture
- [x] Implement weight initialization strategies (He, Uniform, Normal)
- [x] Implement training and evaluation methods
- [ ] Verify all initialization strategies work correctly
- [ ] Test with different hyperparameters

### Data Processing
- [x] Create `DataLoader.py` for Fashion-MNIST
- [x] Implement Z-score normalization
- [x] Implement Min-max normalization
- [x] Create 3x3 color jitter augmentation
- [x] Create horizontal flip augmentation (10% of data)
- [ ] Optimize augmentation performance (currently slow)

### Overfitting Detection
- [x] Create `OverfitDetector` class with statistical criterion
- [x] Implement criterion: E_V(t) > mean(E_V) + std(E_V)
- [x] Add adaptive threshold scaling
- [ ] Test overfitting detection in training loop
- [ ] Tune scale parameter for best early stopping

### Training & Results
- [x] Create `Results.py` for logging metrics
- [x] Update Results.py with hierarchical folder structure (model_name/seed/results.csv)
- [ ] Update training scripts to use new Results structure
- [ ] Run experiments with all initialization strategies
- [ ] Run experiments with different normalization methods
- [ ] Run experiments with and without augmentation
- [ ] Collect results across multiple seeds (3-5 seeds minimum)

---

## Stage 2: Pruning and Dual Regularization ✓ (Implementation Done)

### Implementation
- [x] Create `Stage2Model.py` with 2x hidden units (2056)
- [x] Implement global magnitude-based pruning
- [x] Implement mask enforcement during training
- [x] Add dual regularization (L1 + inverse penalty)
- [x] Derive gradients for custom loss function

### Testing & Experiments
- [ ] Test pruning at different rates (25%, 50%, 75%)
- [ ] Measure sparsity after pruning
- [ ] Test dual regularization with different λ1, λ2 values
- [ ] Compare with Stage 1 baseline
- [ ] Run experiments across multiple seeds
- [ ] Analyze accuracy vs. sparsity trade-off

---

## Stage 3: Advanced Techniques (Choose One)

### Option A: RigL (Rigging the Lottery)
- [x] Create `RigLModel.py` skeleton
- [ ] Implement dynamic sparse training
- [ ] Implement grow-prune cycles
- [ ] Add gradient-based weight selection
- [ ] Test with different grow/prune schedules
- [ ] Compare with static pruning (Stage 2)

### Option B: Semi-Supervised Graph-Based Learning
- [x] Create `SemiSupervisedModel.py` skeleton
- [ ] Implement graph construction from features
- [ ] Implement label propagation algorithm
- [ ] Integrate with neural network training
- [ ] Test with different labeled/unlabeled ratios
- [ ] Compare with fully supervised baseline

### Decision
- [ ] **DECIDE which Stage 3 approach to use**
- [ ] Complete chosen implementation
- [ ] Run full experiments
- [ ] Analyze results

---

## Report Writing

### LaTeX Report
- [x] Create `report.tex` with IEEE conference format
- [ ] Fill in Introduction section
  - [ ] Background and motivation
  - [ ] Problem statement
  - [ ] Objectives
- [ ] Complete Stage 1 section
  - [ ] Architecture description
  - [ ] Weight initialization strategies (equations)
  - [ ] Results tables/figures
- [ ] Complete Standardization section
  - [ ] Z-score formulation
  - [ ] Min-max formulation
  - [ ] Comparison results
- [ ] Complete Augmentation section
  - [ ] 3x3 jitter algorithm description
  - [ ] Horizontal flip description
  - [ ] Impact on performance
- [ ] Complete Overfitting Detection section
  - [ ] Mathematical criterion
  - [ ] Implementation details
  - [ ] Example detection plots
- [ ] Complete Gradient Derivations section
  - [ ] Cross-entropy + softmax gradient
  - [ ] L1 regularization gradient
  - [ ] Inverse penalty gradient
  - [ ] Full combined gradient
- [ ] Complete Stage 2 section
  - [ ] Pruning algorithm description
  - [ ] Dual regularization formulation
  - [ ] Loss function breakdown
  - [ ] Results and analysis
- [ ] Complete Stage 3 section
  - [ ] Chosen method description
  - [ ] Algorithm details
  - [ ] Results and comparison
- [ ] Complete Results section
  - [ ] Experimental setup
  - [ ] Tables for all experiments
  - [ ] Figures/plots for key findings
  - [ ] Comparison across stages
- [ ] Complete Statistical Analysis section
  - [ ] Mean ± std across seeds
  - [ ] Confidence intervals
  - [ ] Significance testing
  - [ ] Ablation studies
- [ ] Complete Conclusion
  - [ ] Summary of findings
  - [ ] Key contributions
  - [ ] Limitations
  - [ ] Future work
- [ ] Add references
- [ ] Add figures and tables
- [ ] Proofread and format

---

## Code Quality & Documentation

### Code Review
- [ ] Add docstrings to all classes and methods
- [ ] Add type hints where appropriate
- [ ] Remove debug print statements
- [ ] Add comments for complex logic
- [ ] Ensure consistent naming conventions

### Bug Fixes
- [ ] Fix missing `sqrt` import in `Overfitting.py:36`
- [ ] Verify GPU utilization is working correctly
- [ ] Test all models on CPU and GPU

### Git Management
- [ ] Review `.gitignore` (currently `.giitignore` - typo?)
- [ ] Commit organized changes
- [ ] Clean up `__pycache__` directories
- [ ] Remove or properly track `.idea/` and `.claude/`

---

## Experiments & Analysis

### Hyperparameter Tuning
- [ ] Learning rate search
- [ ] Momentum tuning
- [ ] Batch size experiments
- [ ] Hidden layer size experiments
- [ ] Regularization parameter search (λ1, λ2)

### Seeds & Statistical Validity
- [ ] Define seed list (e.g., [42, 123, 456, 789, 1024])
- [ ] Run all experiments with all seeds
- [ ] Compute mean and std for all metrics
- [ ] Generate confidence intervals

### Visualization
- [ ] Training/validation loss curves
- [ ] Accuracy progression plots
- [ ] Overfitting detection visualization
- [ ] Weight distribution histograms
- [ ] Sparsity visualization
- [ ] Comparison bar charts across models

---

## Deliverables Checklist

- [ ] Complete working code for all stages
- [ ] Comprehensive results in organized CSV structure
- [ ] Completed IEEE format LaTeX report
- [ ] All figures and tables generated
- [ ] Code submitted/uploaded
- [ ] Report PDF generated and submitted
- [ ] Verify all assignment requirements met

---

## Notes

### Current Issues
- Overfitting.py line 36: missing `from math import sqrt` or use `np.sqrt`
- Check if `.giitignore` should be `.gitignore`
- Optimize augmentation functions (3x3 jitter is slow)

### Key Files
- `VanillaModel.py` - Stage 1 baseline
- `Stage2Model.py` - Pruning + dual regularization
- `RigLModel.py` - Dynamic sparse training (Stage 3A)
- `SemiSupervisedModel.py` - Graph-based SSL (Stage 3B)
- `DataLoader.py` - Data preprocessing and augmentation
- `Overfitting.py` - Statistical overfitting detection
- `Results.py` - Hierarchical results logging
- `report.tex` - IEEE conference format report

### Useful Commands
```bash
# Run experiments
python a1.ipynb  # Or convert to .py script

# Generate report PDF
pdflatex report.tex
bibtex report
pdflatex report.tex
pdflatex report.tex
```

---

## Timeline (Suggested)

1. **Stage 1 Completion** - Finish experiments and verify results
2. **Stage 2 Completion** - Run pruning experiments
3. **Stage 3 Decision & Implementation** - Choose and complete advanced technique
4. **Results Collection** - Run all experiments with multiple seeds
5. **Statistical Analysis** - Compute means, stds, significance tests
6. **Report Writing** - Fill all sections with results
7. **Review & Submission** - Final checks and submit

---

**Last Updated:** 2026-02-13
