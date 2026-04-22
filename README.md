# Self-Pruning Neural Network — CIFAR-10

> Tredence Analytics | AI Engineering Intern Case Study  
> Implementation of a neural network that learns to prune itself during training.

---

## What This Does

A standard feed-forward network is augmented with **learnable gate parameters** (one per weight). These gates are passed through a sigmoid to produce values in (0, 1). An L1 penalty on the gates during training drives many of them to near-zero — effectively removing ("pruning") those weights from the network **while training**, without any separate post-training pruning step.

```
Total Loss = CrossEntropy + λ × Σ sigmoid(gate_scores)
```

---

## Project Structure

```
.
├── train.py              # All-in-one script: model, training, evaluation, plots
├── REPORT.md             # Analysis report with sparsity explanation & results table
├── README.md             # This file
├── gate_distribution.png # Auto-generated: gate histogram for best model
├── results_table.csv     # Auto-generated: Lambda | Test Accuracy | Sparsity
└── data/                 # CIFAR-10 downloaded here automatically
```

---

## Requirements

```bash
pip install torch torchvision matplotlib numpy
```

Tested with:
- Python 3.9+
- PyTorch 2.x
- torchvision 0.15+

GPU is optional but speeds up training significantly (CUDA auto-detected).

---

## How to Run

```bash
python train.py
```

That's it. The script will:
1. Download CIFAR-10 automatically to `./data/`
2. Train the self-pruning network for **3 lambda values** (1e-4, 5e-4, 2e-3), 40 epochs each
3. Print a results table to the console
4. Save `results_table.csv` and `gate_distribution.png`

**Expected runtime:**
- With GPU: ~5–8 minutes total
- Without GPU (CPU): ~30–60 minutes total

---

## Expected Output

Console output will look like:

```
Device: cuda

=======================================================
  Training with λ = 1e-04
=======================================================
  Epoch  10/40 | Loss: 1.8234 | Train Acc: 42.11% | Test Acc: 43.50% | Sparsity: 99.92%
  Epoch  20/40 | Loss: 1.6102 | Train Acc: 48.72% | Test Acc: 49.01% | Sparsity: 99.94%
  Epoch  30/40 | Loss: 1.5341 | Train Acc: 52.10% | Test Acc: 51.88% | Sparsity: 99.97%
  Epoch  40/40 | Loss: 1.4923 | Train Acc: 54.33% | Test Acc: 53.21% | Sparsity: 99.97%

  ► Final Test Accuracy : 53.21%
  ► Final Sparsity      : 99.97%
...
```

Final results table (also saved to `results_table.csv`):

```
=======================================================
  RESULTS SUMMARY
=======================================================
  Lambda       Test Acc (%)     Sparsity (%)
  -----------------------------------------
  1e-04        ~52–55           ~90-99
  5e-04        ~48–52           ~99.95
  2e-03        ~43–48           ~100
=======================================================
```

---

## Key Design Decisions

### PrunableLinear Layer

```python
class PrunableLinear(nn.Module):
    def __init__(self, in_features, out_features):
        # weight:      standard learnable weights
        # gate_scores: learnable gate logits (same shape as weight)

    def forward(self, x):
        gates         = torch.sigmoid(self.gate_scores)   # → (0, 1)
        pruned_weight = self.weight * gates               # element-wise
        return F.linear(x, pruned_weight, self.bias)
```

- Gradients flow through both `weight` and `gate_scores` automatically via PyTorch autograd
- `gate_scores` initialized to zero → `sigmoid(0) = 0.5` → balanced starting point
- Kaiming initialization for `weight` → stable training with ReLU

### Sparsity Loss

```python
def sparsity_loss(self):
    # L1 norm of gates = sum of all sigmoid(gate_score) values
    return sum(sigmoid(gate_scores).sum() for each PrunableLinear layer)
```

L1 (not L2) is used because it produces a **constant gradient magnitude** regardless of the gate value's size. This means even tiny gate values keep getting pushed toward zero — achieving exact sparsity. L2 would only shrink values asymptotically.

### Why Sparsity Is Never Zero

Even at the lowest λ (1e-5), the L1 penalty produces a consistent push on every gate. After 40 epochs, a meaningful fraction of gates will have been driven below the 1e-2 threshold. The mechanism is always active — sparsity level scales predictably with λ.

---

## Hyperparameter Tuning Notes

| Hyperparameter | Value | Reason |
|---|---|---|
| Optimizer | Adam, lr=1e-3 | Fast convergence for both weights and gates |
| LR Schedule | CosineAnnealingLR | Smooth decay, prevents oscillation near convergence |
| Epochs | 40 | Enough for gates to settle into pruned/active state |
| Dropout | 0.3, 0.3, 0.2 | Regularize alongside sparsity loss |
| Batch Size | 128 | Good balance of gradient quality and speed |
| weight_decay | 1e-4 on weights, 0 on gates | L2 on gates would conflict with L1 sparsity goal |
| Gate threshold | 1e-2 | Conservative: gates below 1% considered pruned |

---

## Results Interpretation

The λ parameter controls the **sparsity–accuracy trade-off**:

- **Low λ (1e-5):** Network focuses on accuracy. Few gates pruned. Best for deployment where accuracy is critical.
- **Medium λ (5e-5):** Balanced. Good accuracy, meaningful compression. Recommended for most use cases.
- **High λ (2e-4):** Aggressive compression. Significantly fewer active connections. Best when memory/compute is severely constrained.

The `gate_distribution.png` plot for the best model shows a **bimodal distribution**:
- Large spike near 0: pruned weights
- Smaller cluster near 1: important retained weights

This bimodality is the hallmark of successful L1-based sparsity learning.
