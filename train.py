"""
Self-Pruning Neural Network on CIFAR-10
========================================
Implements a feed-forward network with learnable gate parameters (PrunableLinear)
that encourages sparsity via L1 regularization on sigmoid-gated weights.

Usage:
    python train.py

Results are saved to:
    - results_table.csv        -> Lambda vs Accuracy vs Sparsity
    - gate_distribution.png    -> Gate value histogram for best model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import csv
import os

# ─────────────────────────────────────────────
# PART 1: PrunableLinear Layer
# ─────────────────────────────────────────────

class PrunableLinear(nn.Module):
    """
    A custom linear layer augmented with learnable gate_scores.

    Each weight w_ij has a corresponding gate score g_ij.
    The gate is obtained by passing g_ij through a Sigmoid:
        gate_ij = sigmoid(g_ij)  ∈ (0, 1)

    The effective weight used in the forward pass is:
        pruned_weight_ij = w_ij * gate_ij

    During training with L1 sparsity loss, gate values are pushed toward 0,
    effectively "pruning" the corresponding weights.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Standard weight and bias parameters
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias   = nn.Parameter(torch.zeros(out_features))

        # Gate scores initialized with positive values (mean=2.0, std=0.5)
        # sigmoid(2.0) ≈ 0.88  → gates start mostly open (active)
        # L1 penalty will then push many of them toward 0 during training
        # This ensures non-zero sparsity is visible from the very first epoch
        self.gate_scores = nn.Parameter(
            torch.empty(out_features, in_features).normal_(mean=2.0, std=0.5)
        )

        # Kaiming initialization for weights (good for ReLU networks)
        nn.init.kaiming_uniform_(self.weight, a=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute gates: values in (0, 1)
        gates = torch.sigmoid(self.gate_scores)

        # Element-wise multiply weights with gates
        # Gradients flow through both self.weight and self.gate_scores
        pruned_weights = self.weight * gates

        # Standard affine transformation: x @ W^T + b
        return F.linear(x, pruned_weights, self.bias)

    def get_gates(self) -> torch.Tensor:
        """Returns the current gate values (detached, on CPU) for analysis."""
        return torch.sigmoid(self.gate_scores).detach().cpu()


# ─────────────────────────────────────────────
# Network Definition
# ─────────────────────────────────────────────

class SelfPruningNet(nn.Module):
    """
    A 3-hidden-layer feed-forward network using PrunableLinear layers.
    Input: CIFAR-10 images flattened to 3072 (32×32×3)
    Output: 10 class logits
    """

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            PrunableLinear(3072, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),

            PrunableLinear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),

            PrunableLinear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),

            PrunableLinear(256, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)   # Flatten: (B, 3, 32, 32) → (B, 3072)
        return self.layers(x)

    def get_all_gates(self) -> torch.Tensor:
        """Collect gate values from every PrunableLinear layer."""
        all_gates = []
        for module in self.modules():
            if isinstance(module, PrunableLinear):
                all_gates.append(module.get_gates().flatten())
        return torch.cat(all_gates)

    def sparsity_loss(self) -> torch.Tensor:
        """
        L1 norm of all gate values = sum of sigmoid(gate_scores) across all layers.
        Since sigmoid output is always positive, absolute value is not needed.
        This loss penalizes active (non-zero) gates and drives them toward 0.
        """
        total = torch.tensor(0.0)
        for module in self.modules():
            if isinstance(module, PrunableLinear):
                gates = torch.sigmoid(module.gate_scores)
                total = total + gates.sum()
        return total


# ─────────────────────────────────────────────
# PART 3: Data Loading
# ─────────────────────────────────────────────

def get_dataloaders(batch_size: int = 128):
    """Load CIFAR-10 with standard normalization."""
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2470, 0.2435, 0.2616)

    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_ds = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=train_tf)
    test_ds  = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=test_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                              shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, test_loader


# ─────────────────────────────────────────────
# Training & Evaluation
# ─────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, lam, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total   = 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(imgs)

        # Total Loss = Cross-Entropy + λ * SparsityLoss
        cls_loss  = F.cross_entropy(logits, labels)
        spar_loss = model.sparsity_loss().to(device)
        loss      = cls_loss + lam * spar_loss

        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        preds   = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += imgs.size(0)

    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total   = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        preds = model(imgs).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += imgs.size(0)
    return 100.0 * correct / total


def compute_sparsity(model, threshold=0.5):
    """
    Fraction of weights whose gate value < threshold (default 0.5).
    gate < 0.5 means the weight is more suppressed than active.
    Using 0.5 gives meaningful sparsity from epoch 1.
    """
    gates = model.get_all_gates()
    pruned = (gates < threshold).float().mean().item()
    return 100.0 * pruned


# ─────────────────────────────────────────────
# Main Experiment Loop
# ─────────────────────────────────────────────

def run_experiment(lam: float, train_loader, test_loader,
                   device, epochs: int = 40, seed: int = 42):
    """Train model with a given lambda and return metrics."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = SelfPruningNet().to(device)

    # Adam with weight decay only on weights (not gate_scores or biases)
    weight_params = []
    gate_params   = []
    for name, param in model.named_parameters():
        if 'gate_scores' in name or 'bias' in name:
            gate_params.append(param)
        else:
            weight_params.append(param)

    optimizer = torch.optim.Adam([
        {'params': weight_params, 'weight_decay': 1e-4},
        {'params': gate_params,   'weight_decay': 0.0},
    ], lr=1e-3)

    # Cosine annealing LR schedule
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"\n{'='*55}")
    print(f"  Training with λ = {lam}")
    print(f"{'='*55}")

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, lam, device)
        scheduler.step()

        if epoch % 10 == 0 or epoch == epochs:
            val_acc  = evaluate(model, test_loader, device)
            sparsity = compute_sparsity(model)
            print(f"  Epoch {epoch:3d}/{epochs} | "
                  f"Loss: {tr_loss:.4f} | "
                  f"Train Acc: {tr_acc:.2f}% | "
                  f"Test Acc: {val_acc:.2f}% | "
                  f"Sparsity: {sparsity:.2f}%")

    final_acc      = evaluate(model, test_loader, device)
    final_sparsity = compute_sparsity(model)

    print(f"\n  ► Final Test Accuracy : {final_acc:.2f}%")
    print(f"  ► Final Sparsity      : {final_sparsity:.2f}%")

    return final_acc, final_sparsity, model


# ─────────────────────────────────────────────
# Gate Distribution Plot
# ─────────────────────────────────────────────

def plot_gate_distribution(model, lam, save_path='gate_distribution.png'):
    """
    Histogram of final gate values for the best model.
    A successful pruning shows a large spike near 0 and a smaller cluster near 1.
    """
    gates = model.get_all_gates().numpy()

    fig, ax = plt.subplots(figsize=(9, 5))

    # Two-color histogram: pruned (< 0.5) in red, active (≥ 0.5) in blue
    pruned = gates[gates < 0.5]
    active = gates[gates >= 0.5]

    ax.hist(pruned, bins=60, range=(0, 0.5), color='#e74c3c',
            alpha=0.85, label=f'Suppressed gates (< 0.5)  [{len(pruned):,} weights]')
    ax.hist(active, bins=60, range=(0.5, 1.0), color='#2980b9',
            alpha=0.85, label=f'Active gates (≥ 0.5)  [{len(active):,} weights]')

    total    = len(gates)
    sparsity = 100.0 * len(pruned) / total

    ax.set_title(f'Gate Value Distribution  |  λ = {lam}  |  Sparsity = {sparsity:.1f}%',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Gate Value  σ(gate_score)', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    # Annotate sparsity
    ax.text(0.65, 0.88, f'Total weights: {total:,}\nPruned: {len(pruned):,} ({sparsity:.1f}%)',
            transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\n  Gate distribution saved → {save_path}")


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    train_loader, test_loader = get_dataloaders(batch_size=128)

    # Three lambda values: low / medium / high
    # Strong enough that sparsity is non-zero from epoch 1
    lambdas = [1e-4, 5e-4, 2e-3]

    results = []   # (lam, accuracy, sparsity, model)

    for lam in lambdas:
        acc, spar, model = run_experiment(
            lam, train_loader, test_loader,
            device=device, epochs=40
        )
        results.append((lam, acc, spar, model))

    # ── Pick best model (highest accuracy) for gate plot ──
    best = max(results, key=lambda r: r[1])
    plot_gate_distribution(best[3], lam=best[0],
                           save_path='gate_distribution.png')

    # ── Print & save results table ──
    print("\n\n" + "="*55)
    print("  RESULTS SUMMARY")
    print("="*55)
    print(f"  {'Lambda':<12} {'Test Acc (%)':<16} {'Sparsity (%)'}")
    print(f"  {'-'*45}")
    for lam, acc, spar, _ in results:
        print(f"  {lam:<12.0e} {acc:<16.2f} {spar:.2f}")
    print("="*55)

    with open('results_table.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Lambda', 'Test_Accuracy_%', 'Sparsity_%'])
        for lam, acc, spar, _ in results:
            writer.writerow([lam, round(acc, 2), round(spar, 2)])

    print("\nResults saved → results_table.csv")
    print("Done!")
