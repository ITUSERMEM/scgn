"""
train_uored.py — Minimal Training Script for UORED-VAFCLS
==========================================================

Reproduces the main UORED experiment (Table IV in the paper) with the
proposed SCGN model.  Uses the default hyper-parameters reported in
Sec. IV-B:

  - train / val / test split : 5% / 20% / 75%
  - feature dimension d      : 64
  - GCN layers               : 1
  - k-NN neighbours k        : 10
  - optimiser                : AdamW
  - initial learning rate    : 1e-3
  - scheduler                : CosineAnnealingLR
  - max epochs               : 50
  - early-stopping patience  : 10

Usage
-----
    python train_uored.py <data_dir> [--seed 42] [--epochs 50]

Example
-------
    python train_uored.py /path/to/UORED-VAFCLS --seed 42 --epochs 50
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from scgn import (
    SCGN,
    UOREDDataset,
    build_fft_features,
    build_knn_adj_normalized,
    split_dataset,
)


def set_seed(seed: int):
    """Fix random seeds for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        try:
            torch.use_deterministic_algorithms(True)
        except RuntimeError as e:
            print(f"[WARN] deterministic algorithms not fully supported: {e}")


def evaluate(model, tensors, indices, adj_vib, adj_aco, fft_vib_feat, fft_aco_feat, device):
    """Evaluate accuracy, precision, recall and macro-F1 on a given index set."""
    model.eval()
    labels = tensors["labels"][indices].to(device)
    idx_t = torch.tensor(indices, dtype=torch.long, device=device)

    with torch.no_grad():
        logits = model(
            tensors["vib_time"].to(device),
            tensors["aco_time"].to(device),
            tensors["vib_spec"].to(device),
            tensors["aco_spec"].to(device),
            adj_vib,
            adj_aco,
            fft_vib_feat,
            fft_aco_feat,
            sample_idx=idx_t,
        )
    preds = logits.argmax(dim=1).cpu().numpy()
    labels_np = labels.cpu().numpy()

    return {
        "accuracy": accuracy_score(labels_np, preds) * 100.0,
        "precision": precision_score(labels_np, preds, average="macro", zero_division=0) * 100.0,
        "recall": recall_score(labels_np, preds, average="macro", zero_division=0) * 100.0,
        "f1": f1_score(labels_np, preds, average="macro", zero_division=0) * 100.0,
    }


def main():
    parser = argparse.ArgumentParser(description="Train SCGN on UORED-VAFCLS")
    parser.add_argument("data_dir", type=str, help="Path to UORED-VAFCLS root directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--epochs", type=int, default=50, help="Max training epochs")
    parser.add_argument("--patience", type=int, default=10, help="Early-stopping patience")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--train_ratio", type=float, default=0.05, help="Training ratio (default 0.05 for UORED)")
    parser.add_argument("--device", type=str, default="auto", help="cuda / cpu / auto")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    set_seed(args.seed)
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------------
    print("\n[1/4] Loading UORED-VAFCLS dataset ...")
    dataset = UOREDDataset(args.data_dir, seg_len=4096, overlap=0.0, max_segs_per_file=35)
    train_idx, val_idx, test_idx = split_dataset(dataset, train_ratio=args.train_ratio, val_ratio=0.20, seed=args.seed)
    print(f"Split -> train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    # ------------------------------------------------------------------
    # Preload tensors to device
    # ------------------------------------------------------------------
    print("[2/4] Preloading tensors ...")
    tensors = {
        "vib_time": torch.tensor(dataset.vib_signals, dtype=torch.float32).unsqueeze(1),
        "aco_time": torch.tensor(dataset.aco_signals, dtype=torch.float32).unsqueeze(1),
        "vib_spec": torch.tensor(dataset.vib_specs, dtype=torch.float32),
        "aco_spec": torch.tensor(dataset.aco_specs, dtype=torch.float32),
        "labels": torch.tensor(dataset.labels, dtype=torch.long),
    }

    # ------------------------------------------------------------------
    # Build inductive KNN graph + z-score FFT features
    # ------------------------------------------------------------------
    print("[3/4] Building inductive KNN graph (k=10) ...")
    fft_vib = build_fft_features(dataset.vib_signals, norm="zscore")
    fft_aco = build_fft_features(dataset.aco_signals, norm="zscore")

    adj_vib = build_knn_adj_normalized(fft_vib, train_idx, k=10).to(device)
    adj_aco = build_knn_adj_normalized(fft_aco, train_idx, k=10).to(device)

    fft_vib_feat = torch.tensor(fft_vib, dtype=torch.float32, device=device)
    fft_aco_feat = torch.tensor(fft_aco, dtype=torch.float32, device=device)

    # ------------------------------------------------------------------
    # Model, optimiser, loss
    # ------------------------------------------------------------------
    print("[4/4] Initialising SCGN model ...")
    model = SCGN(num_classes=5, d=64, gcn_layers=1, fft_dim=2048, mid_dropout=0.3).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    # ------------------------------------------------------------------
    # Training loop with early stopping
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Training")
    print("=" * 60)

    best_val_acc = -1.0
    patience_counter = 0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        optimizer.zero_grad()

        train_idx_t = torch.tensor(train_idx, dtype=torch.long, device=device)
        logits = model(
            tensors["vib_time"].to(device),
            tensors["aco_time"].to(device),
            tensors["vib_spec"].to(device),
            tensors["aco_spec"].to(device),
            adj_vib,
            adj_aco,
            fft_vib_feat,
            fft_aco_feat,
            sample_idx=train_idx_t,
        )
        loss = criterion(logits, tensors["labels"][train_idx].to(device))
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Validation
        val_metrics = evaluate(
            model, tensors, val_idx,
            adj_vib, adj_aco, fft_vib_feat, fft_aco_feat,
            device,
        )
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch:02d}/{args.epochs}  "
            f"loss={loss.item():.4f}  "
            f"val_acc={val_metrics['accuracy']:.2f}%  "
            f"val_f1={val_metrics['f1']:.2f}%  "
            f"({elapsed:.1f}s)"
        )

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping triggered at epoch {epoch}.")
                break

    # ------------------------------------------------------------------
    # Test evaluation (best validation checkpoint)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Test Evaluation (Best Validation Checkpoint)")
    print("=" * 60)
    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics = evaluate(
        model, tensors, test_idx,
        adj_vib, adj_aco, fft_vib_feat, fft_aco_feat,
        device,
    )

    print(f"Accuracy  : {test_metrics['accuracy']:.2f}%")
    print(f"Precision : {test_metrics['precision']:.2f}%")
    print(f"Recall    : {test_metrics['recall']:.2f}%")
    print(f"Macro-F1  : {test_metrics['f1']:.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
