"""
demo.py — Minimal end-to-end example of SCGN
=============================================

This script demonstrates:
  1. Loading the UORED-VAFCLS dataset
  2. Building inductive KNN graphs in FFT-magnitude space
  3. Running a forward pass through the SCGN model
  4. Printing tensor shapes at each stage

Usage
-----
    python demo.py <path_to_UORED_dataset>

If no dataset path is provided, the script falls back to synthetic data
so that it can still be executed for a smoke test.
"""

import sys
import torch

from scgn import (
    UOREDDataset,
    split_dataset,
    build_fft_features,
    build_knn_adj_normalized,
    SCGN,
)


def demo_synthetic():
    """Run a smoke test with synthetic data (no external dataset required)."""
    print("=" * 70)
    print("SCGN Demo — Synthetic Data Smoke Test")
    print("=" * 70)

    N, L = 128, 4096
    num_classes = 5
    d = 64
    fft_dim = 2048

    # Synthetic tensors
    vib_time = torch.randn(N, 1, L)
    aco_time = torch.randn(N, 1, L)
    vib_spec = torch.randn(N, 1, 129, 53)
    aco_spec = torch.randn(N, 1, 129, 53)
    labels = torch.randint(0, num_classes, (N,))

    # Synthetic FFT features for graph construction
    vib_fft = torch.randn(N, fft_dim).numpy()
    aco_fft = torch.randn(N, fft_dim).numpy()
    train_idx = torch.arange(0, N // 2).numpy()

    # Build explicit adjacency + FFT features
    adj_vib = build_knn_adj_normalized(vib_fft, train_idx, k=10)
    adj_aco = build_knn_adj_normalized(aco_fft, train_idx, k=10)
    fft_vib_feat = torch.tensor(vib_fft, dtype=torch.float32)
    fft_aco_feat = torch.tensor(aco_fft, dtype=torch.float32)

    # Model
    model = SCGN(num_classes=num_classes, d=d, gcn_layers=1, fft_dim=fft_dim)
    model.eval()

    with torch.no_grad():
        logits = model(
            vib_time, aco_time, vib_spec, aco_spec,
            adj_vib, adj_aco, fft_vib_feat, fft_aco_feat,
        )

    print(f"\nInput shapes:")
    print(f"  vib_time : {tuple(vib_time.shape)}")
    print(f"  aco_time : {tuple(aco_time.shape)}")
    print(f"  vib_spec : {tuple(vib_spec.shape)}")
    print(f"  aco_spec : {tuple(aco_spec.shape)}")
    print(f"\nGraph adjacency shapes:")
    print(f"  adj_vib : {tuple(adj_vib.shape)}")
    print(f"  adj_aco : {tuple(adj_aco.shape)}")
    print(f"\nOutput logits : {tuple(logits.shape)}")
    print(f"Predicted class : {logits.argmax(dim=1)[:10].tolist()} ...")
    print("\nOK — synthetic smoke test passed.")


def demo_real(data_dir: str):
    """Run a forward pass on real UORED data."""
    print("=" * 70)
    print("SCGN Demo — Real UORED Data Forward Pass")
    print("=" * 70)

    # 1. Load dataset
    dataset = UOREDDataset(data_dir, seg_len=4096, overlap=0.0, max_segs_per_file=10)

    # 2. Split
    train_idx, val_idx, test_idx = split_dataset(dataset, train_ratio=0.20, val_ratio=0.20)
    print(f"\nDataset split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    # 3. Preload tensors
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vib_time = torch.tensor(dataset.vib_signals, dtype=torch.float32, device=device).unsqueeze(1)
    aco_time = torch.tensor(dataset.aco_signals, dtype=torch.float32, device=device).unsqueeze(1)
    vib_spec = torch.tensor(dataset.vib_specs, dtype=torch.float32, device=device)
    aco_spec = torch.tensor(dataset.aco_specs, dtype=torch.float32, device=device)
    labels = torch.tensor(dataset.labels, dtype=torch.long, device=device)

    # 4. Build FFT features & explicit adjacency
    fft_vib = build_fft_features(dataset.vib_signals, norm="zscore")
    fft_aco = build_fft_features(dataset.aco_signals, norm="zscore")
    adj_vib = build_knn_adj_normalized(fft_vib, train_idx, k=10).to(device)
    adj_aco = build_knn_adj_normalized(fft_aco, train_idx, k=10).to(device)
    fft_vib_feat = torch.tensor(fft_vib, dtype=torch.float32, device=device)
    fft_aco_feat = torch.tensor(fft_aco, dtype=torch.float32, device=device)

    # 5. Model
    model = SCGN(num_classes=5, d=64, gcn_layers=1, fft_dim=2048).to(device)
    model.eval()

    with torch.no_grad():
        logits = model(
            vib_time, aco_time, vib_spec, aco_spec,
            adj_vib, adj_aco, fft_vib_feat, fft_aco_feat,
        )

    preds = logits.argmax(dim=1)
    acc = (preds == labels).float().mean().item()

    print(f"\nInput shapes:")
    print(f"  vib_time : {tuple(vib_time.shape)}")
    print(f"  aco_time : {tuple(aco_time.shape)}")
    print(f"  vib_spec : {tuple(vib_spec.shape)}")
    print(f"  aco_spec : {tuple(aco_spec.shape)}")
    print(f"\nOutput logits : {tuple(logits.shape)}")
    print(f"Accuracy (untrained, random weights) : {acc * 100:.2f}%")
    print("\nOK — real-data forward pass passed.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("No dataset path provided — running synthetic smoke test.\n")
        demo_synthetic()
    else:
        demo_real(sys.argv[1])
