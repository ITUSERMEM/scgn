# SCGN — Spectral Coupling Graph Network for Bimodal Information

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)

Official PyTorch implementation of the paper:

> **Rotating Machinery Fault Diagnosis via a Spectral Coupling Graph Network for Bimodal Information**  
> *Information Fusion* (Elsevier)

---

## Overview

The **Spectral Coupling Graph Network (SCGN)** is a graph-guided bimodal fault-diagnosis framework designed for rotating machinery.  Its core design philosophy is to **enhance single-modality representation quality before fusion** while keeping the decision stage structurally simple.

**Three sequential stages (Sec. III of the paper)**

1. **Frequency-domain graph construction & spectral propagation**  
   FFT magnitude spectra → cosine-similarity k-NN graphs → inductive GCN message passing (Eq. 4–7)

2. **Dual-domain local encoding**  
   1-D CNN on time-domain waveforms + 2-D CNN on STFT spectrograms (Eq. 8–9)

3. **Cross-modal joint discrimination**  
   Temporal–graph interaction projector → sensor-level aggregation → concatenation-based linear classifier (Eq. 10–14)

<p align="center">
  <img src="https://via.placeholder.com/800x300?text=SCGN+Architecture+Diagram" alt="SCGN architecture" width="90%">
  <br>
  <em>Overall architecture: two symmetric sensor branches with independent parameters.</em>
</p>

---

## Installation

```bash
# Clone the repository
git clone https://github.com/yourname/scgn.git
cd scgn

# Install dependencies
pip install -r requirements.txt
```

**Requirements**
- Python ≥ 3.10
- PyTorch ≥ 2.0
- NumPy, pandas, SciPy, scikit-learn

---

## Quick Start

### 1. Synthetic smoke test (no dataset required)

```bash
python demo.py
```

### 2. Real-data forward pass (UORED-VAFCLS dataset)

```bash
python demo.py /path/to/UORED-VAFCLS
```

### 3. Full training on UORED (reproduces Table IV)

```bash
python train_uored.py /path/to/UORED-VAFCLS --seed 42 --epochs 50
```

This script trains the proposed SCGN with the exact hyper-parameters reported
in Sec. IV-B of the paper and prints test accuracy, precision, recall and
macro-F1.

### 4. Minimal training snippet

```python
import torch
from scgn import (
    UOREDDataset, split_dataset,
    build_fft_features, build_hetero_graph_inductive, SCGN
)

# Load data
dataset = UOREDDataset("/path/to/UORED-VAFCLS")
train_idx, val_idx, test_idx = split_dataset(dataset, train_ratio=0.20, val_ratio=0.20)

# Build graph
fft_vib = build_fft_features(dataset.vib_signals)
fft_aco = build_fft_features(dataset.aco_signals)
hetero_data = build_hetero_graph_inductive(fft_vib, fft_aco, train_idx, k=10)

# Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SCGN(num_classes=5, d=64, gcn_layers=1, fft_dim=2048).to(device)

# Training loop (illustrative)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
criterion = torch.nn.CrossEntropyLoss()

model.train()
# ... load tensors, forward, backward, step ...
```

---

## API Reference

### `scgn.dataset`

| Symbol | Description |
|--------|-------------|
| `UOREDDataset(data_dir)` | UORED-VAFCLS dataset loader. Returns dicts with `vib_time`, `aco_time`, `vib_spec`, `aco_spec`, `label`. |
| `split_dataset(dataset, train_ratio, val_ratio, seed)` | Stratified random split. Returns `(train_idx, val_idx, test_idx)`. |
| `build_fft_features(signals)` | Normalised FFT-magnitude features for graph construction (Eq. 4). |

### `scgn.encoders`

| Symbol | Description |
|--------|-------------|
| `CNN1DEncoder(out_dim=64)` | Temporal 1-D CNN encoder with InstanceNorm (Eq. 8–9). |
| `CNN2DEncoder(out_dim=64)` | Spectrogram 2-D CNN encoder with InstanceNorm. |

### `scgn.graph_builder`

| Symbol | Description |
|--------|-------------|
| `build_hetero_graph_inductive(vib_fft, aco_fft, train_idx, k=10)` | Inductive KNN graph in FFT space (Eq. 5–6). |

### `scgn.model`

| Symbol | Description |
|--------|-------------|
| `SCGN(num_classes=5, d=64, gcn_layers=1, fft_dim=2048)` | End-to-end SCGN model (Sec. III). |

---

## Paper-to-Code Mapping

| Paper Section | File | Symbol |
|---------------|------|--------|
| Sec. III-B-1  Frequency-Domain Node Features | `dataset.py` | `build_fft_features` |
| Sec. III-B-2  Inductive k-NN Adjacency | `graph_builder.py` | `build_hetero_graph_inductive` |
| Sec. III-B-3  Spectral Graph Convolution | `model.py` | `GCNLayer`, `GCNStack` |
| Sec. III-C    Local Temporal & Spectro-Temporal Encoding | `encoders.py` | `CNN1DEncoder`, `CNN2DEncoder` |
| Sec. III-D-1  Temporal–Graph Interaction Projection | `model.py` | `TFProjector` |
| Sec. III-D-2  Sensor-Level Aggregation | `model.py` | `SymmetricAggregator` |
| Sec. III-E    Cross-Modal Joint Decision Head | `model.py` | `ConcatFusion` |
| Sec. III-A    Overall Architecture | `model.py` | `SCGN` |

---

## Citation

If you find this work useful, please consider citing:

```bibtex
@article{scgn2025,
  title={Rotating Machinery Fault Diagnosis via a Spectral Coupling Graph Network for Bimodal Information},
  journal={Information Fusion},
  publisher={Elsevier},
  year={2025},
  note={Under review}
}
```

---

## License

This project is released under the MIT License.

---

## Contact

For questions or issues, please open a GitHub issue or contact the corresponding author.
