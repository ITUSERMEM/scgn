# SCGN

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)

A deep learning framework that fuses vibration and acoustic signals via graph convolutional networks for rotating machinery fault diagnosis. The core design is to **enhance single-modality representation quality before fusion** while keeping the decision stage structurally simple.

**Three sequential stages**

1. **Frequency-domain graph construction & spectral propagation**  
   FFT magnitude spectra → cosine-similarity k-NN graphs → inductive GCN message passing

2. **Dual-domain local encoding**  
   1-D CNN on time-domain waveforms + 2-D CNN on STFT spectrograms

3. **Cross-modal joint discrimination**  
   Temporal–graph interaction projector → sensor-level aggregation → concatenation-based linear classifier

---

## Installation

```bash
# Clone the repository
git clone https://github.com/ITUSERMEM/scgn.git
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

### 3. Full training on UORED

```bash
python train_uored.py /path/to/UORED-VAFCLS --seed 42 --epochs 50
```

This script trains the SCGN model and prints test accuracy, precision, recall and macro-F1.

## API Reference

### `scgn.dataset`

| Symbol | Description |
|--------|-------------|
| `UOREDDataset(data_dir)` | UORED-VAFCLS dataset loader. Returns dicts with `vib_time`, `aco_time`, `vib_spec`, `aco_spec`, `label`. |
| `split_dataset(dataset, train_ratio, val_ratio, seed)` | Stratified random split. Returns `(train_idx, val_idx, test_idx)`. |
| `build_fft_features(signals, norm="l2")` | Normalised FFT-magnitude features for graph construction. |

### `scgn.encoders`

| Symbol | Description |
|--------|-------------|
| `CNN1DEncoder(out_dim=64)` | Temporal 1-D CNN encoder with InstanceNorm. |
| `CNN2DEncoder(out_dim=64)` | Spectrogram 2-D CNN encoder with InstanceNorm. |

### `scgn.graph_builder`

| Symbol | Description |
|--------|-------------|
| `build_hetero_graph_inductive(vib_fft, aco_fft, train_idx, k=10)` | Inductive KNN graph in FFT space (legacy HeteroData API). |
| `build_knn_adj_normalized(fft_features, train_idx, k=10)` | Dense row-normalized KNN adjacency (recommended). |

### `scgn.model`

| Symbol | Description |
|--------|-------------|
| `SCGN(num_classes=5, d=64, gcn_layers=1, fft_dim=2048)` | End-to-end SCGN model. |

---

## License

This project is released under the MIT License.
