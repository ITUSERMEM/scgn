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

## Datasets

The repository supports three public datasets used in the paper. Download links are provided below.

| Dataset | Modality | Classes | Train / Val / Test | Source |
|---|---|---|---|---|
| **UORED-VAFCLS** | Vibration + Acoustic | 5 | 5% / 20% / 75% | [Mendeley Data](https://data.mendeley.com/datasets/y2px5tg92h/5) |
| **HUST** | Vibration (Z-axis) + Acoustic | 6 | 20% / 20% / 60% | [GitHub – HUSTmotor-multi-modal-dataset](https://github.com/CHAOZHAO-1/HUSTmotor-multi-modal-dataset) |
| **PU** | Vibration + Current | 9 | 20% / 20% / 60% | [Paderborn KAt DataCenter](https://mb.uni-paderborn.de/kat/forschung/bearing-datacenter/data-sets-and-download) |

> **Note:** Place each dataset in its own directory and pass the path to the corresponding training script.

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

### 3. Full training

**UORED-VAFCLS**
```bash
python train_uored.py /path/to/UORED-VAFCLS --seed 42 --epochs 50
```

**HUST Motor**
```bash
python train_hust.py /path/to/HUST --seed 42 --epochs 50
```

**PU Bearing**
```bash
python train_pu.py /path/to/PU --seed 42 --epochs 50
```

Each script trains the SCGN model and prints test accuracy, precision, recall and macro-F1.

---

## Reproducibility & Results

All results below were obtained with a single run (`seed=42`) on an NVIDIA RTX 3090 / A100 GPU. The reported paper values are means ± standard deviations over 10 independent runs.

### UORED-VAFCLS

```bash
python train_uored.py /path/to/UORED-VAFCLS --seed 42 --epochs 50
```

| Metric | This run | Paper (Table 6) |
|---|---|---|
| Accuracy | **93.72%** | 93.59 ± 1.61% |
| Precision | **93.81%** | — |
| Recall | **93.72%** | — |
| Macro-F1 | **93.67%** | 93.38 ± 1.57% |

### HUST Motor

```bash
python train_hust.py /path/to/HUST --seed 42 --epochs 50
```

| Metric | This run | Paper (Table 6) |
|---|---|---|
| Accuracy | **94.97%** | 93.87 ± 2.77% |
| Precision | **94.97%** | — |
| Recall | **94.97%** | — |
| Macro-F1 | **94.90%** | 93.82 ± 2.81% |

<details>
<summary>Training log (click to expand)</summary>

```
Epoch 01/50  loss=1.9641  val_acc=35.94%  val_f1=27.20%
Epoch 02/50  loss=1.6997  val_acc=58.85%  val_f1=54.20%
Epoch 03/50  loss=1.4966  val_acc=61.98%  val_f1=56.26%
Epoch 04/50  loss=1.4049  val_acc=67.19%  val_f1=63.01%
Epoch 05/50  loss=1.2859  val_acc=67.19%  val_f1=63.29%
Epoch 06/50  loss=1.1540  val_acc=75.52%  val_f1=74.24%
Epoch 07/50  loss=1.0941  val_acc=79.17%  val_f1=78.27%
Epoch 08/50  loss=0.9655  val_acc=79.17%  val_f1=78.70%
Epoch 09/50  loss=0.8916  val_acc=80.21%  val_f1=80.14%
Epoch 10/50  loss=0.8577  val_acc=81.25%  val_f1=81.28%
Epoch 11/50  loss=0.7886  val_acc=81.77%  val_f1=81.89%
Epoch 12/50  loss=0.7363  val_acc=83.85%  val_f1=84.04%
Epoch 13/50  loss=0.7138  val_acc=82.29%  val_f1=82.75%
Epoch 14/50  loss=0.6146  val_acc=83.85%  val_f1=84.25%
Epoch 15/50  loss=0.5532  val_acc=87.50%  val_f1=87.62%
Epoch 16/50  loss=0.5293  val_acc=89.06%  val_f1=89.00%
Epoch 17/50  loss=0.4946  val_acc=88.02%  val_f1=87.87%
Epoch 18/50  loss=0.4519  val_acc=85.94%  val_f1=85.86%
Epoch 19/50  loss=0.4039  val_acc=89.06%  val_f1=89.09%
Epoch 20/50  loss=0.3629  val_acc=90.10%  val_f1=90.02%
Epoch 21/50  loss=0.3523  val_acc=90.62%  val_f1=90.64%
Epoch 22/50  loss=0.3133  val_acc=91.15%  val_f1=91.11%
Epoch 23/50  loss=0.2900  val_acc=88.54%  val_f1=88.47%
Epoch 24/50  loss=0.2804  val_acc=90.62%  val_f1=90.50%
Epoch 25/50  loss=0.2434  val_acc=90.62%  val_f1=90.45%
Epoch 26/50  loss=0.2341  val_acc=91.67%  val_f1=91.58%
Epoch 27/50  loss=0.2207  val_acc=91.67%  val_f1=91.61%
Epoch 28/50  loss=0.1940  val_acc=89.58%  val_f1=89.48%
Epoch 29/50  loss=0.1850  val_acc=88.02%  val_f1=87.95%
Epoch 30/50  loss=0.1876  val_acc=89.06%  val_f1=88.95%
Epoch 31/50  loss=0.1757  val_acc=90.62%  val_f1=90.47%
Epoch 32/50  loss=0.1678  val_acc=92.19%  val_f1=91.95%
Epoch 33/50  loss=0.1651  val_acc=91.67%  val_f1=91.45%
Epoch 34/50  loss=0.1589  val_acc=92.71%  val_f1=92.57%
Epoch 35/50  loss=0.1473  val_acc=93.75%  val_f1=93.63%
Epoch 36/50  loss=0.1426  val_acc=94.27%  val_f1=94.20%
Epoch 37/50  loss=0.1323  val_acc=93.23%  val_f1=93.15%
Epoch 38/50  loss=0.1318  val_acc=93.23%  val_f1=93.15%
Epoch 39/50  loss=0.1301  val_acc=92.71%  val_f1=92.69%
Epoch 40/50  loss=0.1201  val_acc=93.23%  val_f1=93.15%
Epoch 41/50  loss=0.1240  val_acc=93.23%  val_f1=93.15%
Epoch 42/50  loss=0.1299  val_acc=92.71%  val_f1=92.62%
Epoch 43/50  loss=0.1227  val_acc=93.23%  val_f1=93.14%
Epoch 44/50  loss=0.1121  val_acc=93.75%  val_f1=93.66%
Epoch 45/50  loss=0.1304  val_acc=94.27%  val_f1=94.17%
Epoch 46/50  loss=0.1196  val_acc=94.27%  val_f1=94.17%
Early stopping triggered at epoch 46.

Test Evaluation (Best Validation Checkpoint)
Accuracy  : 94.97%
Precision : 94.97%
Recall    : 94.97%
Macro-F1  : 94.90%
```
</details>

### PU Bearing

```bash
python train_pu.py /path/to/PU --seed 42 --epochs 50
```

| Metric | This run | Paper (Table 5) |
|---|---|---|
| Accuracy | **97.63%** | 97.68 ± 0.75% |
| Precision | **97.66%** | 97.79 ± 0.67% |
| Recall | **97.63%** | — |
| Macro-F1 | **97.63%** | 97.67 ± 0.76% |

<details>
<summary>Training log (click to expand)</summary>

```
Epoch 01/50  loss=2.4021  val_acc=18.44%  val_f1=16.08%
Epoch 02/50  loss=2.1706  val_acc=41.11%  val_f1=33.38%
Epoch 03/50  loss=2.0364  val_acc=46.00%  val_f1=38.14%
Epoch 04/50  loss=1.9227  val_acc=59.11%  val_f1=55.04%
Epoch 05/50  loss=1.8087  val_acc=74.89%  val_f1=72.67%
Epoch 06/50  loss=1.7266  val_acc=77.33%  val_f1=75.74%
Epoch 07/50  loss=1.6002  val_acc=81.78%  val_f1=80.98%
Epoch 08/50  loss=1.4774  val_acc=85.78%  val_f1=85.38%
Epoch 09/50  loss=1.3880  val_acc=89.11%  val_f1=89.03%
Epoch 10/50  loss=1.3377  val_acc=92.67%  val_f1=92.64%
Epoch 11/50  loss=1.2076  val_acc=95.11%  val_f1=95.11%
Epoch 12/50  loss=1.1547  val_acc=96.22%  val_f1=96.22%
Epoch 13/50  loss=1.0896  val_acc=95.78%  val_f1=95.79%
Epoch 14/50  loss=1.0163  val_acc=95.11%  val_f1=95.09%
Epoch 15/50  loss=0.9825  val_acc=95.56%  val_f1=95.55%
Epoch 16/50  loss=0.9638  val_acc=96.00%  val_f1=96.01%
Epoch 17/50  loss=0.8930  val_acc=97.56%  val_f1=97.56%
Epoch 18/50  loss=0.8529  val_acc=98.22%  val_f1=98.23%
Epoch 19/50  loss=0.8070  val_acc=98.22%  val_f1=98.23%
Epoch 20/50  loss=0.7772  val_acc=98.22%  val_f1=98.23%
Epoch 21/50  loss=0.7410  val_acc=98.00%  val_f1=98.01%
Epoch 22/50  loss=0.7195  val_acc=97.56%  val_f1=97.56%
Epoch 23/50  loss=0.6940  val_acc=97.78%  val_f1=97.77%
Epoch 24/50  loss=0.6522  val_acc=97.78%  val_f1=97.77%
Epoch 25/50  loss=0.6387  val_acc=97.78%  val_f1=97.77%
Epoch 26/50  loss=0.6286  val_acc=97.56%  val_f1=97.56%
Epoch 27/50  loss=0.5886  val_acc=97.78%  val_f1=97.77%
Epoch 28/50  loss=0.5737  val_acc=97.78%  val_f1=97.77%
Early stopping triggered at epoch 28.

Test Evaluation (Best Validation Checkpoint)
Accuracy  : 97.63%
Precision : 97.66%
Recall    : 97.63%
Macro-F1  : 97.63%
```
</details>

---

## API Reference

### `scgn.dataset`

| Symbol | Description |
|--------|-------------|
| `UOREDDataset(data_dir)` | UORED-VAFCLS dataset loader. Returns dicts with `vib_time`, `aco_time`, `vib_spec`, `aco_spec`, `label`. |
| `HUSTDataset(data_dir)` | HUST motor dataset loader. |
| `PUDataset(data_root)` | Paderborn University bearing dataset loader. |
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
