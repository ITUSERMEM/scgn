"""
dataset.py — UORED-VAFCLS Dataset Loader
========================================

Dataset: University of Ottawa Rolling-element Dataset
         Vibration and Acoustic Faults under Constant Load and Speed conditions

Sensors : Accelerometer (vibration) + Microphone (acoustic)
Sampling: 42,000 Hz (both modalities, no alignment needed)
Per file: 420,000 points = 10 seconds

Classes (5):
  0 — Healthy        (H_*_0.csv, 20 files)
  1 — Inner Race     (I_*_{1,2}.csv, 10 files)
  2 — Outer Race     (O_*_{1,2}.csv, 10 files)
  3 — Ball           (B_*_{1,2}.csv, 10 files)
  4 — Cage           (C_*_{1,2}.csv, 10 files)

CSV columns: Accelerometer, Acoustic, Speed, Load, Temperature Difference

Paper: Sec. IV-A, Table 1 (UORED conditions)
"""

import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import scipy.signal as signal
import torch
from torch.utils.data import Dataset


# ------------------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------------------
UORED_CLASSES = ["Healthy", "InnerRace", "OuterRace", "Ball", "Cage"]
NUM_CLASSES = 5

SUBFOLDER_MAP = {
    "1_Healthy":           {"prefix": "H", "label": 0},
    "2_Inner_Race_Faults": {"prefix": "I", "label": 1},
    "3_Outer_Race_Faults": {"prefix": "O", "label": 2},
    "4_Ball_Faults":       {"prefix": "B", "label": 3},
    "5_Cage_Faults":       {"prefix": "C", "label": 4},
}

FS = 42_000               # Sampling rate (Hz), identical for both modalities
SEGMENT_LEN = 4096        # Segment length (samples, ~0.0975 s @ 42 kHz)
SAMPLES_PER_FILE = 420_000

TRAIN_RATIO = 0.20
VAL_RATIO = 0.20
TEST_RATIO = 0.60


# ------------------------------------------------------------------------------
# CSV I/O
# ------------------------------------------------------------------------------
def load_csv_signals(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """Read a UORED CSV file and return vibration and acoustic signals.

    Returns
    -------
    acc : np.ndarray, shape (N,)
        Accelerometer signal.
    aco : np.ndarray, shape (N,)
        Acoustic signal.
    """
    df = pd.read_csv(filepath)
    acc = df["Accelerometer"].values.astype(np.float64)
    aco = df["Acoustic"].values.astype(np.float64)
    return acc, aco


# ------------------------------------------------------------------------------
# Signal preprocessing
# ------------------------------------------------------------------------------
def segment_signal(
    sig: np.ndarray, seg_len: int = SEGMENT_LEN, overlap: float = 0.0
) -> np.ndarray:
    """Slice a long 1-D signal into fixed-length segments.

    Parameters
    ----------
    sig : np.ndarray, shape (N,)
    seg_len : int
        Samples per segment.
    overlap : float
        Overlap ratio (0.0 = no overlap).

    Returns
    -------
    segments : np.ndarray, shape (num_segments, seg_len)
    """
    step = int(seg_len * (1 - overlap))
    num_segs = (len(sig) - seg_len) // step + 1

    segments = np.zeros((num_segs, seg_len), dtype=sig.dtype)
    for i in range(num_segs):
        start = i * step
        segments[i] = sig[start : start + seg_len]

    return segments


def compute_stft_spectrogram(
    signal_1d: np.ndarray,
    n_fft: int = 256,
    hop_length: int = 80,
    fs: int = FS,
) -> np.ndarray:
    """Compute a log-compressed STFT magnitude spectrogram.

    For 4096-point signal @ 42 kHz with default settings:
        freq_bins  = n_fft / 2 + 1 = 129
        time_steps = (4096 - 256) / 80 + 1 = 49

    Returns
    -------
    spectrogram : np.ndarray, shape (1, 129, time_steps)
        Log-compressed magnitude spectrogram.
    """
    window = np.hanning(n_fft)
    _, _, Zxx = signal.stft(
        signal_1d,
        fs=fs,
        window=window,
        nperseg=n_fft,
        noverlap=n_fft - hop_length,
    )
    mag = np.abs(Zxx)
    mag = np.log1p(mag)
    return mag[np.newaxis, :, :]  # (1, F, T)


# ------------------------------------------------------------------------------
# Dataset class
# ------------------------------------------------------------------------------
class UOREDDataset(Dataset):
    """UORED-VAFCLS vibration + acoustic fusion dataset.

    Each sample contains:
      - vib_time : (1, 4096)   time-domain vibration signal
      - aco_time : (1, 4096)   time-domain acoustic signal
      - vib_spec : (1, 129, ~49)  vibration STFT spectrogram
      - aco_spec : (1, 129, ~49)  acoustic STFT spectrogram
      - label    : int in [0, 4]

    Paper: Sec. IV-A
    """

    def __init__(
        self,
        data_dir: str,
        seg_len: int = SEGMENT_LEN,
        overlap: float = 0.0,
        max_segs_per_file: int = 35,
    ):
        """
        Parameters
        ----------
        data_dir : str
            Root directory containing the sub-folder
            '1_CSV_Raw_Data_Files (.csv)/'.
        seg_len : int
            Segment length in samples.
        overlap : float
            Overlap ratio between consecutive segments.
        max_segs_per_file : int
            Maximum segments extracted per CSV file.  Default 35 yields
            60 files x 35 = 2100 samples total.  Set <= 0 for unlimited.
        """
        self.data_dir = data_dir
        self.csv_dir = os.path.join(data_dir, "1_CSV_Raw_Data_Files (.csv)")
        self.seg_len = seg_len

        if not os.path.isdir(self.csv_dir):
            raise FileNotFoundError(
                f"UORED dataset directory not found: {self.csv_dir}\n"
                f"Please ensure the data_dir points to the UORED-VAFCLS root "
                f"and contains the sub-folder '1_CSV_Raw_Data_Files (.csv)/'."
            )

        all_vib_segs = []
        all_aco_segs = []
        all_labels = []

        print(f"[UOREDDataset] Loading data from: {data_dir}")

        for subfolder, info in SUBFOLDER_MAP.items():
            folder_path = os.path.join(self.csv_dir, subfolder)
            if not os.path.isdir(folder_path):
                print(f"  Warning: sub-folder missing, skipping: {subfolder}")
                continue

            csv_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".csv")])
            label = info["label"]
            class_name = UORED_CLASSES[label]

            class_seg_count = 0
            for csv_file in csv_files:
                filepath = os.path.join(folder_path, csv_file)
                acc, aco = load_csv_signals(filepath)

                vib_segs = segment_signal(acc, seg_len, overlap)
                aco_segs = segment_signal(aco, seg_len, overlap)

                n_segs = min(len(vib_segs), len(aco_segs))
                if max_segs_per_file > 0:
                    n_segs = min(n_segs, max_segs_per_file)
                vib_segs = vib_segs[:n_segs]
                aco_segs = aco_segs[:n_segs]

                all_vib_segs.append(vib_segs)
                all_aco_segs.append(aco_segs)
                all_labels.extend([label] * n_segs)
                class_seg_count += n_segs

            print(
                f"  {class_name:12s} label={label}  files={len(csv_files):2d}  "
                f"-> {class_seg_count} segments"
            )

        # Concatenate across files
        self.vib_signals = np.concatenate(all_vib_segs, axis=0)  # (N, 4096)
        self.aco_signals = np.concatenate(all_aco_segs, axis=0)  # (N, 4096)
        self.labels = np.array(all_labels, dtype=np.int64)

        # Per-sample z-score normalization
        self._normalize_zscore()

        # Pre-compute STFT spectrograms
        print("  Computing STFT spectrograms ...")
        self.vib_specs = np.stack(
            [compute_stft_spectrogram(self.vib_signals[i]) for i in range(len(self))]
        )
        self.aco_specs = np.stack(
            [compute_stft_spectrogram(self.aco_signals[i]) for i in range(len(self))]
        )

        # Summary statistics
        total = len(self.labels)
        for c in range(NUM_CLASSES):
            cnt = int((self.labels == c).sum())
            print(
                f"  Class {c} ({UORED_CLASSES[c]}): {cnt} samples ({cnt / total * 100:.1f}%)"
            )
        print(f"  Total: {total} samples")

    def _normalize_zscore(self) -> None:
        """Apply per-sample z-score normalization."""
        for i in range(len(self.vib_signals)):
            v = self.vib_signals[i]
            self.vib_signals[i] = (v - v.mean()) / (v.std() + 1e-8)

            a = self.aco_signals[i]
            self.aco_signals[i] = (a - a.mean()) / (a.std() + 1e-8)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "vib_time": torch.tensor(self.vib_signals[idx], dtype=torch.float32).unsqueeze(0),
            "aco_time": torch.tensor(self.aco_signals[idx], dtype=torch.float32).unsqueeze(0),
            "vib_spec": torch.tensor(self.vib_specs[idx], dtype=torch.float32),
            "aco_spec": torch.tensor(self.aco_specs[idx], dtype=torch.float32),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ------------------------------------------------------------------------------
# Stratified split + FFT features for graph construction
# ------------------------------------------------------------------------------
def split_dataset(
    dataset: UOREDDataset,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stratified random split into train / validation / test indices.

    Returns
    -------
    train_idx, val_idx, test_idx : np.ndarray
    """
    rng = np.random.RandomState(seed)
    train_idx, val_idx, test_idx = [], [], []

    for c in range(NUM_CLASSES):
        class_idx = np.where(dataset.labels == c)[0]
        rng.shuffle(class_idx)

        n_c = len(class_idx)
        n_train = max(1, int(n_c * train_ratio))
        n_val = max(1, int(n_c * val_ratio))
        # Ensure train+val does not exhaust the class
        if n_train + n_val >= n_c:
            n_val = max(1, n_c - n_train - 1)

        train_idx.extend(class_idx[:n_train])
        val_idx.extend(class_idx[n_train : n_train + n_val])
        test_idx.extend(class_idx[n_train + n_val :])

    return np.array(train_idx), np.array(val_idx), np.array(test_idx)


def build_fft_features(signals: np.ndarray, remove_dc: bool = True, norm: str = "l2") -> np.ndarray:
    """Build normalized FFT-magnitude features for KNN graph construction.

    Paper: Sec. III-B-1 "Frequency-Domain Node Features", Eq. (4)

    Parameters
    ----------
    signals : np.ndarray, shape (N, L)
        Time-domain signals.
    remove_dc : bool
        If True, discard the DC component (first FFT bin).
    norm : str
        Normalisation mode: "l2" (per-sample L2 norm, default) or
        "zscore" (zero-mean, unit-variance per sample).

    Returns
    -------
    fft_feat : np.ndarray, shape (N, d_f)
        Normalized FFT magnitude spectra.
    """
    fft_mag = np.abs(np.fft.rfft(signals, axis=-1))  # (N, L//2+1)
    if remove_dc:
        fft_mag = fft_mag[:, 1:]  # (N, L//2)

    if norm == "l2":
        norms = np.linalg.norm(fft_mag, axis=1, keepdims=True) + 1e-8
        fft_mag = fft_mag / norms
    elif norm == "zscore":
        mu = fft_mag.mean(axis=1, keepdims=True)
        std = fft_mag.std(axis=1, keepdims=True) + 1e-8
        fft_mag = (fft_mag - mu) / std
    else:
        raise ValueError(f"Unknown norm='{norm}'; choose 'l2' or 'zscore'.")
    return fft_mag.astype(np.float32)


# ------------------------------------------------------------------------------
# Self-test
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python dataset.py <data_dir>")
        sys.exit(1)

    data_dir = sys.argv[1]
    print("=" * 70)
    print("UORED-VAFCLS Dataset Loader — Quick Verification")
    print("=" * 70)

    ds = UOREDDataset(data_dir, seg_len=SEGMENT_LEN, overlap=0.0)
    train_idx, val_idx, test_idx = split_dataset(ds)
    print(f"\nSplit: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    sample = ds[0]
    print("\nSample shapes:")
    for k, v in sample.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {tuple(v.shape)}  dtype={v.dtype}  range=[{v.min():.4f}, {v.max():.4f}]")

    fft_vib = build_fft_features(ds.vib_signals)
    fft_aco = build_fft_features(ds.aco_signals)
    print(f"\nFFT features: vib={fft_vib.shape}, aco={fft_aco.shape}")
    print("\nOK — loader verification passed.")
