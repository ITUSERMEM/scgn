"""
dataset_pu.py — Paderborn University (PU) Bearing Dataset Loader
================================================================

Dataset: Paderborn University bearing dataset (vibration + current)
         Chair of Design and Drive Technology, Paderborn University

Sensors : Accelerometer (vibration) + Motor current
Sampling: 64 kHz raw → segmented at 4096-point windows → decimated to 16 kHz / 1024 points
Per class: 250 samples of 1024-point segments

Classes (9):
  0 — K001  (Healthy)
  1 — KA01  (Artificial OR, small)
  2 — KA03  (Artificial OR, medium)
  3 — KA05  (Artificial OR, large)
  4 — KA07  (Real OR, small)
  5 — KA08  (Real OR, large)
  6 — KI01  (Artificial IR, small)
  7 — KI05  (Real IR, small)
  8 — KI07  (Real IR, large)

Paper: Sec. IV-A, Table 1 (PU conditions)

Notes
-----
- Segment FIRST at 64 kHz (4096-point), THEN decimate each segment to 1024-point.
  This prevents filtfilt zero-phase temporal leakage across segment boundaries.
- Vibration is per-sample z-scored inside the loader.
- Current is **not** normalized here; global z-score (train-set μ/σ) should be
  applied externally (see train_pu.py) to match the paper protocol.
"""

import hashlib
import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import scipy.io as sio
import scipy.signal as signal
import torch
from torch.utils.data import Dataset


# ------------------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------------------
PU_CLASSES = ["K001", "KA01", "KA03", "KA05", "KA07", "KA08", "KI01", "KI05", "KI07"]
NUM_CLASSES = len(PU_CLASSES)
SAMPLES_PER_CLASS = 250
SIGNAL_LENGTH = 4096
RAW_FS = 64000
FS = RAW_FS // 4
TRAIN_RATIO = 0.20
VAL_RATIO = 0.20


# ------------------------------------------------------------------------------
# Signal preprocessing
# ------------------------------------------------------------------------------
def remove_fundamental_frequency(x: np.ndarray, fs: int = RAW_FS,
                                 f0: float = 50.0,
                                 num_harmonics: int = 5,
                                 q: float = 30.0) -> np.ndarray:
    """Notch-filter 50 Hz and its harmonics from current signal."""
    filtered = np.asarray(x, dtype=np.float64).copy()
    for harmonic in range(1, num_harmonics + 1):
        freq = f0 * harmonic
        if freq >= fs / 2:
            break
        b, a = signal.iirnotch(freq, q, fs)
        filtered = signal.filtfilt(b, a, filtered)
    return filtered.astype(np.float32)


def extract_signals_from_mat(path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    """Extract vibration (channel 6) and current (channel 1) from a .mat file."""
    data = sio.loadmat(path)
    data_key = next((k for k in data if not k.startswith("__")), None)
    if data_key is None:
        raise ValueError(f"No MATLAB payload in {path}")
    record = data[data_key][0, 0]
    y_data = record["Y"]
    vib = y_data[0, 6]["Data"][0, :].flatten()
    cur = y_data[0, 1]["Data"][0, :].flatten()
    return vib.astype(np.float32), cur.astype(np.float32)


def segment_signal(x: np.ndarray, segment_length: int = SIGNAL_LENGTH,
                   num_segments: int = SAMPLES_PER_CLASS) -> np.ndarray:
    """Slice a long 1-D signal into fixed-length segments with uniform spacing."""
    total = int(len(x))
    if total < segment_length:
        raise ValueError(f"signal length {total} < {segment_length}")
    if num_segments <= 1:
        starts = [0]
    else:
        step = max(1, (total - segment_length) // (num_segments - 1))
        starts = [min(i * step, total - segment_length) for i in range(num_segments)]
    return np.stack([x[s:s + segment_length] for s in starts], axis=0)


def compute_stft_spectrogram(signal_1d: np.ndarray,
                             n_fft: int = 256,
                             hop_length: int = 16,
                             fs: int = FS) -> np.ndarray:
    """Compute a log-compressed STFT magnitude spectrogram."""
    window = np.hanning(n_fft)
    _, _, zxx = signal.stft(
        signal_1d, fs=fs, window=window,
        nperseg=n_fft, noverlap=n_fft - hop_length)
    mag = np.log1p(np.abs(zxx))
    return mag[np.newaxis, :, :].astype(np.float32)


# ------------------------------------------------------------------------------
# Cache
# ------------------------------------------------------------------------------
def _cache_path(data_root: str, cache_dir: str = ".dataset_cache") -> str:
    os.makedirs(cache_dir, exist_ok=True)
    digest = hashlib.md5(os.path.abspath(data_root).encode("utf-8")).hexdigest()[:12]
    return os.path.join(cache_dir, f"pu_{digest}.npz")


# ------------------------------------------------------------------------------
# Dataset class
# ------------------------------------------------------------------------------
class PUDataset(Dataset):
    """PU vibration-current dataset in the shared two-modality task interface."""

    def __init__(self, data_root: str,
                 samples_per_class: int = SAMPLES_PER_CLASS,
                 use_cache: bool = True):
        self.data_dir = data_root
        self.data_root = Path(data_root)
        self.samples_per_class = int(samples_per_class)
        self.class_names = list(PU_CLASSES)
        self.num_classes = NUM_CLASSES
        self.seg_len = SIGNAL_LENGTH
        self.fs_hz = FS

        if not self.data_root.is_dir():
            raise FileNotFoundError(f"PU data root not found: {self.data_root}")

        cache = _cache_path(data_root)
        if use_cache and os.path.isfile(cache):
            payload = np.load(cache, allow_pickle=True)
            self.vib_signals = payload["vib_signals"].astype(np.float32)
            self.aco_signals = payload["aco_signals"].astype(np.float32)
            self.labels = payload["labels"].astype(np.int64)
            print(f"[PUDataset] Loaded cache: {cache}")
        else:
            self.vib_signals, self.aco_signals, self.labels = self._load_raw()
            if use_cache:
                np.savez_compressed(
                    cache,
                    vib_signals=self.vib_signals,
                    aco_signals=self.aco_signals,
                    labels=self.labels,
                )
                print(f"[PUDataset] Saved cache: {cache}")

        # Per-sample z-score for vibration (preserves relative shape, removes load differences)
        self._normalize_vib_zscore()

        # Current is left unnormalised here — global z-score (train μ/σ) is applied
        # externally in train_pu.py to match the paper protocol.

        print("[PUDataset] Computing STFT spectrograms...")
        self.vib_specs = np.stack([
            compute_stft_spectrogram(self.vib_signals[i])
            for i in range(len(self))
        ], axis=0)
        self.aco_specs = np.stack([
            compute_stft_spectrogram(self.aco_signals[i])
            for i in range(len(self))
        ], axis=0)

        total = len(self.labels)
        for label, name in enumerate(self.class_names):
            cnt = int((self.labels == label).sum())
            print(f"  Class {label} ({name}): {cnt} samples ({cnt / total * 100:.1f}%)")
        print(f"  Total: {total} samples")

    def _load_raw(self):
        """Load raw signals: segment at 64 kHz first, then decimate each segment."""
        all_vib, all_cur, labels = [], [], []
        print(f"[PUDataset] Loading data: {self.data_root}")

        for label, class_name in enumerate(PU_CLASSES):
            class_dir = self.data_root / class_name
            if not class_dir.is_dir():
                raise FileNotFoundError(f"Class directory not found: {class_dir}")
            mat_files = sorted(class_dir.glob("*.mat"))
            if not mat_files:
                raise FileNotFoundError(f"No .mat files found in {class_dir}")

            per_file = self.samples_per_class // len(mat_files) + 1
            vib_parts, cur_parts = [], []

            for mat_file in mat_files:
                vib, cur = extract_signals_from_mat(mat_file)
                cur = remove_fundamental_frequency(cur)

                # Segment at 4x length (4096 @ 64kHz), then decimate each segment
                vib_raw = segment_signal(vib, SIGNAL_LENGTH * 4, num_segments=per_file)
                cur_raw = segment_signal(cur, SIGNAL_LENGTH * 4, num_segments=per_file)

                vib_dec_list, cur_dec_list = [], []
                for v_seg, c_seg in zip(vib_raw, cur_raw):
                    # Causal decimation (zero_phase=False) to prevent temporal leakage
                    v_dec = signal.decimate(v_seg, 4, ftype="iir", zero_phase=False)
                    c_dec = signal.decimate(c_seg, 4, ftype="iir", zero_phase=False)
                    # Pad / truncate to exact length (4096 @ 16kHz = 256ms)
                    if len(v_dec) < SIGNAL_LENGTH:
                        v_dec = np.pad(v_dec, (0, SIGNAL_LENGTH - len(v_dec)), mode="edge")
                    elif len(v_dec) > SIGNAL_LENGTH:
                        v_dec = v_dec[:SIGNAL_LENGTH]
                    if len(c_dec) < SIGNAL_LENGTH:
                        c_dec = np.pad(c_dec, (0, SIGNAL_LENGTH - len(c_dec)), mode="edge")
                    elif len(c_dec) > SIGNAL_LENGTH:
                        c_dec = c_dec[:SIGNAL_LENGTH]
                    vib_dec_list.append(v_dec)
                    cur_dec_list.append(c_dec)

                vib_parts.append(np.stack(vib_dec_list, axis=0))
                cur_parts.append(np.stack(cur_dec_list, axis=0))

            vib_segments = np.concatenate(vib_parts, axis=0)[:self.samples_per_class]
            cur_segments = np.concatenate(cur_parts, axis=0)[:self.samples_per_class]
            all_vib.append(vib_segments)
            all_cur.append(cur_segments)
            labels.extend([label] * len(vib_segments))
            print(f"  {class_name}: files={len(mat_files)} -> {len(vib_segments)} samples")

        return (
            np.concatenate(all_vib, axis=0).astype(np.float32),
            np.concatenate(all_cur, axis=0).astype(np.float32),
            np.asarray(labels, dtype=np.int64),
        )

    def _normalize_vib_zscore(self):
        """Apply per-sample z-score normalization to vibration signals."""
        for i in range(len(self.vib_signals)):
            v = self.vib_signals[i]
            self.vib_signals[i] = (v - v.mean()) / (v.std() + 1e-8)

    def __len__(self):
        return int(self.labels.shape[0])

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
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
def split_dataset(dataset: PUDataset,
                  train_ratio: float = TRAIN_RATIO,
                  val_ratio: float = VAL_RATIO,
                  seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    train_idx, val_idx, test_idx = [], [], []
    labels = np.asarray(dataset.labels)

    for cls in range(dataset.num_classes):
        class_idx = np.where(labels == cls)[0].copy()
        rng.shuffle(class_idx)
        n_cls = class_idx.size
        n_train = max(1, int(round(n_cls * train_ratio)))
        n_train = min(n_train, n_cls - 2)
        n_val = max(1, int(round(n_cls * val_ratio)))
        n_val = min(n_val, n_cls - n_train - 1)
        train_idx.extend(class_idx[:n_train].tolist())
        val_idx.extend(class_idx[n_train:n_train + n_val].tolist())
        test_idx.extend(class_idx[n_train + n_val:].tolist())

    return (np.asarray(train_idx, dtype=np.int64),
            np.asarray(val_idx, dtype=np.int64),
            np.asarray(test_idx, dtype=np.int64))


def build_fft_features(signals: np.ndarray, remove_dc: bool = True, norm: str = "l2") -> np.ndarray:
    """Build normalized FFT-magnitude features for KNN graph construction."""
    fft_mag = np.abs(np.fft.rfft(signals, axis=-1))
    if remove_dc:
        fft_mag = fft_mag[:, 1:]

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


def global_zscore_current(signals: np.ndarray, train_idx: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Global Z-score for current: compute μ/σ on train set, apply to all."""
    train_signals = signals[train_idx]
    mu = train_signals.mean()
    sigma = train_signals.std() + eps
    return (signals - mu) / sigma
