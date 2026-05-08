"""
dataset_hust.py — HUST Motor Vibro-Acoustic Dataset Loader
===========================================================

Dataset: HUST motor multi-modal dataset
         Huazhong University of Science and Technology

Sensors : Accelerometer (vibration, Z-axis) + Microphone (acoustic)
Sampling: 25,600 Hz (both modalities)
Per file: variable length, segmented into 4096-point windows

Classes (6):
  0 — Healthy
  1 — BearingFault
  2 — RotorBow
  3 — BrokenRotor
  4 — Misalignment
  5 — Unbalance

Paper: Sec. IV-A, Table 3 (HUST conditions)
"""

import os
import re
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import scipy.signal as signal
import torch
from torch.utils.data import Dataset


# ------------------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------------------
HUST_CLASSES = [
    "Healthy",
    "BearingFault",
    "RotorBow",
    "BrokenRotor",
    "Misalignment",
    "Unbalance",
]
NUM_CLASSES = len(HUST_CLASSES)

FAULT_TOKEN_TO_LABEL = {
    "H": 0,
    "BF": 1,
    "BOW": 2,
    "BROKEN": 3,
    "MISAL": 4,
    "UNBAL": 5,
}

AVAILABLE_SPEEDS = (5, 10, 20, 30)

FS = 25600
SEGMENT_LEN = 4096
TRAIN_RATIO = 0.20
VAL_RATIO = 0.20

_FILENAME_RE = re.compile(
    r"^(?P<fault>BROKEN|MISAL|UNBAL|BOW|BF|H)_(?P<speed>\d+)HZ\.txt$",
    re.IGNORECASE,
)


# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------
def _resolve_raw_dir(data_dir: str) -> str:
    if os.path.basename(os.path.normpath(data_dir)).lower() == "raw data":
        return data_dir
    return os.path.join(data_dir, "Raw data")


def parse_hust_filename(filename: str) -> Tuple[str, int, int]:
    match = _FILENAME_RE.match(filename)
    if match is None:
        raise ValueError(f"Unrecognized HUST filename: {filename}")
    fault = match.group("fault").upper()
    speed = int(match.group("speed"))
    return fault, speed, FAULT_TOKEN_TO_LABEL[fault]


def parse_speeds(speeds: Optional[Iterable[int] | str]) -> Tuple[int, ...]:
    if speeds is None:
        return AVAILABLE_SPEEDS
    if isinstance(speeds, str):
        text = speeds.strip().lower()
        if text in ("", "all", "*"):
            return AVAILABLE_SPEEDS
        values = [int(part.strip()) for part in text.split(",") if part.strip()]
    else:
        values = [int(v) for v in speeds]
    invalid = sorted(set(values) - set(AVAILABLE_SPEEDS))
    if invalid:
        raise ValueError(f"Unsupported HUST speeds: {invalid}; expected {AVAILABLE_SPEEDS}")
    return tuple(values)


def load_txt_signals(filepath: str, vibration_channel: str = "z") -> Tuple[np.ndarray, np.ndarray]:
    """Load one HUST txt file and return vibration + sound signals."""
    data_start = None
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f):
            if line.strip().lower() == "time (seconds) and data channels":
                data_start = line_no + 1
                break
    if data_start is None:
        raise ValueError(f"Could not find data header in {filepath}")

    data = np.loadtxt(filepath, skiprows=data_start, dtype=np.float32)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 5:
        raise ValueError(f"Expected at least 5 columns in {filepath}, got {data.shape[1]}")

    xyz = data[:, 1:4]
    sound = data[:, 4]
    vibration_channel = vibration_channel.lower()
    if vibration_channel == "x":
        vib = xyz[:, 0]
    elif vibration_channel == "y":
        vib = xyz[:, 1]
    elif vibration_channel == "z":
        vib = xyz[:, 2]
    elif vibration_channel == "rms":
        centered = xyz - xyz.mean(axis=0, keepdims=True)
        vib = np.sqrt(np.mean(centered * centered, axis=1))
    else:
        raise ValueError("vibration_channel must be one of: x, y, z, rms")
    return vib.astype(np.float32), sound.astype(np.float32)


def segment_signal(sig: np.ndarray, seg_len: int = SEGMENT_LEN,
                   overlap: float = 0.0) -> np.ndarray:
    if not 0 <= overlap < 1:
        raise ValueError(f"overlap must be in [0, 1), got {overlap}")
    step = max(1, int(round(seg_len * (1 - overlap))))
    if len(sig) < seg_len:
        raise ValueError(f"signal length {len(sig)} is shorter than segment length {seg_len}")
    starts = range(0, len(sig) - seg_len + 1, step)
    return np.stack([sig[start:start + seg_len] for start in starts], axis=0)


def compute_stft_spectrogram(signal_1d: np.ndarray,
                             n_fft: int = 256,
                             hop_length: int = 80,
                             fs: int = FS) -> np.ndarray:
    window = np.hanning(n_fft)
    _, _, zxx = signal.stft(
        signal_1d, fs=fs, window=window,
        nperseg=n_fft, noverlap=n_fft - hop_length)
    mag = np.log1p(np.abs(zxx))
    return mag[np.newaxis, :, :].astype(np.float32)


# ------------------------------------------------------------------------------
# Dataset class
# ------------------------------------------------------------------------------
class HUSTDataset(Dataset):
    """HUST two-modality fault-classification dataset."""

    def __init__(self, data_dir: str,
                 seg_len: int = SEGMENT_LEN,
                 overlap: float = 0.0,
                 max_segs_per_file: int = 0,
                 speeds: Optional[Iterable[int] | str] = None,
                 vibration_channel: str = "z"):
        self.data_dir = data_dir
        self.raw_dir = _resolve_raw_dir(data_dir)
        self.seg_len = int(seg_len)
        self.overlap = float(overlap)
        self.max_segs_per_file = int(max_segs_per_file)
        self.speeds = parse_speeds(speeds)
        self.vibration_channel = vibration_channel.lower()
        self.class_names = list(HUST_CLASSES)
        self.num_classes = NUM_CLASSES

        if not os.path.isdir(self.raw_dir):
            raise FileNotFoundError(f"HUST raw data directory not found: {self.raw_dir}")

        all_vib, all_aco, all_labels = [], [], []

        print(f"[HUSTDataset] Loading data: {self.raw_dir}")
        print(f"  speeds={self.speeds}, vibration_channel={self.vibration_channel}")

        files = sorted(f for f in os.listdir(self.raw_dir) if f.lower().endswith(".txt"))
        class_counts = {label: 0 for label in range(NUM_CLASSES)}
        used_files = 0

        for filename in files:
            fault, speed, label = parse_hust_filename(filename)
            if speed not in self.speeds:
                continue
            filepath = os.path.join(self.raw_dir, filename)
            vib, aco = load_txt_signals(filepath, vibration_channel=self.vibration_channel)
            vib_segs = segment_signal(vib, self.seg_len, self.overlap)
            aco_segs = segment_signal(aco, self.seg_len, self.overlap)
            n_segs = min(len(vib_segs), len(aco_segs))
            if self.max_segs_per_file > 0:
                n_segs = min(n_segs, self.max_segs_per_file)
            vib_segs = vib_segs[:n_segs]
            aco_segs = aco_segs[:n_segs]

            all_vib.append(vib_segs)
            all_aco.append(aco_segs)
            all_labels.extend([label] * n_segs)
            class_counts[label] += n_segs
            used_files += 1

        if not all_vib:
            raise ValueError(
                f"No HUST txt files matched speeds={self.speeds} in {self.raw_dir}")

        self.vib_signals = np.concatenate(all_vib, axis=0).astype(np.float32)
        self.aco_signals = np.concatenate(all_aco, axis=0).astype(np.float32)
        self.labels = np.asarray(all_labels, dtype=np.int64)

        self._normalize_zscore()

        print("  Computing STFT spectrograms...")
        self.vib_specs = np.stack([
            compute_stft_spectrogram(self.vib_signals[i], fs=FS)
            for i in range(len(self))
        ], axis=0)
        self.aco_specs = np.stack([
            compute_stft_spectrogram(self.aco_signals[i], fs=FS)
            for i in range(len(self))
        ], axis=0)

        total = len(self.labels)
        print(f"  files used: {used_files}")
        for label, name in enumerate(HUST_CLASSES):
            cnt = int(class_counts[label])
            print(f"  Class {label} ({name}): {cnt} samples ({cnt / total * 100:.1f}%)")
        print(f"  Total: {total} samples")

    def _normalize_zscore(self):
        for arr in (self.vib_signals, self.aco_signals):
            mu = arr.mean(axis=1, keepdims=True)
            std = arr.std(axis=1, keepdims=True) + 1e-8
            arr[:] = (arr - mu) / std

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
def split_dataset(dataset: HUSTDataset,
                  train_ratio: float = TRAIN_RATIO,
                  val_ratio: float = VAL_RATIO,
                  seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    train_idx, val_idx, test_idx = [], [], []
    labels = np.asarray(dataset.labels)

    for cls in range(dataset.num_classes):
        class_idx = np.where(labels == cls)[0].copy()
        if class_idx.size < 3:
            raise ValueError(f"Class {cls} has only {class_idx.size} samples")
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
