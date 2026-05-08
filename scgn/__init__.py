"""
SCGN — Spectral Coupling Graph Network for Bimodal Information
===============================================================

Core implementation of the proposed method in:
  "Rotating Machinery Fault Diagnosis via a Spectral Coupling Graph
   Network for Bimodal Information" (Information Fusion, Elsevier)

Modules
-------
dataset       : UORED-VAFCLS dataset loader and split utilities
encoders      : CNN1DEncoder (temporal) and CNN2DEncoder (spectrogram)
graph_builder : Inductive KNN graph construction in FFT-magnitude space
model         : SCGN model (Sec. III of the paper)
"""

from .dataset import UOREDDataset, split_dataset, build_fft_features
from .encoders import CNN1DEncoder, CNN2DEncoder
from .graph_builder import build_hetero_graph_inductive, build_knn_adj_normalized
from .model import SCGN

__all__ = [
    "UOREDDataset",
    "split_dataset",
    "build_fft_features",
    "CNN1DEncoder",
    "CNN2DEncoder",
    "build_hetero_graph_inductive",
    "build_knn_adj_normalized",
    "SCGN",
]
