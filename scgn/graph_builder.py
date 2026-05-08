
"""
graph_builder.py — Inductive KNN Graph Construction
===================================================

Constructs per-sensor k-NN graphs in the FFT-magnitude feature space.
The construction is strictly *inductive*:
  • Training nodes are bidirectionally connected among themselves.
  • Non-training (val / test) nodes receive edges FROM training neighbours
    but do NOT send edges back — this prevents transductive leakage.

Paper: Sec. III-B "Frequency Graph Construction and Spectral Propagation"
       Eq. (4) — FFT magnitude + z-score standardisation
       Eq. (5) — Cosine-similarity k-NN adjacency
       Eq. (6) — Degree-normalised propagation operator
"""

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize


def build_hetero_graph_inductive(
    vib_fft: np.ndarray,
    aco_fft: np.ndarray,
    train_idx: np.ndarray,
    k: int = 10,
) -> dict:
    """Build per-sensor inductive KNN graphs in frequency-magnitude space.

    Parameters
    ----------
    vib_fft : np.ndarray, shape (N, d_f)
        Normalised FFT magnitude spectra for the vibration modality.
    aco_fft : np.ndarray, shape (N, d_f)
        Normalised FFT magnitude spectra for the acoustic modality.
    train_idx : np.ndarray, shape (n_train,)
        Indices of training samples.
    k : int
        Number of nearest neighbours (default 10, as in the paper).

    Returns
    -------
    hetero_data : dict
        A dict with keys 'vib_node' and 'aco_node'.  Each node store has:
          - adj_norm : torch.Tensor, shape (N, N)
              Normalised propagation operator (Eq. 6 in the paper).
          - edge_index : torch.Tensor, shape (2, E)
              Directed edge list [src; dst].
          - edge_attr : torch.Tensor, shape (E,)
              Edge weights (cosine similarity).
          - x : torch.Tensor, shape (N, 1)
              Dummy node attribute (required by PyG-style data objects).
    """
    # Lightweight HeteroData replacement (no torch_geometric dependency required)
    class _AttrStore(dict):
        def __getattr__(self, name):
            try:
                return self[name]
            except KeyError as exc:
                raise AttributeError(name) from exc

        def __setattr__(self, name, value):
            self[name] = value

        def to(self, device):
            out = _AttrStore()
            for key, value in self.items():
                if torch.is_tensor(value):
                    out[key] = value.to(device)
                elif hasattr(value, "to"):
                    out[key] = value.to(device)
                else:
                    out[key] = value
            return out

    class HeteroData(dict):
        def __getitem__(self, key):
            if key not in self:
                self[key] = _AttrStore()
            return dict.__getitem__(self, key)

        def to(self, device):
            out = HeteroData()
            for key, value in self.items():
                out[key] = value.to(device) if hasattr(value, "to") else value
            return out

    def _build_adj_norm(src, dst, wt, num_nodes):
        adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
        if src:
            src_t = torch.tensor(src, dtype=torch.long)
            dst_t = torch.tensor(dst, dtype=torch.long)
            wt_t = torch.tensor(wt, dtype=torch.float32)
            adj[dst_t, src_t] = wt_t
        adj.fill_diagonal_(1.0)
        deg = adj.sum(dim=1).clamp(min=1e-8)
        deg_inv_sqrt = deg.rsqrt()
        return deg_inv_sqrt[:, None] * adj * deg_inv_sqrt[None, :]

    data = HeteroData()
    N = len(vib_fft)
    train_idx = np.asarray(train_idx)
    if len(train_idx) == 0:
        raise ValueError("train_idx is empty; cannot build graph with zero training nodes.")
    train_set = set(train_idx.tolist())
    non_train_idx = np.array(
        [i for i in range(N) if i not in train_set], dtype=np.int64
    )

    for name, fft in [("vib_node", vib_fft), ("aco_node", aco_fft)]:
        fft_norm = normalize(fft, norm="l2")
        fft_train = fft_norm[train_idx]
        k_train = min(k + 1, len(train_idx))
        if k_train <= 1:
            raise ValueError(f"k_train={k_train} is too small; need at least 2 neighbors (including self).")

        nn_model = NearestNeighbors(
            n_neighbors=k_train, metric="cosine", algorithm="brute", n_jobs=1
        )
        nn_model.fit(fft_train)
        d_tr, i_tr = nn_model.kneighbors(fft_train)

        src, dst, wt = [], [], []

        # Training-training edges (bidirectional)
        for li in range(len(train_idx)):
            gi = int(train_idx[li])
            for j in range(1, d_tr.shape[1]):  # skip self (first neighbour)
                gj = int(train_idx[i_tr[li, j]])
                sim = float(1.0 - d_tr[li, j])
                if sim > 0:
                    src.append(gi)
                    dst.append(gj)
                    wt.append(sim)

        # Training-non_train edges (unidirectional: train -> non_train only)
        if len(non_train_idx) > 0:
            fft_nontr = fft_norm[non_train_idx]
            k_q = min(k, len(train_idx))
            d_te, i_te = nn_model.kneighbors(fft_nontr, n_neighbors=k_q)
            for li in range(len(non_train_idx)):
                gi = int(non_train_idx[li])
                for j in range(d_te.shape[1]):
                    gj = int(train_idx[i_te[li, j]])
                    sim = float(1.0 - d_te[li, j])
                    if sim > 0:
                        src.append(gj)  # train node sends
                        dst.append(gi)  # non-train node receives
                        wt.append(sim)

        data[name, "intra", name].edge_index = torch.tensor([src, dst], dtype=torch.long)
        data[name, "intra", name].edge_attr = torch.tensor(wt, dtype=torch.float32)
        data[name].x = torch.zeros(N, 1)
        data[name].adj_norm = _build_adj_norm(src, dst, wt, N)

    return data


def build_knn_adj_normalized(fft_features, train_idx, k=10):
    """Dense row-normalized KNN adjacency for graph baselines.

    Matrix rows are receiver nodes because the dense layers compute
    ``adj @ X``. Inductive protocol: KNN is fit on train only; non-training
    nodes receive messages from training neighbors, but training nodes never
    receive messages from non-training nodes.
    """
    N = len(fft_features)
    train_idx = np.asarray(train_idx)
    if len(train_idx) == 0:
        raise ValueError("train_idx is empty; cannot build graph with zero training nodes.")
    fft_norm = normalize(fft_features, norm="l2")
    fft_train = fft_norm[train_idx]
    k_train = min(k + 1, len(train_idx))
    if k_train <= 1:
        raise ValueError(f"k_train={k_train} is too small; need at least 2 neighbors (including self).")
    nn_model = NearestNeighbors(
        n_neighbors=k_train, metric="cosine", algorithm="brute", n_jobs=1)
    nn_model.fit(fft_train)

    A = np.zeros((N, N), dtype=np.float32)
    d_tr, i_tr = nn_model.kneighbors(fft_train)
    for li in range(len(train_idx)):
        gi = int(train_idx[li])
        for j in range(1, d_tr.shape[1]):
            gj = int(train_idx[i_tr[li, j]])
            sim = max(0, 1.0 - d_tr[li, j])
            A[gi, gj] = sim
            A[gj, gi] = sim
    non_train_idx = np.array(
        [i for i in range(N) if i not in set(train_idx.tolist())])
    if len(non_train_idx) > 0:
        fft_nontr = fft_norm[non_train_idx]
        k_q = min(k, len(train_idx))
        d_te, i_te = nn_model.kneighbors(fft_nontr, n_neighbors=k_q)
        for li in range(len(non_train_idx)):
            gi = int(non_train_idx[li])
            for j in range(d_te.shape[1]):
                gj = int(train_idx[i_te[li, j]])
                sim = max(0, 1.0 - d_te[li, j])
                A[gi, gj] = sim
    A_tilde = A + np.eye(N, dtype=np.float32)
    row_sum = A_tilde.sum(axis=1, keepdims=True).clip(min=1e-8)
    return torch.tensor(A_tilde / row_sum, dtype=torch.float32)
