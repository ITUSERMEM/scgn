"""
model.py — Spectral Coupling Graph Network (SCGN)
=================================================

Core model implementation aligned with Sec. III of the paper:
  "Rotating Machinery Fault Diagnosis via a Spectral Coupling
   Network for Bimodal Information" (Information Fusion)

Architecture (three sequential stages)
--------------------------------------
1. Frequency-domain graph construction & spectral propagation  (Sec. III-B)
2. Dual-domain local encoding                                   (Sec. III-C)
3. Cross-modal joint discrimination                            (Sec. III-D / III-E)

Modules
-------
GCNLayer            : Single residual GCN layer                  (Eq. 7)
GCNStack            : Stacked GCN layers
TFProjector         : Temporal–Graph interaction projector       (Eq. 10–11)
SymmetricAggregator : Intra-sensor aggregation                   (Eq. 12–13)
ConcatFusion        : Cross-modal joint decision head            (Eq. 14)
SCGN                : End-to-end top-level model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders import CNN1DEncoder, CNN2DEncoder


# ==============================================================================
# Sec. III-B-3  Spectral Graph Convolution
# ==============================================================================
class GCNLayer(nn.Module):
    """Single residual GCN layer with deterministic dense adjacency.

    Paper: Sec. III-B-3, Eq. (7)

    Forward implements:
        h = norm( proj(x) + relu( A_norm @ x @ W_gcn + b_gcn ) )
    where ``proj`` is a linear projection (or Identity when in_dim == out_dim).
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.norm = nn.LayerNorm(out_dim)
        self.proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        nn.init.xavier_uniform_(self.lin.weight)

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor, shape (N, in_dim)
            Node feature matrix.
        adj_norm : torch.Tensor, shape (N, N)
            Normalised propagation operator.

        Returns
        -------
        torch.Tensor, shape (N, out_dim)
        """
        h = torch.matmul(adj_norm, self.lin(x)) + self.bias
        h = F.relu(h)
        return self.norm(self.proj(x) + h)


class GCNStack(nn.Module):
    """Stack of GCN layers.

    Paper: Sec. III-B-3.  The default configuration in the paper uses a
    single layer (num_layers=1).
    """

    def __init__(self, fft_dim: int, hidden_dim: int = 64, num_layers: int = 1):
        super().__init__()
        self.layers = nn.ModuleList([GCNLayer(fft_dim, hidden_dim)])
        for _ in range(num_layers - 1):
            self.layers.append(GCNLayer(hidden_dim, hidden_dim))

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, adj_norm)
        return x


# ==============================================================================
# Sec. III-D-1  Temporal–Graph Interaction Projection
# ==============================================================================
class TFProjector(nn.Module):
    """Interaction projector that deeply couples temporal (CNN) and graph
    (GCN) embeddings within a single sensor branch.

    Paper: Sec. III-D-1, Eq. (10)–(11)

    Forward:
        h_int = W_int [z_t || z_gcn] + b_int
        u_int = LayerNorm( Dropout( ReLU( h_int ) ) )
    where || denotes concatenation.
    """

    def __init__(self, d: int = 64, dropout: float = 0.3):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d * 2, d),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d),
        )

    def forward(self, h_cnn: torch.Tensor, h_gcn: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        h_cnn : torch.Tensor, shape (N, d)
            Time-domain CNN embedding  (z_t in the paper).
        h_gcn : torch.Tensor, shape (N, d)
            Frequency-domain GCN embedding  (z_gcn in the paper).

        Returns
        -------
        torch.Tensor, shape (N, d)
            Coupled representation  (u_int in the paper).
        """
        return self.proj(torch.cat([h_cnn, h_gcn], dim=-1))


# ==============================================================================
# Sec. III-D-2  Sensor-Level Aggregation
# ==============================================================================
class SymmetricAggregator(nn.Module):
    """Intra-sensor aggregation with per-sensor independent parameters.

    Paper: Sec. III-D-2, Eq. (12)–(13)

    Forward:
        h_agg = W_agg [u_int || z_tf] + b_agg
        u     = LayerNorm( Dropout( ReLU( h_agg ) ) )

    The two sensor branches (vibration / acoustic) each own an independent
    aggregator so that heterogeneous feature-space distributions can be
    adaptively weighted.
    """

    def __init__(self, d: int = 64, dropout: float = 0.3):
        super().__init__()
        self.vib_fuse = nn.Sequential(
            nn.Linear(d * 2, d),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d),
        )
        self.aco_fuse = nn.Sequential(
            nn.Linear(d * 2, d),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d),
        )

    def forward(
        self,
        F_vib_t: torch.Tensor,
        F_aco_t: torch.Tensor,
        F_vib_s: torch.Tensor,
        F_aco_s: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        F_vib_t, F_aco_t : torch.Tensor, shape (N, d)
            Coupled temporal–graph representations (u_int) for vibration
            and acoustic branches.
        F_vib_s, F_aco_s : torch.Tensor, shape (N, d)
            Time-frequency (spectrogram) embeddings (z_tf) for vibration
            and acoustic branches.

        Returns
        -------
        node_vib, node_aco : torch.Tensor, each shape (N, d)
            Final sensor-level representations (u in the paper).
        """
        node_vib = self.vib_fuse(torch.cat([F_vib_t, F_vib_s], dim=-1))
        node_aco = self.aco_fuse(torch.cat([F_aco_t, F_aco_s], dim=-1))
        return node_vib, node_aco


# ==============================================================================
# Sec. III-E  Cross-Modal Joint Decision Head
# ==============================================================================
class ConcatFusion(nn.Module):
    """Lightweight concatenation-based joint decision head.

    Paper: Sec. III-E, Eq. (14)

    Forward:
        p_hat = softmax( W_c [u^{(1)} || u^{(2)}] + b_c )
    """

    def __init__(self, d: int = 64, num_classes: int = 5, dropout: float = 0.0):
        super().__init__()
        self.clf = nn.Linear(d * 2, num_classes)

    def forward(self, hv: torch.Tensor, ha: torch.Tensor):
        """
        Parameters
        ----------
        hv : torch.Tensor, shape (N, d)
            Vibration sensor-level representation.
        ha : torch.Tensor, shape (N, d)
            Acoustic sensor-level representation.

        Returns
        -------
        p : torch.Tensor, shape (N, num_classes)
            Softmax probabilities.
        logits_list : list[torch.Tensor]
            List containing the raw logits twice (for auxiliary-loss symmetry).
        w : torch.Tensor, shape (N, 2)
            Fixed equal fusion weights.
        """
        logits = self.clf(torch.cat([hv, ha], dim=-1))
        p = F.softmax(logits, dim=-1)
        w = torch.full((hv.shape[0], 2), 0.5, device=hv.device, dtype=hv.dtype)
        return p, [logits, logits], w


# ==============================================================================
# Sec. III-A  Overall Architecture — SCGN
# ==============================================================================
class SCGN(nn.Module):
    """Spectral Coupling Graph Network (SCGN).

    Paper: Sec. III-A, Fig. 1

    The model consists of two strictly symmetric sensor branches with
    independent parameters.  Each branch contains:
      1. Frequency-graph construction  (handled externally)
      2. GCN spectral propagation      (GCNStack)
      3. Temporal encoding             (CNN1DEncoder)
      4. Spectro-temporal encoding     (CNN2DEncoder)
      5. Temporal–graph interaction    (TFProjector)
      6. Sensor-level aggregation      (SymmetricAggregator)
    The two sensor-level representations are concatenated and fed into the
    linear decision head (ConcatFusion).

    Parameters
    ----------
    num_classes : int
        Number of fault classes (default 5 for UORED).
    d : int
        Unified feature dimension (default 64, as in the paper).
    gcn_layers : int
        Number of GCN layers (default 1, as in the paper).
    fft_dim : int
        FFT feature dimension (default 2048 for 4096-point signals).
    mid_dropout : float
        Dropout rate for projection / aggregation layers (default 0.3).
    """

    def __init__(
        self,
        num_classes: int = 5,
        d: int = 64,
        gcn_layers: int = 1,
        fft_dim: int = 2048,
        mid_dropout: float = 0.3,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.d = d

        # --- Frequency-graph branch (Sec. III-B) ---
        self.gcn_vib = GCNStack(fft_dim, d, gcn_layers)
        self.gcn_aco = GCNStack(fft_dim, d, gcn_layers)

        # --- Temporal branch (Sec. III-C) ---
        self.cnn_vib = CNN1DEncoder(d)
        self.cnn_aco = CNN1DEncoder(d)

        # --- Spectro-temporal branch (Sec. III-C) ---
        self.cnn_vib_spec = CNN2DEncoder(d)
        self.cnn_aco_spec = CNN2DEncoder(d)

        # --- Interaction projector (Sec. III-D-1) ---
        self.tf_vib = TFProjector(d, dropout=mid_dropout)
        self.tf_aco = TFProjector(d, dropout=mid_dropout)

        # --- Sensor-level aggregation (Sec. III-D-2) ---
        self.agg = SymmetricAggregator(d, dropout=mid_dropout)

        # --- Joint decision head (Sec. III-E) ---
        self.decision = ConcatFusion(d, num_classes, dropout=mid_dropout)

    # --------------------------------------------------------------------------
    # Public API
    # --------------------------------------------------------------------------
    def forward(
        self,
        vib_time: torch.Tensor,
        aco_time: torch.Tensor,
        vib_spec: torch.Tensor,
        aco_spec: torch.Tensor,
        adj_vib: torch.Tensor,
        adj_aco: torch.Tensor,
        fft_vib_feat: torch.Tensor,
        fft_aco_feat: torch.Tensor,
        sample_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass of the SCGN model.

        Parameters
        ----------
        vib_time : torch.Tensor, shape (N, 1, L)
            Time-domain vibration signals (full dataset).
        aco_time : torch.Tensor, shape (N, 1, L)
            Time-domain acoustic signals (full dataset).
        vib_spec : torch.Tensor, shape (N, 1, F, T)
            Vibration STFT spectrograms (full dataset).
        aco_spec : torch.Tensor, shape (N, 1, F, T)
            Acoustic STFT spectrograms (full dataset).
        adj_vib : torch.Tensor, shape (N, N)
            Normalised dense adjacency for vibration branch.
        adj_aco : torch.Tensor, shape (N, N)
            Normalised dense adjacency for acoustic branch.
        fft_vib_feat : torch.Tensor, shape (N, fft_dim)
            FFT-magnitude node features for vibration branch.
        fft_aco_feat : torch.Tensor, shape (N, fft_dim)
            FFT-magnitude node features for acoustic branch.
        sample_idx : torch.Tensor, optional, shape (n,)
            Global sample indices.  When provided, all inputs and GCN
            embeddings are sliced to this subset.

        Returns
        -------
        logits : torch.Tensor, shape (n, num_classes)
            Class logits (before softmax).
        """
        # 1. Graph features (Sec. III-B-3) — computed on the *full* graph
        h_gcn_v = self.gcn_vib(fft_vib_feat, adj_vib)
        h_gcn_a = self.gcn_aco(fft_aco_feat, adj_aco)

        # Select the relevant rows when a subset is passed
        if sample_idx is not None:
            sample_idx = sample_idx.to(device=h_gcn_v.device, dtype=torch.long)
            vib_time = vib_time[sample_idx]
            aco_time = aco_time[sample_idx]
            vib_spec = vib_spec[sample_idx]
            aco_spec = aco_spec[sample_idx]
            h_gcn_v = h_gcn_v[sample_idx]
            h_gcn_a = h_gcn_a[sample_idx]

        # 2. Temporal CNN features (Sec. III-C)
        h_cnn_v = self.cnn_vib(vib_time)
        h_cnn_a = self.cnn_aco(aco_time)

        # 3. Temporal–Graph interaction (Sec. III-D-1)
        h_v = self.tf_vib(h_cnn_v, h_gcn_v)
        h_a = self.tf_aco(h_cnn_a, h_gcn_a)

        # 4. Spectro-temporal CNN features (Sec. III-C)
        h_spec_v = self.cnn_vib_spec(vib_spec)
        h_spec_a = self.cnn_aco_spec(aco_spec)

        # 5. Sensor-level aggregation (Sec. III-D-2)
        node_v, node_a = self.agg(h_v, h_a, h_spec_v, h_spec_a)

        # 6. Cross-modal joint decision (Sec. III-E)
        _p, logits_list, _w = self.decision(node_v, node_a)
        return logits_list[0]
