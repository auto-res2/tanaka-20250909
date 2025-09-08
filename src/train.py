"""
train.py – model definitions and training helpers for FlashGAT project
"""
from __future__ import annotations

import time
from typing import List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

# optional / soft-dependencies -------------------------------------------------
try:
    from torch_geometric.nn import GATConv  # Vanilla GAT baseline
except Exception:  # pragma: no cover – keep CPU-only runners alive
    GATConv = None

try:
    from flash_attn.flash_attention import FlashAttention  # noqa: F401
except Exception:
    FlashAttention = None

# -----------------------------------------------------------------------------
__all__ = [
    # utilities
    "EdgeScoreMemory",
    "EXP3Bandit",
    "FlashSparseAttention",
    # model building blocks
    "FlashGATLayer",
    "FlashGAT",
    "VanillaGAT",
    "get_baseline",
    # training helper
    "train_one_epoch",
]

# =============================================================================
# 1.  Helper components
# =============================================================================


class EdgeScoreMemory:
    """Maintain 8-bit rolling importance score for every edge."""

    def __init__(self, num_edges: int, device: torch.device):
        self.scores = torch.full((num_edges,), 128, dtype=torch.uint8, device=device)

    def update(self, idx: torch.Tensor, new_scores: torch.Tensor, decay: float):
        old = self.scores[idx].float() / 255.0
        updated = decay * old + (1.0 - decay) * new_scores
        self.scores[idx] = (updated.clamp_(0, 1) * 255).to(torch.uint8)

    def get(self, idx: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.scores[idx].float() / 255.0


class EXP3Bandit:
    """Node-local EXP3 bandit that allocates Top-B neighbours."""

    def __init__(self, gamma: float = 0.05):
        self.gamma = gamma
        self.weight_dict: Dict[int, torch.Tensor] = {}

    # ---------------------------------------------------------------------
    def _ensure(self, node_id: int, deg: int):
        if node_id not in self.weight_dict:
            self.weight_dict[node_id] = torch.ones(
                deg, dtype=torch.float32, device="cpu"
            )

    # ---------------------------------------------------------------------
    def sample_top_b(self, node_id: int, deg: int, B: int) -> torch.Tensor:
        self._ensure(node_id, deg)
        w = self.weight_dict[node_id]
        p = (1 - self.gamma) * w / w.sum() + self.gamma / deg
        _, idx = torch.topk(p, k=min(B, deg))
        return idx  # local indices

    # ---------------------------------------------------------------------
    def update(self, node_id: int, chosen: torch.Tensor, reward: torch.Tensor):
        w = self.weight_dict[node_id]
        p = w / w.sum()
        est_r = reward / p[chosen]
        w[chosen] *= torch.exp(self.gamma * est_r / len(w))


# =============================================================================
# 2.  Flash-sparse attention wrapper
# =============================================================================


class FlashSparseAttention(nn.Module):
    """Block-sparse FlashAttention thin wrapper (requires flash-attn)."""

    def __init__(self, embed_dim: int, num_heads: int, B: int, causal: bool):
        super().__init__()
        assert (
            FlashAttention is not None
        ), "flash_attn is required – pip install flash-attn"
        self.flash = FlashAttention()
        self.flash.dropout_p = 0.0
        self.flash.causal = causal
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.B = B

    # ------------------------------------------------------------------
    def forward(
        self, qkv: torch.Tensor, cu_seqlens: torch.Tensor, max_seqlen: int
    ) -> torch.Tensor:  # noqa: D401
        # qkv: (blocks, B, 3, H, D_h)
        blocks, blk, *_ = qkv.shape
        qkv_r = qkv.reshape(blocks * blk, 3, self.num_heads, self.embed_dim // self.num_heads)
        out = self.flash(qkv_r, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
        return out.view(blocks, blk, self.embed_dim)


# =============================================================================
# 3.  FlashGAT model
# =============================================================================


class FlashGATLayer(nn.Module):
    def __init__(self, dim: int, heads: int, B: int, causal: bool):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.B = B
        self.heads = heads
        self.dim = dim
        self.attn = FlashSparseAttention(dim, heads, B, causal)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        score_mem: EdgeScoreMemory,
        bandit: EXP3Bandit,
    ) -> torch.Tensor:
        src, dst = edge_index  # E
        q = self.q_proj(x)
        k = self.k_proj(x)
        preview = (q[dst] * k[src]).sum(-1).sigmoid()  # importance [E]
        score_mem.update(torch.arange(src.size(0), device=x.device), preview, decay=0.9)

        # Top-B neighbour sampling
        packs: List[torch.Tensor] = []
        for n in torch.unique(dst):
            idx_e = (dst == n).nonzero(as_tuple=False).flatten()
            local_top = bandit.sample_top_b(int(n), idx_e.numel(), self.B)
            packs.append(idx_e[local_top])
        chosen_edges = torch.cat(packs)

        padded = torch.nn.functional.pad(
            chosen_edges, (0, (-chosen_edges.numel()) % self.B), value=chosen_edges[0]
        )
        blocks = padded.view(-1, self.B)

        qkv = torch.stack(
            [
                q[dst[blocks]],
                k[src[blocks]],
                self.v_proj(x)[src[blocks]],
            ],
            dim=2,
        )  # (blocks, B, 3, D)
        cu = torch.arange(0, (blocks.size(0) + 1) * self.B, step=self.B, device=x.device)
        attn_out = self.attn(qkv, cu_seqlens=cu.to(torch.int32), max_seqlen=self.B)
        out_nodes = torch.zeros_like(x)
        out_nodes.index_add_(0, dst[blocks].view(-1), attn_out.view(-1, self.dim))
        return self.out_proj(out_nodes)


class FlashGAT(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hid_dim: int,
        num_layers: int,
        heads: int,
        B: int,
        causal: bool,
        num_classes: int,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        dims = [in_dim] + [hid_dim] * num_layers
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            self.layers.append(FlashGATLayer(d_out, heads, B, causal))
        self.norm = nn.LayerNorm(hid_dim)
        self.classifier = nn.Linear(hid_dim, num_classes)
        self.score_mem: EdgeScoreMemory | None = None
        self.bandit = EXP3Bandit()

    # ------------------------------------------------------------------
    def set_edge_memory(self, num_edges: int, device: torch.device):
        self.score_mem = EdgeScoreMemory(num_edges, device)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):  # noqa: D401
        assert self.score_mem is not None, "set_edge_memory() must be called first"
        for layer in self.layers:
            x = x + layer(x, edge_index, self.score_mem, self.bandit)
            x = F.elu(x)
        x = self.norm(x)
        return self.classifier(x)


# =============================================================================
# 4.  Baseline (vanilla) GAT
# =============================================================================


class VanillaGAT(nn.Module):
    def __init__(
        self, in_dim: int, hid: int, heads: int, num_layers: int, num_classes: int
    ):
        super().__init__()
        if GATConv is None:
            raise RuntimeError("torch_geometric not available – VanillaGAT unsupported")
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(in_dim, hid, heads=heads, dropout=0.6))
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hid * heads, hid, heads=heads, dropout=0.6))
        self.convs.append(
            GATConv(hid * heads, num_classes, heads=1, concat=False, dropout=0.6)
        )

    # ------------------------------------------------------------------
    def forward(self, x, edge_index):  # noqa: D401
        for conv in self.convs[:-1]:
            x = F.elu(conv(x, edge_index))
        return self.convs[-1](x, edge_index)


# ----------------------------------------------------------------------------

def get_baseline(name: str, **kw):
    if name == "vanilla":
        return VanillaGAT(**kw)

    class _Stub(nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = nn.Parameter(torch.zeros(1))

        def forward(self, x, edge_index):  # noqa: D401
            num_classes = kw.get("num_classes", 2)
            return torch.zeros(x.size(0), num_classes, device=x.device)

    return _Stub()


# =============================================================================
# 5.  Training helper
# =============================================================================


def train_one_epoch(model: nn.Module, data, optimizer, scaler, device, mp: bool):
    """Single-epoch optimiser step with AMP support."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    with torch.cuda.amp.autocast(enabled=mp):
        out = model(data.x.to(device), data.edge_index.to(device))
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask].to(device))
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    return loss.item()
