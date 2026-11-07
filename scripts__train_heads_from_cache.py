"""Train and evaluate R1-R6 + Oracle-EDES heads on cached frame embeddings.

Every head consumes an (N, T, 512) float32 cache of L2-normalized EchoCLIP frame
embeddings. The visual tower is never updated, so all training here is on
precomputed features. Cache files come from ``scripts/extract_frames_embeddings.py``.

Rows
----
R0          zero-shot, official stride 0:min(40,T):2 (separate cache)
R1          zero-shot, top-frac median rule, 32 uniform frames
R1U16       zero-shot, top-frac median rule, 16 uniform frames
R2          mean pooling + linear regression
R3          mean pooling + 2-layer MLP (256, dropout 0.2)
R4          2-layer Transformer encoder + CLS pooling + linear regression
R4_MEAN     R4 with mean pooling over encoder outputs (pooling ablation)
R5          R4 + soft-label multi-positive contrastive pretraining, then regression
R6          R5 + VAL-only calibration (Platt, temperature, split conformal)
R5_EDES     R5 with the annotated ED/ES frame pair (label-assisted)
ORACLE_EDES zero-shot on the annotated ED/ES frame pair (label-assisted)

Each run writes one JSON with config, seed, sizes, command, git commit, data
root, weight hash and the full metric block.

Usage
-----
python scripts/train_heads_from_cache.py --experiment R2 --seeds 42 43 44
python scripts/train_heads_from_cache.py --experiment all --seeds 42
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from _paths import cache_dir, results_dir, logs_dir

CACHE = cache_dir()
RESULTS = results_dir()
LOGS = logs_dir()
WEIGHTS = ROOT / "models" / "echo-clip" / "open_clip_pytorch_model.bin"
DATA_ROOT = Path(r"E:\Datasets\EchoNet-Dynamic")

EXPERIMENT_IDS = (
    "R0",
    "R1",
    "R1U16",
    "R2",
    "R3",
    "R4",
    "R4_MEAN",
    "R5",
    "R5_EDES",
    "R6",
    "ORACLE_EDES",
)


# --------------------------------------------------------------------------
# small utilities
# --------------------------------------------------------------------------
def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    with open(LOGS / "train_heads.log", "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True
        ).strip()
    except Exception:
        return "unknown"


_SHA: Dict[str, str] = {}


def sha256_file(path: Path) -> str:
    key = str(path)
    if key in _SHA:
        return _SHA[key]
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    _SHA[key] = h.hexdigest()
    return _SHA[key]


def command_line() -> str:
    return "python " + " ".join(sys.argv)


def _cuda() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def _torch_version() -> str:
    try:
        import torch

        return torch.__version__
    except Exception:
        return "unavailable"


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
@dataclass
class Split:
    ids: np.ndarray
    emb: np.ndarray
    ef: np.ndarray
    edv: np.ndarray
    esv: np.ndarray
    ed_frame: np.ndarray
    es_frame: np.ndarray
    path: str

    def __len__(self) -> int:
        return int(self.emb.shape[0])

    @property
    def n_frames(self) -> int:
        return int(self.emb.shape[1])

    def subset(self, idx: np.ndarray) -> "Split":
        return Split(
            ids=self.ids[idx],
            emb=self.emb[idx],
            ef=self.ef[idx],
            edv=self.edv[idx],
            esv=self.esv[idx],
            ed_frame=self.ed_frame[idx],
            es_frame=self.es_frame[idx],
            path=self.path,
        )


_CACHE_MEM: Dict[str, Split] = {}


def load_split(name: str, cache_tag: str = "32_bgr") -> Split:
    key = f"{name}:{cache_tag}"
    if key in _CACHE_MEM:
        return _CACHE_MEM[key]
    path = CACHE / f"{name}_{cache_tag}.npz"
    if not path.exists():
        raise FileNotFoundError(f"missing cache {path}")
    z = np.load(path, allow_pickle=False)
    split = Split(
        ids=z["ids"],
        emb=z["emb"].astype(np.float32),
        ef=z["ef"].astype(np.float64),
        edv=z["edv"].astype(np.float64),
        esv=z["esv"].astype(np.float64),
        ed_frame=z["ed_frame"].astype(np.int64),
        es_frame=z["es_frame"].astype(np.int64),
        path=str(path),
    )
    _CACHE_MEM[key] = split
    return split


def take_frames(emb: np.ndarray, k: int) -> np.ndarray:
    """Evenly subsample the cached T axis (32 -> 16 -> 8) without re-encoding."""
    t = emb.shape[1]
    if k >= t:
        return emb
    idx = np.linspace(0, t - 1, k).round().astype(int)
    return emb[:, idx, :]


_FILELIST = None
_NFRAMES: Dict[str, int] = {}


def num_frames_of(stem: str) -> int:
    """NumberOfFrames from FileList.csv for a video stem."""
    global _FILELIST
    if stem in _NFRAMES:
        return _NFRAMES[stem]
    if _FILELIST is None:
        import pandas as pd

        fl = pd.read_csv(DATA_ROOT / "FileList.csv")
        fl["stem"] = fl["FileName"].astype(str).str.replace(r"\.avi$", "", regex=True)
        _FILELIST = dict(zip(fl["stem"], fl["NumberOfFrames"]))
    n = int(_FILELIST.get(stem, 0))
    _NFRAMES[stem] = n
    return n


def edes_frames(split: Split) -> np.ndarray:
    """Snapshot the annotated ED and ES frames from the 32-frame uniform cache.

    Slot j of a T-frame uniform cache sits at round(j*(N-1)/(T-1)). We map the
    annotated ED/ES index to the nearest slot. Videos without a tracing row fall
    back to the first and last slot.
    """
    emb = split.emb
    t = emb.shape[1]
    out = np.zeros((len(split), 2, emb.shape[-1]), dtype=np.float32)
    for i in range(len(split)):
        stem = str(split.ids[i])
        n = num_frames_of(stem)
        ed = int(split.ed_frame[i])
        es = int(split.es_frame[i])
        if n <= 1 or ed < 0 or es < 0:
            a, b = 0, t - 1
        else:
            a = int(round(ed / (n - 1) * (t - 1)))
            b = int(round(es / (n - 1) * (t - 1)))
        out[i, 0] = emb[i, int(np.clip(a, 0, t - 1))]
        out[i, 1] = emb[i, int(np.clip(b, 0, t - 1))]
    return out


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def _rankdata(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def _auc(y: np.ndarray, s: np.ndarray) -> float:
    y = np.asarray(y).astype(bool)
    s = np.asarray(s, dtype=np.float64)
    n_pos = int(y.sum())
    n_neg = int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _rankdata(s)
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _auprc(y: np.ndarray, s: np.ndarray) -> float:
    y = np.asarray(y).astype(bool)
    s = np.asarray(s, dtype=np.float64)
    if y.sum() == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    ys = y[order]
    tp = np.cumsum(ys).astype(np.float64)
    fp = np.cumsum(~ys).astype(np.float64)
    precision = tp / np.maximum(tp + fp, 1.0)
    recall = tp / float(ys.sum())
    # average precision = sum over thresholds of precision * delta recall
    drecall = np.diff(np.concatenate([[0.0], recall]))
    return float(np.sum(precision * drecall))


def _ece(y: np.ndarray, p: np.ndarray, bins: int = 15) -> float:
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    n = len(p)
    for b in range(bins):
        if b < bins - 1:
            m = (p >= edges[b]) & (p < edges[b + 1])
        else:
            m = (p >= edges[b]) & (p <= edges[b + 1])
        if m.sum() == 0:
            continue
        total += (m.sum() / n) * abs(y[m].mean() - p[m].mean())
    return float(total)


def _brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((np.asarray(p) - np.asarray(y, dtype=np.float64)) ** 2))


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3:
        return float("nan")
    return float(np.corrcoef(_rankdata(a), _rankdata(b))[0, 1])


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


def regression_metrics(y: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    y = np.asarray(y, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    resid = pred - y
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    out = {
        "mae": float(np.mean(np.abs(resid))),
        "rmse": float(math.sqrt(np.mean(resid ** 2))),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "bias": float(np.mean(resid)),
        "pearson": float(np.corrcoef(y, pred)[0, 1]) if len(y) > 2 else float("nan"),
        "spearman": _spearman(y, pred),
        "n": int(len(y)),
        "y_mean": float(y.mean()),
        "y_sd": float(y.std(ddof=1)),
        "pred_mean": float(pred.mean()),
        "pred_sd": float(pred.std(ddof=1)),
    }
    for thr in (50, 40, 30):
        cls = (y < thr).astype(np.int64)
        risk = -pred  # lower predicted EF -> higher risk
        out[f"auc_ef_lt{thr}"] = _auc(cls, risk)
        out[f"auprc_ef_lt{thr}"] = _auprc(cls, risk)
        out[f"n_pos_ef_lt{thr}"] = int(cls.sum())
    return out


def classification_metrics(y_bin: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    return {
        "auc": _auc(y_bin, p),
        "auprc": _auprc(y_bin, p),
        "ece_15": _ece(y_bin, p, 15),
        "brier": _brier(y_bin, p),
        "n": int(len(p)),
        "n_pos": int(np.asarray(y_bin).sum()),
        "p_mean": float(np.mean(p)),
    }


def bootstrap_ci(y, pred, metric, n_boot=1000, seed=42) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    n = len(y)
    vals: List[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb, pb = y[idx], pred[idx]
        if metric == "mae":
            vals.append(float(np.mean(np.abs(pb - yb))))
        else:
            thr = 50 if metric == "auc50" else 40
            cls = (yb < thr).astype(int)
            if cls.sum() == 0 or cls.sum() == len(cls):
                continue
            vals.append(_auc(cls, -pb))
    arr = np.array(vals)
    return {
        "mean": float(arr.mean()),
        "sd": float(arr.std(ddof=1)),
        "lo": float(np.percentile(arr, 2.5)),
        "hi": float(np.percentile(arr, 97.5)),
        "n_boot": int(len(arr)),
    }


# --------------------------------------------------------------------------
# zero-shot on cached embeddings
# --------------------------------------------------------------------------
def load_text_prompts() -> Tuple[np.ndarray, np.ndarray]:
    path = CACHE / "text_ef.npz"
    z = np.load(path, allow_pickle=False)
    return z["emb"].astype(np.float32), z["values"].astype(np.float32)


def zero_shot_predict(emb: np.ndarray, top_frac: float = 0.2) -> np.ndarray:
    """Official top-frac median rule on cached frame embeddings.

    For every frame the 202 candidates (2 templates x EF 0..100) are ranked by
    cosine similarity. Ranked EF values are averaged over frames, then the median
    of the top ``top_frac`` share of that averaged vector is the prediction.
    """
    temb, values = load_text_prompts()
    sim = np.einsum("ntd,cd->ntc", emb, temb, optimize=True)
    order = np.argsort(-sim, axis=-1)
    ranked_vals = values[order]
    avg = ranked_vals.mean(axis=1)
    k = max(1, int(avg.shape[1] * top_frac))
    return np.median(avg[:, :k], axis=1)


# --------------------------------------------------------------------------
# torch heads
# --------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    import random

    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class MLPRegressor:
    """Mean-pooled feature -> 2-layer MLP (R3)."""

    def __init__(self, dim: int = 512, hidden: int = 256, dropout: float = 0.2):
        import torch.nn as nn

        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def to(self, device):
        self.net.to(device)
        return self

    def train(self):
        self.net.train()
        return self

    def eval(self):
        self.net.eval()
        return self

    def __call__(self, x):
        return self.net(x).squeeze(-1)

    def parameters(self):
        return self.net.parameters()

    def state_dict(self):
        return self.net.state_dict()

    def load_state_dict(self, sd):
        self.net.load_state_dict(sd)


class LinearRegressor(MLPRegressor):
    """Mean-pooled feature -> linear regression (R2)."""

    def __init__(self, dim: int = 512, hidden: int = 0, dropout: float = 0.0):
        import torch.nn as nn

        self.net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))


class TemporalRegressor:
    """Temporal Transformer over frame embeddings (R4/R5/R6).

    ``pool='cls'``   prepend a learnable CLS token, read its output (R4/R5/R6)
    ``pool='mean'``  encode the frame sequence, mean-pool the outputs (R4_MEAN)
    """

    def __init__(
        self,
        dim: int = 512,
        n_layers: int = 2,
        n_heads: int = 8,
        max_frames: int = 64,
        dropout: float = 0.1,
        pool: str = "cls",
        proj_dim: int = 128,
    ):
        import torch
        import torch.nn as nn

        pool = pool.lower()
        head_dim = dim

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.pool = pool
                self.cls = nn.Parameter(torch.zeros(1, 1, dim)) if pool == "cls" else None
                if self.cls is not None:
                    nn.init.normal_(self.cls, std=0.02)
                self.pos = nn.Parameter(torch.zeros(1, max_frames + 1, dim))
                nn.init.normal_(self.pos, std=0.01)
                layer = nn.TransformerEncoderLayer(
                    d_model=dim,
                    nhead=n_heads,
                    dim_feedforward=dim * 4,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
                self.ln = nn.LayerNorm(head_dim)
                self.head = nn.Linear(head_dim, 1)
                self.proj = nn.Linear(head_dim, proj_dim)

            def forward(self, x, return_emb: bool = False):
                b, t, _ = x.shape
                if self.pool == "cls":
                    tok = torch.cat([self.cls.expand(b, -1, -1), x], dim=1)
                    tok = tok + self.pos[:, : t + 1]
                    out = self.encoder(tok)
                    z = self.ln(out[:, 0])
                else:
                    out = self.encoder(x + self.pos[:, :t])
                    z = self.ln(out.mean(dim=1))
                pred = self.head(z).squeeze(-1)
                if return_emb:
                    return pred, torch.nn.functional.normalize(self.proj(z), dim=-1)
                return pred

        self.net = Net()
        self.max_frames = max_frames
        self.pool = pool

    def to(self, device):
        self.net.to(device)
        return self

    def train(self):
        self.net.train()
        return self

    def eval(self):
        self.net.eval()
        return self

    def __call__(self, x, return_emb: bool = False):
        return self.net(x, return_emb=return_emb)

    def parameters(self):
        return self.net.parameters()

    def state_dict(self):
        return self.net.state_dict()

    def load_state_dict(self, sd):
        self.net.load_state_dict(sd)


def fit_regression(
    head,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    *,
    epochs: int = 120,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 64,
    device: str = "cpu",
    patience: int = 25,
    seed: int = 42,
    verbose: bool = False,
) -> Tuple[object, Dict]:
    """AdamW + cosine schedule, early stop on VAL MAE, restore best weights."""
    import copy

    import torch

    set_seed(seed)
    head = head.to(device)
    y_mean = float(y_train.mean())
    y_std = float(y_train.std() + 1e-6)
    yt = torch.from_numpy(((y_train - y_mean) / y_std).astype(np.float32))
    yv = torch.from_numpy(((y_val - y_mean) / y_std).astype(np.float32))
    xt = torch.from_numpy(np.ascontiguousarray(x_train, dtype=np.float32))
    xv = torch.from_numpy(np.ascontiguousarray(x_val, dtype=np.float32))

    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    loss_fn = torch.nn.SmoothL1Loss(beta=0.1)
    n = xt.shape[0]
    best = {"mae": float("inf"), "state": None, "epoch": -1}
    rng = np.random.default_rng(seed)
    for ep in range(epochs):
        head.train()
        perm = rng.permutation(n)
        for s in range(0, n, batch_size):
            idx = perm[s : s + batch_size]
            xb = xt[idx].to(device)
            yb = yt[idx].to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(head(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(head.parameters()), 1.0)
            opt.step()
        sched.step()
        head.eval()
        with torch.no_grad():
            pv = head(xv.to(device)).cpu().numpy() * y_std + y_mean
        mae = float(np.mean(np.abs(pv - y_val)))
        if mae < best["mae"] - 1e-5:
            best = {"mae": mae, "state": copy.deepcopy(head.state_dict()), "epoch": ep}
        elif ep - best["epoch"] > patience:
            break
        if verbose and ep % 10 == 0:
            log(f"    ep {ep:3d} val MAE {mae:.3f} (best {best['mae']:.3f})")
    if best["state"] is not None:
        head.load_state_dict(best["state"])
    state = {
        "y_mean": y_mean,
        "y_std": y_std,
        "best_val_mae": best["mae"],
        "best_epoch": best["epoch"],
        "epochs_run": ep + 1 if epochs else 0,
    }
    return head, state


def predict_regression(head, x: np.ndarray, state: Dict, device="cpu", batch=128) -> np.ndarray:
    import torch

    head.eval()
    out = []
    with torch.no_grad():
        for s in range(0, x.shape[0], batch):
            xb = torch.from_numpy(np.ascontiguousarray(x[s : s + batch], dtype=np.float32)).to(device)
            out.append(head(xb).cpu().numpy())
    return (np.concatenate(out) * state["y_std"] + state["y_mean"]).astype(np.float64)


def pretrain_contrastive(
    head,
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    tau: float = 5.0,
    epochs: int = 40,
    lr: float = 1e-3,
    batch_size: int = 64,
    device: str = "cpu",
    seed: int = 42,
    smooth_sigma: float = 5.0,
    verbose: bool = False,
) -> Tuple[object, List[float]]:
    """Soft-label multi-positive contrastive pretraining (R5).

    Positivity weight w_ij = exp(-(EF_i - EF_j)^2 / (2 sigma^2)), normalized
    inside the batch. The loss is soft-target InfoNCE at temperature tau

        L = -sum_j w_ij log softmax_j(z_i . z_j / tau)

    where z are 128-d projections of the pooled temporal representation.
    """
    import torch
    import torch.nn.functional as F

    set_seed(seed)
    head = head.to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    xt = torch.from_numpy(np.ascontiguousarray(x_train, dtype=np.float32))
    yt = torch.from_numpy(np.asarray(y_train, dtype=np.float32))
    n = xt.shape[0]
    rng = np.random.default_rng(seed)
    curve: List[float] = []
    for ep in range(epochs):
        head.train()
        perm = rng.permutation(n)
        tot, cnt = 0.0, 0
        for s in range(0, n, batch_size):
            idx = perm[s : s + batch_size]
            if len(idx) < 8:
                continue
            xb, yb = xt[idx].to(device), yt[idx].to(device)
            _, z = head(xb, return_emb=True)
            w = torch.exp(-((yb[:, None] - yb[None, :]) ** 2) / (2.0 * smooth_sigma ** 2))
            w = w / w.sum(dim=1, keepdim=True)
            logp = F.log_softmax(z @ z.t() / tau, dim=-1)
            loss = -(w * logp).sum(dim=-1).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(head.parameters()), 1.0)
            opt.step()
            tot += float(loss.detach())
            cnt += 1
        sched.step()
        curve.append(tot / max(cnt, 1))
        if verbose and ep % 10 == 0:
            log(f"    contrastive ep {ep:3d} loss {curve[-1]:.4f}")
    return head, curve


# --------------------------------------------------------------------------
# calibration (R6)
# --------------------------------------------------------------------------
def platt_fit(y_bin: np.ndarray, score: np.ndarray) -> Tuple[float, float]:
    """Logistic calibration p = sigmoid(a*s + b), Newton iterations."""
    s = np.asarray(score, dtype=np.float64)
    y = np.asarray(y_bin, dtype=np.float64)
    a, b = 1.0, 0.0
    for _ in range(200):
        p = sigmoid(a * s + b)
        w = p * (1 - p) + 1e-9
        g_a = float(np.sum((p - y) * s))
        g_b = float(np.sum(p - y))
        h_aa = float(np.sum(w * s * s)) + 1e-6
        h_ab = float(np.sum(w * s))
        h_bb = float(np.sum(w)) + 1e-6
        det = h_aa * h_bb - h_ab * h_ab
        if abs(det) < 1e-12:
            break
        da = (h_bb * g_a - h_ab * g_b) / det
        db = (-h_ab * g_a + h_aa * g_b) / det
        # damped step keeps Newton stable on near-separable scores
        step = 1.0
        a_new, b_new = a - step * da, b - step * db
        if not np.isfinite(a_new) or not np.isfinite(b_new):
            break
        a, b = a_new, b_new
        if abs(da) < 1e-9 and abs(db) < 1e-9:
            break
    return float(a), float(b)


def temperature_fit(p: np.ndarray, y_bin: np.ndarray) -> float:
    """Single temperature T on logit(p), minimizing VAL log loss (golden section)."""
    eps = 1e-6
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1 - eps)
    logit = np.log(p / (1 - p))
    y = np.asarray(y_bin, dtype=np.float64)

    def nll(T: float) -> float:
        q = np.clip(sigmoid(logit / T), eps, 1 - eps)
        return float(-np.mean(y * np.log(q) + (1 - y) * np.log(1 - q)))

    lo, hi = 0.05, 20.0
    for _ in range(100):
        m1 = lo + (hi - lo) / 3.0
        m2 = hi - (hi - lo) / 3.0
        if nll(m1) < nll(m2):
            hi = m2
        else:
            lo = m1
    return float((lo + hi) / 2.0)


def conformal_quantile(residuals: np.ndarray, alpha: float) -> float:
    """Split-conformal absolute residual quantile with (n+1) correction."""
    r = np.sort(np.abs(np.asarray(residuals, dtype=np.float64)))
    n = len(r)
    k = int(math.ceil((n + 1) * (1 - alpha)))
    k = min(max(k, 1), n)
    return float(r[k - 1])


def raw_threshold_probability(pred: np.ndarray, thr: float, sigma: float = 4.0) -> np.ndarray:
    """Pre-calibration probability on a fixed EF scale.

    p = sigmoid((thr - pred) / sigma) with sigma fixed at 4 EF points. This is a
    deterministic, untrained map from the point estimate to a probability, used
    as the "before calibration" reference for R6.
    """
    return sigmoid((thr - np.asarray(pred, dtype=np.float64)) / sigma)


def calibrate_block(y_val, pred_val, y_test, pred_test, alphas=(0.1, 0.2)) -> Dict:
    out: Dict = {"thresholds": {}, "conformal": {}, "temperature": {}}
    for thr in (50, 40, 30):
        yv = (y_val < thr).astype(int)
        yt = (y_test < thr).astype(int)
        p_val_raw = raw_threshold_probability(pred_val, thr)
        p_te_raw = raw_threshold_probability(pred_test, thr)
        a, b = platt_fit(yv, p_val_raw)
        p_val_pl = sigmoid(a * p_val_raw + b)
        p_te_pl = sigmoid(a * p_te_raw + b)
        T = temperature_fit(p_val_raw, yv)
        eps = 1e-6
        p_te_ts = sigmoid(
            np.log(np.clip(p_te_raw, eps, 1 - eps) / (1 - np.clip(p_te_raw, eps, 1 - eps))) / T
        )
        out["thresholds"][f"ef_lt{thr}"] = {
            "platt_a": a,
            "platt_b": b,
            "temperature": T,
            "val": {
                "raw": classification_metrics(yv, p_val_raw),
                "platt": classification_metrics(yv, p_val_pl),
            },
            "test": {
                "raw": classification_metrics(yt, p_te_raw),
                "platt": classification_metrics(yt, p_te_pl),
                "temperature": classification_metrics(yt, p_te_ts),
            },
        }
    for alpha in alphas:
        q = conformal_quantile(pred_val - y_val, alpha)
        lo, hi = pred_test - q, pred_test + q
        out["conformal"][f"alpha_{alpha}"] = {
            "alpha": alpha,
            "target_coverage": 1 - alpha,
            "q": q,
            "coverage": float(np.mean((y_test >= lo) & (y_test <= hi))),
            "width_mean": float(np.mean(hi - lo)),
            "width_median": float(np.median(hi - lo)),
            "n_val": int(len(y_val)),
            "n_test": int(len(y_test)),
        }
    return out


# --------------------------------------------------------------------------
# experiment driver
# --------------------------------------------------------------------------
@dataclass
class ExpConfig:
    experiment: str
    seed: int
    n_frames: int = 32
    top_frac: float = 0.2
    train_limit: Optional[int] = None
    calibrate: bool = False
    alpha: float = 0.1
    pool: str = "cls"
    epochs: int = 120
    pretrain_epochs: int = 40
    tau: float = 5.0
    smooth_sigma: float = 5.0
    extra: Dict = field(default_factory=dict)


def _slice_train(split: Split, limit: Optional[int], seed: int) -> Split:
    if limit is None or limit >= len(split):
        return split
    rng = np.random.default_rng(seed + 7)
    idx = np.sort(rng.permutation(len(split))[:limit])
    return split.subset(idx)


def _load_seed_result(exp: str, seed: int, frames: int, pool: str = "cls") -> Optional[Dict]:
    """Load the JSON of an already-trained row (used by R6 -> R5)."""
    pat = f"{exp}_s{seed}_f{frames}_{pool}.json"
    p = RESULTS / pat
    if not p.exists():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    d["_source_file"] = p.name
    return d


def run_experiment(cfg: ExpConfig, device: str = "cpu") -> Dict:
    t0 = time.time()
    log(
        f"=== {cfg.experiment} seed={cfg.seed} frames={cfg.n_frames} "
        f"pool={cfg.pool} top={cfg.top_frac} n_train={cfg.train_limit} ==="
    )

    cache_tag = cfg.extra.get("cache_tag", "32_bgr")
    zero_shot = cfg.experiment in ("R0", "R1", "R1U16", "ORACLE_EDES")
    if cfg.experiment == "R0":
        # the official stride cache holds 20 frames (0:min(40,T):2)
        cache_tag = "stride_bgr"
        cfg.n_frames = 20
    # zero-shot rows never train, so missing TRAIN/VAL caches are not fatal
    tr = None if zero_shot else _slice_train(load_split("train", cache_tag), cfg.train_limit, cfg.seed)
    try:
        va = load_split("val", cache_tag)
    except FileNotFoundError:
        if not zero_shot:
            raise
        va = None
        log("  val cache absent; test-only zero-shot report")
    te = load_split("test500", cache_tag)

    result: Dict = {
        "experiment": cfg.experiment,
        "config": {
            "n_frames": cfg.n_frames,
            "pool": cfg.pool,
            "top_frac": cfg.top_frac,
            "train_limit": cfg.train_limit,
            "calibrate": cfg.calibrate,
            "alpha": cfg.alpha,
            "epochs": cfg.epochs,
            "pretrain_epochs": cfg.pretrain_epochs,
            "tau": cfg.tau,
            "smooth_sigma": cfg.smooth_sigma,
            "cache_tag": cache_tag,
            **cfg.extra,
        },
        "seed": cfg.seed,
        "n_train": len(tr) if tr is not None else 0,
        "n_val": len(va) if va is not None else 0,
        "n_test": len(te),
        "command": command_line(),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_commit": git_commit(),
        "data_root": str(DATA_ROOT),
        "cache": {
            "train": tr.path if tr is not None else None,
            "val": va.path if va is not None else None,
            "test": te.path,
        },
        "weights_sha256": sha256_file(WEIGHTS) if WEIGHTS.exists() else "missing",
        "device": device,
        "host": socket.gethostname(),
        "torch": _torch_version(),
    }

    name = cfg.experiment
    pred_test: Optional[np.ndarray] = None
    pred_val: Optional[np.ndarray] = None
    xtr = take_frames(tr.emb, cfg.n_frames) if tr is not None else None
    xva = take_frames(va.emb, cfg.n_frames) if va is not None else None
    xte = take_frames(te.emb, cfg.n_frames)

    if name in ("R0", "R1", "R1U16", "ORACLE_EDES"):
        if name == "R0":
            xte_z = xte  # stride cache already holds 20 official frames
            xva_z = xva
            result["config"]["frame_selection"] = "official_stride_0_40_2"
        elif name == "ORACLE_EDES":
            xte_z = edes_frames(te)
            xva_z = edes_frames(va) if va is not None else None
            result["config"]["frame_selection"] = "annotated_ed_es_nearest_cache_slot"
        else:
            xte_z, xva_z = xte, xva
            result["config"]["frame_selection"] = f"uniform_{cfg.n_frames}"
        pred_test = zero_shot_predict(xte_z, cfg.top_frac)
        result["metrics"] = regression_metrics(te.ef, pred_test)
        if va is not None and xva_z is not None:
            pred_val = zero_shot_predict(xva_z, cfg.top_frac)
            result["metrics_val"] = regression_metrics(va.ef, pred_val)

    elif name in ("R2", "R3"):
        ftr, fva, fte = xtr.mean(1), xva.mean(1), xte.mean(1)
        head = LinearRegressor(dim=ftr.shape[1]) if name == "R2" else MLPRegressor(
            dim=ftr.shape[1], hidden=256, dropout=0.2
        )
        head, state = fit_regression(
            head, ftr, tr.ef, fva, va.ef,
            seed=cfg.seed, device=device, epochs=cfg.epochs,
        )
        pred_val = predict_regression(head, fva, state, device)
        pred_test = predict_regression(head, fte, state, device)
        result["fit"] = state
        result["n_params"] = int(sum(p.numel() for p in head.parameters()))
        result["metrics"] = regression_metrics(te.ef, pred_test)
        result["metrics_val"] = regression_metrics(va.ef, pred_val)

    elif name in ("R4", "R4_MEAN", "R5", "R5_EDES", "R6"):
        pool_kind = "mean" if name == "R4_MEAN" or cfg.pool == "mean" else "cls"
        if name == "R6":
            # R6 = R5 predictions + calibration. Reuse the same-seed R5 result so
            # the two rows differ only by the calibration layer, by construction.
            r5 = _load_seed_result("R5", cfg.seed, cfg.n_frames)
            if r5 is None:
                raise SystemExit(
                    f"R6 needs R5 seed {cfg.seed} first (run --experiment R5 --seeds {cfg.seed})"
                )
            pred_val = np.array(r5["predictions"]["val_pred"], dtype=np.float64)
            pred_test = np.array(r5["predictions"]["test_pred"], dtype=np.float64)
            result["metrics"] = r5["metrics"]
            result["metrics_val"] = r5["metrics_val"]
            result["base_experiment"] = "R5"
            result["base_result_file"] = r5["_source_file"]
            result["n_params"] = r5.get("n_params")
        else:
            head = TemporalRegressor(
                dim=xtr.shape[-1], n_layers=2, n_heads=8,
                max_frames=max(cfg.n_frames, 32), pool=pool_kind,
            )
            curve: List[float] = []
            if name in ("R5", "R5_EDES"):
                xtr_c = edes_frames(tr) if name == "R5_EDES" else xtr
                head, curve = pretrain_contrastive(
                    head, xtr_c, tr.ef,
                    tau=cfg.tau, epochs=cfg.pretrain_epochs, seed=cfg.seed,
                    device=device, smooth_sigma=cfg.smooth_sigma,
                )
            head, state = fit_regression(
                head, xtr, tr.ef, xva, va.ef,
                seed=cfg.seed, device=device, epochs=cfg.epochs,
            )
            pred_val = predict_regression(head, xva, state, device)
            pred_test = predict_regression(head, xte, state, device)
            result["fit"] = state
            result["pretrain_curve"] = curve
            result["n_params"] = int(sum(p.numel() for p in head.parameters()))
            result["metrics"] = regression_metrics(te.ef, pred_test)
            result["metrics_val"] = regression_metrics(va.ef, pred_val)

    else:
        raise SystemExit(f"unknown experiment {name}")

    if name == "R6":
        result["calibration"] = calibrate_block(va.ef, pred_val, te.ef, pred_test)
    elif cfg.calibrate:
        result["calibration"] = calibrate_block(va.ef, pred_val, te.ef, pred_test, (cfg.alpha,))

    result["predictions"] = {
        "test_ids": [str(x) for x in te.ids],
        "val_ids": [str(x) for x in va.ids] if va is not None else [],
        "test_true": te.ef.tolist(),
        "test_pred": pred_test.tolist(),
        "val_true": va.ef.tolist() if va is not None else [],
        "val_pred": pred_val.tolist() if pred_val is not None else [],
    }
    result["bootstrap"] = {
        "mae": bootstrap_ci(te.ef, pred_test, "mae", seed=cfg.seed + 1),
        "auc50": bootstrap_ci(te.ef, pred_test, "auc50", seed=cfg.seed + 1),
        "auc40": bootstrap_ci(te.ef, pred_test, "auc40", seed=cfg.seed + 1),
    }
    result["elapsed_sec"] = round(time.time() - t0, 1)

    RESULTS.mkdir(parents=True, exist_ok=True)
    tag = f"{name}_s{cfg.seed}_f{cfg.n_frames}_{cfg.pool}"
    if cfg.train_limit:
        tag += f"_n{cfg.train_limit}"
    if cfg.top_frac != 0.2:
        tag += f"_top{int(cfg.top_frac * 100)}"
    if name == "R6":
        tag += "_a10_20"
    if cfg.extra.get("tag"):
        tag += "_" + str(cfg.extra["tag"])
    out = RESULTS / f"{tag}.json"
    out.write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    log(
        f"  MAE {result['metrics']['mae']:.3f} | AUC50 {result['metrics']['auc_ef_lt50']:.3f} "
        f"| R2 {result['metrics']['r2']:.3f} | {out.name} ({result['elapsed_sec']}s)"
    )
    return result


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="R2")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--frames", type=int, default=32)
    ap.add_argument("--top-frac", type=float, default=0.2)
    ap.add_argument("--train-limit", type=int, default=None)
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--pool", default="cls", choices=["cls", "mean"])
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--pretrain-epochs", type=int, default=40)
    ap.add_argument("--tau", type=float, default=5.0)
    ap.add_argument("--smooth-sigma", type=float, default=5.0)
    ap.add_argument("--cache-tag", default="32_bgr")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--device", default="cuda" if _cuda() else "cpu")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        print("\n".join(EXPERIMENT_IDS))
        return 0

    targets = EXPERIMENT_IDS if args.experiment == "all" else args.experiment.split(",")
    for exp in targets:
        for seed in args.seeds:
            cfg = ExpConfig(
                experiment=exp,
                seed=seed,
                n_frames=args.frames,
                top_frac=args.top_frac,
                train_limit=args.train_limit,
                calibrate=args.calibrate,
                alpha=args.alpha,
                pool=args.pool,
                epochs=args.epochs,
                pretrain_epochs=args.pretrain_epochs,
                tau=args.tau,
                smooth_sigma=args.smooth_sigma,
                extra={"cache_tag": args.cache_tag, "tag": args.tag},
            )
            run_experiment(cfg, device=args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
