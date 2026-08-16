"""Cardiac-cycle frame sampling for EchoCLIP-TC.

Strategies
----------
random   : without-replacement random indices (training default)
uniform  : linspace over the clip (inference default)
ed_es    : keep end-diastole / end-systole if known; fill the rest uniformly
mixed    : choose random / uniform / ed_es (ed_es only if indices exist)
"""

from typing import Optional, Sequence, Union

import numpy as np

STRATEGIES = ("random", "uniform", "ed_es", "mixed")


def _clamp_index(index: Optional[int], num_frames: int) -> Optional[int]:
    if index is None:
        return None
    if num_frames <= 0:
        return None
    return int(max(0, min(num_frames - 1, int(index))))


def _unique_sorted(indices: Sequence[int]) -> np.ndarray:
    return np.array(sorted(dict.fromkeys(int(i) for i in indices)), dtype=int)


def sample_uniform(num_frames: int, n_samples: int) -> np.ndarray:
    if n_samples >= num_frames:
        return np.arange(num_frames, dtype=int)
    if n_samples <= 0:
        return np.zeros((0,), dtype=int)
    return np.linspace(0, num_frames - 1, n_samples, dtype=int)


def sample_random(
    num_frames: int, n_samples: int, seed: Optional[int] = None
) -> np.ndarray:
    if n_samples >= num_frames:
        return np.arange(num_frames, dtype=int)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(num_frames, size=n_samples, replace=False))


def sample_ed_es(
    num_frames: int,
    n_samples: int,
    ed_index: Optional[int] = None,
    es_index: Optional[int] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Prefer ED/ES frames; fill remaining slots from a uniform grid."""
    if n_samples >= num_frames:
        return np.arange(num_frames, dtype=int)

    ed = _clamp_index(ed_index, num_frames)
    es = _clamp_index(es_index, num_frames)
    must = []
    for idx in (ed, es):
        if idx is not None and idx not in must:
            must.append(idx)

    if not must:
        return sample_uniform(num_frames, n_samples)

    if n_samples <= len(must):
        return np.array(sorted(must[:n_samples]), dtype=int)

    grid = sample_uniform(num_frames, n_samples)
    chosen = list(must)
    for idx in grid.tolist():
        if idx not in chosen:
            chosen.append(idx)
        if len(chosen) >= n_samples:
            break

    if len(chosen) < n_samples:
        rng = np.random.default_rng(seed)
        rest = [i for i in range(num_frames) if i not in chosen]
        need = min(n_samples - len(chosen), len(rest))
        if need > 0:
            chosen.extend(rng.choice(rest, size=need, replace=False).tolist())

    chosen = sorted(chosen)
    # If trimming dropped a required index, swap from the tail.
    if len(chosen) > n_samples:
        kept = [i for i in chosen if i in must]
        extras = [i for i in chosen if i not in must]
        chosen = sorted(kept + extras[: max(0, n_samples - len(kept))])
    return np.array(chosen[:n_samples], dtype=int)


def sample_cycle_indices(
    num_frames: int,
    n_samples: int,
    strategy: str = "random",
    ed_index: Optional[int] = None,
    es_index: Optional[int] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Return sorted frame indices of length min(n_samples, num_frames)."""
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    if n_samples <= 0:
        return np.zeros((0,), dtype=int)

    key = str(strategy).strip().lower()
    if key == "mixed":
        rng = np.random.default_rng(seed)
        options = ["random", "uniform"]
        if ed_index is not None or es_index is not None:
            options.append("ed_es")
        key = str(rng.choice(options))
        seed = None if seed is None else int(rng.integers(0, 2**31 - 1))

    if key == "uniform":
        return sample_uniform(num_frames, n_samples)
    if key in ("ed_es", "edes", "ed-es"):
        return sample_ed_es(num_frames, n_samples, ed_index, es_index, seed)
    if key == "random":
        return sample_random(num_frames, n_samples, seed)
    raise ValueError(f"Unknown sampling strategy '{strategy}'. Use {STRATEGIES}.")


def pad_or_trim_indices(
    indices: Union[np.ndarray, Sequence[int]],
    n_samples: int,
    num_frames: int,
) -> np.ndarray:
    """Repeat the last index so batched tensors share a fixed T."""
    idx = np.asarray(indices, dtype=int)
    if idx.size == 0:
        idx = np.array([0], dtype=int)
    if idx.size >= n_samples:
        return idx[:n_samples]
    last = int(idx[-1]) if idx.size else 0
    last = int(max(0, min(num_frames - 1, last)))
    pad = np.full(n_samples - idx.size, last, dtype=int)
    return np.concatenate([idx, pad])
