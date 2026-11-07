"""Cache frozen EchoCLIP frame embeddings for the R0-R6 paper matrix.

One pass over the AVI files, one row per video:

    ids       (N,)      file name stem
    emb       (N,T,512) float32 L2-normalized frame embeddings
    ef        (N,)      EF label from FileList.csv (via manifest)
    edv, esv  (N,)
    ed_frame  (N,)      ED frame index from VolumeTracings.csv (via manifest)
    es_frame  (N,)

Preprocessing follows the documented official path: OpenCV decode, letterbox
crop/scale straight to 224x224 with zoom=0.1, CLIP mean/std normalization.
Frame selection is either ``uniform`` (paper heads) or ``official_stride``
(0:min(40,T):2, R0 only).

The run is resumable: every ``--ckpt-every`` videos the accumulated arrays are
written to ``<out>.part.npz`` and a restart skips ids already present.

Usage
-----
python scripts/extract_frames_embeddings.py --split test500 --limit 20 --smoke
python scripts/extract_frames_embeddings.py --split train --limit 2000 --frames 32
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from _paths import cache_dir  # noqa: E402

DEFAULT_DATA = Path(r"E:\Datasets\EchoNet-Dynamic")
DEFAULT_OUT = cache_dir()
DEFAULT_MANIFEST = ROOT / "data" / "echonet_dynamic"
WEIGHTS_DIR = ROOT / "models" / "echo-clip"

SPLIT_FILE = {
    "train": "train.json",
    "val": "val.json",
    "test": "test.json",
    "test500": "test_subset_500.json",
    "anchor": "r0_external_anchor_5000.json",
}

CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

EF_TEMPLATES = (
    "THE LEFT VENTRICULAR EJECTION FRACTION IS ESTIMATED TO BE <#>% ",
    "LV EJECTION FRACTION IS <#>%. ",
)


# --------------------------------------------------------------------------
# environment shims
# --------------------------------------------------------------------------
def install_lzma_stub() -> None:
    """torchvision/open_clip import hygiene on this Windows box (_lzma DLL)."""
    if "lzma" in sys.modules:
        return
    fake = types.ModuleType("lzma")

    class LZMAError(Exception):
        pass

    fake.LZMAError = LZMAError
    fake.open = lambda *a, **k: (_ for _ in ()).throw(LZMAError("lzma unavailable"))
    sys.modules["lzma"] = fake
    sys.modules["_lzma"] = fake


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------
def load_echo_clip(device: str, dtype=None):
    install_lzma_stub()
    import open_clip
    import torch

    cfg_path = WEIGHTS_DIR / "open_clip_config.json"
    cfg = json.loads(cfg_path.read_text())["model_cfg"]
    model = open_clip.create_model(
        cfg["vision_cfg"]["timm_model_name"],
        pretrained=None,
        precision="fp32",
    )
    sd = torch.load(WEIGHTS_DIR / "open_clip_pytorch_model.bin", map_location="cpu")
    report = model.load_state_dict(sd, strict=False)
    if report.missing_keys or report.unexpected_keys:
        raise RuntimeError(
            f"weight mismatch: missing={len(report.missing_keys)} "
            f"unexpected={len(report.unexpected_keys)}"
        )
    model = model.to(device).eval()
    if dtype is not None:
        model = model.to(dtype=dtype)
    return model


def build_ef_prompts():
    prompts, values = [], []
    for tmpl in EF_TEMPLATES:
        for v in range(0, 101):
            prompts.append(tmpl.replace("<#>", str(v)))
            values.append(float(v))
    return prompts, values


def encode_text_prompts(model, device: str) -> dict:
    install_lzma_stub()
    import open_clip
    import torch
    import torch.nn.functional as F

    tok = open_clip.get_tokenizer(f"local-dir:{WEIGHTS_DIR}")
    prompts, values = build_ef_prompts()
    # Encode the 202 prompts in small chunks. A single 202x77 forward pass
    # trips the Windows TDR watchdog on a 4 GB card ("the launch timed out
    # and was terminated"), so the batch is capped and every chunk is moved
    # to host memory before the next one starts.
    chunk = 32
    chunks = []
    with torch.no_grad():
        for s in range(0, len(prompts), chunk):
            part = prompts[s : s + chunk]
            ids = tok(part).to(device)
            e = model.encode_text(ids)
            chunks.append(F.normalize(e, dim=-1).float().cpu().numpy())
            del ids, e
        emb = np.concatenate(chunks, axis=0)
    if device == "cuda":
        torch.cuda.empty_cache()
    return {
        "prompts": np.array(prompts),
        "values": np.array(values, dtype=np.float32),
        "emb": emb.astype(np.float32),
    }


# --------------------------------------------------------------------------
# frames
# --------------------------------------------------------------------------
def sample_indices(n_total: int, k: int, mode: str) -> np.ndarray:
    """Frame indices; always returns exactly ``k`` entries (clamped repeats)."""
    if mode == "official_stride":
        # 0:min(40,T):2 -> 20 frames; store exactly 20 slots
        end = min(40, n_total)
        idx = np.arange(0, end, 2, dtype=int)
        if idx.size < 20:
            idx = np.concatenate([idx, np.full(20 - idx.size, idx[-1], dtype=int)])
        return idx[:20]
    if n_total <= k:
        idx = np.arange(n_total, dtype=int)
        return np.concatenate([idx, np.full(k - n_total, n_total - 1, dtype=int)])
    return np.linspace(0, n_total - 1, k, dtype=int)


def read_and_crop(avi: Path, k: int, mode: str, color: str) -> np.ndarray:
    """Return (T,224,224,3) uint8 frames after the documented crop."""
    import cv2

    from echoclip.official_parity import crop_and_scale_official

    cap = cv2.VideoCapture(str(avi))
    frames = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        cap.release()
    if not frames:
        raise RuntimeError(f"no frames decoded: {avi}")
    idx = sample_indices(len(frames), k, mode)
    out = []
    for i in idx:
        f = frames[int(i)]
        c = crop_and_scale_official(f, (224, 224), zoom=0.1)
        if color == "rgb":
            c = cv2.cvtColor(c, cv2.COLOR_BGR2RGB)
        out.append(c)
    return np.stack(out, axis=0)


def to_tensor(batch: np.ndarray):
    import torch

    x = torch.from_numpy(batch).permute(0, 3, 1, 2).float().div_(255.0)
    mean = torch.from_numpy(CLIP_MEAN).view(1, 3, 1, 1)
    std = torch.from_numpy(CLIP_STD).view(1, 3, 1, 1)
    return (x - mean) / std


# --------------------------------------------------------------------------
# main loop
# --------------------------------------------------------------------------
def load_state(part: Path):
    if not part.exists():
        return None
    try:
        z = np.load(part, allow_pickle=False)
        return {k: z[k] for k in z.files}
    except Exception as exc:  # corrupt partial file -> restart that file
        log(f"could not read {part} ({exc}); starting fresh")
        return None


def save_state(part: Path, store: dict) -> None:
    tmp = part.with_suffix(".tmp.npz")
    np.savez(tmp, **store)
    os.replace(tmp, part)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test500", choices=sorted(SPLIT_FILE))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--frames", type=int, default=32)
    ap.add_argument("--sample", default="uniform", choices=["uniform", "official_stride"])
    ap.add_argument("--color", default="bgr", choices=["bgr", "rgb"])
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="cuda" if _cuda() else "cpu")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--ckpt-every", type=int, default=100)
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--data-root", default=str(DEFAULT_DATA))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true", help="20 videos, timing only")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_root = Path(args.data_root)
    manifest = DEFAULT_MANIFEST / SPLIT_FILE[args.split]
    pairs = json.loads(manifest.read_text())["pairs"]
    if args.split == "test500":
        # keep the official subset order
        pass
    else:
        rng = np.random.default_rng(args.seed)
        order = rng.permutation(len(pairs))
        pairs = [pairs[i] for i in order]
    if args.smoke:
        args.limit = 20
    if args.limit:
        pairs = pairs[: args.limit]

    tag_frames = "stride" if args.sample == "official_stride" else str(args.frames)
    stem = f"{args.split}_{tag_frames}_{args.color}"
    part = out_dir / f"{stem}.part.npz"
    final = out_dir / f"{stem}.npz"

    log(
        f"split={args.split} n={len(pairs)} frames={tag_frames} color={args.color} "
        f"device={args.device} batch={args.batch_size}"
    )

    state = None if args.smoke else load_state(part)
    done = set()
    if state is not None:
        done = set(str(x) for x in state["ids"])
        log(f"resuming from {part} with {len(done)} videos")

    import torch

    if args.device == "cpu":
        torch.set_num_threads(os.cpu_count() or 8)
    model = load_echo_clip(args.device)

    text_pack = encode_text_prompts(model, args.device)
    np.savez(out_dir / "text_ef.npz", **text_pack)
    log(f"text prompts cached: {text_pack['emb'].shape}")

    todo = [p for p in pairs if Path(p["file_name"]).stem not in done]
    log(f"{len(todo)} videos to encode")

    store = {k: [] for k in ("ids", "emb", "ef", "edv", "esv", "ed_frame", "es_frame")}
    if state is not None:
        for k in store:
            store[k] = [v for v in state[k]]

    lock = threading.Lock()
    buffer: list = []
    t_start = time.time()
    n_done_this_run = 0
    errors: list = []

    def encode_buffer() -> None:
        nonlocal buffer
        if not buffer:
            return
        items = buffer
        buffer = []
        flat = np.concatenate([it[1] for it in items], axis=0)
        chunks = []
        with torch.no_grad():
            for s in range(0, flat.shape[0], args.batch_size):
                xb = to_tensor(flat[s : s + args.batch_size]).to(args.device)
                with torch.autocast("cuda", enabled=False):
                    try:
                        e = model.encode_image(xb)
                    except RuntimeError as exc:
                        # a TDR reset leaves the context unusable; record and
                        # let the outer loop end so the checkpoint is kept
                        msg = str(exc)
                        if "timed out" in msg or "CUDA error" in msg:
                            errors.append(f"CUDA failure on batch starting {s}: {msg}")
                            raise SystemExit(3)
                        raise
                e = torch.nn.functional.normalize(e.float(), dim=-1)
                chunks.append(e.cpu().numpy())
                del xb, e
        emb = np.concatenate(chunks, axis=0).astype(np.float32)
        off = 0
        for meta, frames in items:
            n = frames.shape[0]
            emb_v = emb[off : off + n].copy()
            off += n
            # uniform-32 always yields the same count; stride varies with T
            store["ids"].append(meta["file_name"].split(".")[0])
            store["emb"].append(emb_v)
            store["ef"].append(meta["ef"])
            store["edv"].append(meta.get("edv", np.nan))
            store["esv"].append(meta.get("esv", np.nan))
            store["ed_frame"].append(meta.get("ed_frame", -1))
            store["es_frame"].append(meta.get("es_frame", -1))

    def pack_store() -> dict:
        n = len(store["ids"])
        if n == 0:
            return {k: np.array([]) for k in store}
        widths = {v.shape[0] for v in store["emb"]}
        if len(widths) > 1:
            # pad ragged sequences (stride mode) to the max length
            tmax = max(widths)
            padded = []
            for v in store["emb"]:
                if v.shape[0] < tmax:
                    pad = np.zeros((tmax - v.shape[0], v.shape[1]), dtype=np.float32)
                    v = np.concatenate([v, pad], axis=0)
                padded.append(v)
            store["emb"] = padded
        return {
            "ids": np.array(store["ids"]),
            "emb": np.stack(store["emb"]).astype(np.float32),
            "ef": np.array(store["ef"], dtype=np.float32),
            "edv": np.array(store["edv"], dtype=np.float32),
            "esv": np.array(store["esv"], dtype=np.float32),
            "ed_frame": np.array(store["ed_frame"], dtype=np.int32),
            "es_frame": np.array(store["es_frame"], dtype=np.int32),
        }

    def worker(pair):
        avi = data_root / "Videos" / Path(pair["file_name"]).name
        return pair, read_and_crop(avi, args.frames, args.sample, args.color)

    # Bounded in-flight window: keep at most workers*3 videos decoded ahead so
    # peak RAM stays near a few hundred MB instead of the whole split.
    t0 = time.time()
    window = max(args.workers * 3, 6)
    t_ckpt = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = {}
        it = iter(todo)
        for _ in range(window):
            p = next(it, None)
            if p is None:
                break
            pending[pool.submit(worker, p)] = p
        while pending:
            done_futs = []
            for fut in list(pending):
                if fut.done():
                    done_futs.append(fut)
            if not done_futs:
                time.sleep(0.02)
                continue
            for fut in done_futs:
                pending.pop(fut, None)
                p = next(it, None)
                if p is not None:
                    pending[pool.submit(worker, p)] = p
                try:
                    meta, frames = fut.result()
                except Exception as exc:  # noqa: BLE001
                    errors.append(str(exc))
                    log(f"ERROR {exc}")
                    continue
                buffer.append((meta, frames))
                n_done_this_run += 1
                if sum(it2[1].shape[0] for it2 in buffer) >= args.batch_size * 4:
                    encode_buffer()
                if n_done_this_run % 20 == 0:
                    rate = n_done_this_run / max(time.time() - t0, 1e-6)
                    log(
                        f"{n_done_this_run}/{len(todo)} videos, {rate:.2f} vid/s, "
                        f"eta {(len(todo) - n_done_this_run) / max(rate, 1e-6) / 60:.1f} min"
                    )
                if not args.smoke and time.time() - t_ckpt > 240:
                    encode_buffer()
                    save_state(part, pack_store())
                    t_ckpt = time.time()
                    log(f"  checkpoint at {len(store['ids'])} videos")
    encode_buffer()

    packed = pack_store()
    if args.smoke:
        log(f"smoke done in {time.time() - t0:.1f}s for {n_done_this_run} videos")
        log(f"emb shape {packed['emb'].shape}")
        log(f"norm check {np.linalg.norm(packed['emb'][0][0]):.4f}")
        return 0

    save_state(final, packed)
    if part.exists():
        part.unlink()
    log(
        f"saved {final} emb={packed['emb'].shape} "
        f"total {time.time() - t_start:.1f}s errors={len(errors)}"
    )
    if errors:
        (out_dir / f"{stem}_errors.json").write_text(json.dumps(errors, indent=1))
    return 0


def _cuda() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


if __name__ == "__main__":
    raise SystemExit(main())
