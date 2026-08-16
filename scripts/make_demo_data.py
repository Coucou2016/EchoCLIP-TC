"""Create synthetic echo-like images and paired report text for smoke tests."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

REPORTS = [
    "LEFT VENTRICULAR EJECTION FRACTION IS ESTIMATED TO BE 55%. NORMAL LEFT VENTRICULAR SIZE AND SYSTOLIC FUNCTION.",
    "MILDLY DILATED LEFT VENTRICLE. LV EJECTION FRACTION IS 45%. NO SIGNIFICANT VALVULAR ABNORMALITY.",
    "ECHO DENSITY IN RIGHT VENTRICLE SUGGESTIVE OF CATHETER, PACER LEAD, OR ICD LEAD. NORMAL LEFT VENTRICULAR FUNCTION.",
    "THE INFERIOR VENA CAVA DEMONSTRATES LESS THAN 50% COLLAPSE CONSISTENT WITH ELEVATED RIGHT ATRIAL PRESSURE (8MMHG).",
    "SEVERE DILATED LEFT VENTRICLE. LV EJECTION FRACTION IS 30%. MODERATE MITRAL REGURGITATION.",
    "A BIOPROSTHETIC STENT-VALVE IS PRESENT IN THE AORTIC POSITION. NORMAL RIGHT VENTRICULAR SIZE.",
]


def synthetic_echo_image(seed: int, size: int = 256) -> Image.Image:
    rng = random.Random(seed)
    arr = np.zeros((size, size), dtype=np.uint8)
    cy, cx = size // 2, size // 2
    y, x = np.ogrid[:size, :size]
    mask = (x - cx) ** 2 + (y - cy) ** 2 < (size * 0.35) ** 2
    arr[mask] = rng.randint(40, 90)
    noise = np.random.default_rng(seed).integers(0, 16, (size, size), dtype=np.uint8)
    arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr, mode="L").convert("RGB")
    draw = ImageDraw.Draw(img)
    draw.ellipse(
        [cx - size // 4, cy - size // 3, cx + size // 4, cy + size // 3],
        outline=(180, 180, 180),
        width=2,
    )
    return img


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate demo EchoCLIP training data")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(r"E:\Projects\20260522-EchoCLIP\data\demo"),
    )
    parser.add_argument("--num-samples", type=int, default=64)
    args = parser.parse_args()

    img_dir = args.output_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    pairs = []
    for i in range(args.num_samples):
        fname = f"echo_{i:04d}.png"
        synthetic_echo_image(i).save(img_dir / fname)
        pairs.append(
            {
                "image": f"images/{fname}",
                "text": REPORTS[i % len(REPORTS)],
            }
        )

    manifest = {"pairs": pairs}
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {len(pairs)} pairs to {manifest_path}")


if __name__ == "__main__":
    main()
