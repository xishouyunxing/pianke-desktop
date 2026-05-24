#!/usr/bin/env python3
"""Export Python DINOv2 embeddings for Rust ONNX parity checks.

Usage:
    python scripts/export_dinov2_golden.py C:\photos fixtures\expert_parity\dinov2.json

The output contains L2-normalized CLS embeddings from pic_selecter.vision.extract_dinov2.
It is intentionally small and deterministic so Rust can compare ONNX output cosine similarity.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pic_selecter import vision  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def iter_images(folder: Path, limit: int) -> list[Path]:
    paths = [
        p
        for p in sorted(folder.rglob("*"))
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]
    return paths[:limit]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    records = []
    for path in iter_images(args.folder, args.limit):
        with Image.open(path) as img:
            emb = vision.extract_dinov2(img)
        records.append(
            {
                "path": str(path),
                "name": path.name,
                "dim": int(len(emb)),
                "embedding": [round(float(v), 8) for v in emb.tolist()],
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "model": "facebook/dinov2-small",
                "embedding": "cls_l2_normalized",
                "count": len(records),
                "items": records,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
