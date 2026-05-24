#!/usr/bin/env python3
"""Export Python Expert vision outputs for Rust ONNX parity checks.

Usage:
    python scripts/export_dinov2_golden.py C:\photos fixtures\expert_parity\dinov2.json

The output contains L2-normalized DINOv2 CLS embeddings and optional InsightFace face data.
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
from pic_selecter import quality  # noqa: E402

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
    parser.add_argument("--faces", action="store_true", help="also export InsightFace bbox/landmark/embedding data")
    args = parser.parse_args()

    records = []
    for path in iter_images(args.folder, args.limit):
        with Image.open(path) as img:
            rgb = img.convert("RGB")
            emb = vision.extract_dinov2(rgb)
            item = {
                "path": str(path),
                "name": path.name,
                "dinov2_dim": int(len(emb)),
                "embedding": [round(float(v), 8) for v in emb.tolist()],
            }
            if args.faces:
                faces = vision.extract_faces(rgb)
                face_signals = quality._face_signals_from_data(faces, rgb)
                item["face_count"] = len(faces)
                item["face_signals"] = face_signals
                item["faces"] = [
                    {
                        "bbox": [int(v) for v in face["bbox"]],
                        "det_score": round(float(face.get("det_score", 1.0)), 6),
                        "embedding_dim": int(len(face["embedding"])),
                        "embedding": [round(float(v), 8) for v in face["embedding"].tolist()],
                        "kps": None if face.get("kps") is None else [[round(float(x), 4), round(float(y), 4)] for x, y in face["kps"].tolist()],
                        "landmark_2d_68": None if face.get("landmark_2d_68") is None else [[round(float(x), 4), round(float(y), 4)] for x, y in face["landmark_2d_68"].tolist()],
                    }
                    for face in faces
                ]
        records.append(item)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "model": "facebook/dinov2-small",
                "dinov2": "cls_l2_normalized",
                "faces": "insightface_buffalo_l_optional",
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
