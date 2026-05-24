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
from pic_selecter import clustering  # noqa: E402
from pic_selecter.grouper import ImageInfo  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def make_synthetic_images(folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    if any(p.is_file() and p.suffix.lower() in IMAGE_EXTS for p in folder.rglob("*")):
        return
    specs = [
        ("synthetic_warm_gradient.jpg", (220, 96, 60), (255, 210, 130)),
        ("synthetic_cool_gradient.jpg", (45, 115, 210), (165, 220, 255)),
        ("synthetic_checker.jpg", (235, 235, 235), (35, 35, 35)),
        ("synthetic_diagonal.jpg", (40, 180, 120), (240, 245, 210)),
    ]
    for idx, (name, a, b) in enumerate(specs):
        img = Image.new("RGB", (256, 192), a)
        px = img.load()
        for y in range(img.height):
            for x in range(img.width):
                if "checker" in name:
                    use_b = ((x // 24) + (y // 24)) % 2 == 0
                    color = b if use_b else a
                elif "diagonal" in name:
                    t = min(1.0, max(0.0, (x + y) / float(img.width + img.height)))
                    color = tuple(int(a[c] * (1 - t) + b[c] * t) for c in range(3))
                    if abs(x - y) < 4 or abs(x + y - 220) < 4:
                        color = (255, 255, 255)
                else:
                    t = x / max(1, img.width - 1)
                    color = tuple(int(a[c] * (1 - t) + b[c] * t) for c in range(3))
                    if (x - 70 - idx * 22) ** 2 + (y - 92) ** 2 < 34 ** 2:
                        color = (255, 255, 255)
                px[x, y] = color
        img.save(folder / name, quality=94)


def iter_images(folder: Path, limit: int) -> list[Path]:
    paths = [
        p
        for p in sorted(folder.rglob("*"))
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]
    return paths[:limit]


def relative_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def jsonable(value):
    if hasattr(value, "tolist"):
        return jsonable(value.tolist())
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, float):
        return round(value, 8)
    return value


def build_group_fixture(records: list[dict]) -> dict:
    infos = []
    for idx, item in enumerate(records):
        faces = item.get("faces") or []
        infos.append(
            ImageInfo(
                path=item["path"],
                phash="0" * 16,
                timestamp=float(idx * 10),
                mtime=float(idx * 10),
                dinov2=item["embedding"],
                face_embeddings=[face["embedding"] for face in faces],
                exif_summary=None,
            )
        )
    if len(infos) < 2:
        return {"group_indices": [[i] for i in range(len(infos))], "groups": [[r["path"]] for r in records]}
    group_indices = clustering.cluster(infos)
    return {
        "group_indices": group_indices,
        "groups": [[records[i]["path"] for i in group] for group in group_indices],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--faces", action="store_true", help="also export InsightFace bbox/landmark/embedding data")
    parser.add_argument("--path-root", type=Path, default=ROOT, help="root used to store image paths relatively")
    parser.add_argument("--no-synthetic", action="store_true", help="do not create deterministic synthetic images when folder is empty")
    args = parser.parse_args()

    if not args.no_synthetic:
        make_synthetic_images(args.folder)

    records = []
    for path in iter_images(args.folder, args.limit):
        with Image.open(path) as img:
            rgb = img.convert("RGB")
            emb = vision.extract_dinov2(rgb)
            item = {
                "path": relative_path(path, args.path_root),
                "name": path.name,
                "dinov2_dim": int(len(emb)),
                "embedding": [round(float(v), 8) for v in emb.tolist()],
            }
            if args.faces:
                faces = vision.extract_faces(rgb)
                face_signals = quality._face_signals_from_data(faces, rgb)
                item["face_count"] = len(faces)
                item["face_signals"] = jsonable(face_signals)
                item["faces"] = [
                    {
                        "bbox": [round(float(v), 4) for v in face["bbox"]],
                        "det_score": round(float(face.get("det_score", 1.0)), 6),
                        "embedding_dim": int(len(face["embedding"])),
                        "embedding": [round(float(v), 8) for v in face["embedding"].tolist()],
                        "kps": None if face.get("kps") is None else [[round(float(x), 4), round(float(y), 4)] for x, y in face["kps"].tolist()],
                        "landmark_2d_68": None if face.get("landmark_2d_68") is None else [[round(float(x), 4), round(float(y), 4)] for x, y in face["landmark_2d_68"].tolist()],
                    }
                    for face in faces
                ]
        records.append(item)

    grouping = build_group_fixture(records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "schema": "pianke.expert_golden.v1",
                "model": "facebook/dinov2-small",
                "dinov2": "cls_l2_normalized",
                "faces": "insightface_buffalo_l_optional",
                "count": len(records),
                "path_root": relative_path(args.path_root, ROOT),
                "grouping": grouping,
                "items": records,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
