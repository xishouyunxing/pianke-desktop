"""Export and verify pyiqa MUSIQ / CLIP-IQA+ assets for Rust parity.

This script intentionally writes generated ONNX files outside Git-tracked
paths. It mirrors the original Python Expert quality path:

1. PIL RGB conversion
2. long side resize to 1024 with PIL LANCZOS, using int() floor dimensions
3. pyiqa MUSIQ / CLIP-IQA+ inference on the resized PIL image

CLIP-IQA+ can be exported as a dynamic H/W image model. MUSIQ is exported as a
core model that accepts pyiqa multiscale patches; Rust still needs a matching
patch extractor before MUSIQ can be marked parity-ready.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from PIL import Image


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def iter_images(folder: Path, limit: int | None) -> list[Path]:
    paths = sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    return paths[:limit] if limit else paths


def resize_for_pyiqa(img: Image.Image, max_side: int = 1024) -> Image.Image:
    rgb = img.convert("RGB")
    w, h = rgb.size
    if max(w, h) <= max_side:
        return rgb
    scale = max_side / max(w, h)
    return rgb.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)


def load_metric(name: str):
    import pyiqa
    import torch

    metric = pyiqa.create_metric(name, device=torch.device("cpu"), as_loss=False)
    metric.eval()
    return metric


def score_images(paths: Iterable[Path], max_side: int) -> list[dict]:
    import torch

    musiq = load_metric("musiq")
    clipiqa = load_metric("clipiqa+")
    nima = load_metric("nima-vgg16-ava")
    rows = []
    with torch.no_grad():
        for path in paths:
            img = Image.open(path)
            resized = resize_for_pyiqa(img, max_side=max_side)
            musiq_score = float(musiq(resized).item())
            clipiqa_score = float(clipiqa(resized).item())
            nima_score = float(nima(resized).item())
            rows.append(
                {
                    "path": path.as_posix(),
                    "resized_width": resized.width,
                    "resized_height": resized.height,
                    "musiq_score": round(musiq_score, 4),
                    "clipiqa_score": round(clipiqa_score, 6),
                    "nima_score": round(nima_score, 4),
                }
            )
    return rows


def first_dummy(paths: list[Path], max_side: int):
    import torchvision.transforms.functional as tvf

    if not paths:
        raise SystemExit("No images found for ONNX dummy input")
    img = resize_for_pyiqa(Image.open(paths[0]), max_side=max_side)
    return img, tvf.to_tensor(img).unsqueeze(0)


def export_clipiqa(paths: list[Path], out_path: Path, max_side: int) -> None:
    import torch

    _, dummy = first_dummy(paths, max_side)
    metric = load_metric("clipiqa+")

    class ClipIqaWrapper(torch.nn.Module):
        def __init__(self, net):
            super().__init__()
            self.net = net

        def forward(self, x):
            return self.net(x)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        ClipIqaWrapper(metric.net).eval(),
        dummy,
        str(out_path),
        input_names=["input"],
        output_names=["score"],
        opset_version=17,
        dynamo=False,
        dynamic_axes={"input": {0: "batch", 2: "height", 3: "width"}, "score": {0: "batch"}},
    )


def export_musiq_core(paths: list[Path], out_path: Path, max_side: int) -> None:
    import torch
    import torchvision.transforms.functional as tvf
    from pyiqa.archs.musiq_arch import get_multiscale_patches

    img, _ = first_dummy(paths, max_side)
    x = tvf.to_tensor(img).unsqueeze(0)
    metric = load_metric("musiq")
    net = metric.net.eval()
    with torch.no_grad():
        patches = get_multiscale_patches((x - 0.5) * 2, **net.data_preprocess_opts)

    class MusiqCoreWrapper(torch.nn.Module):
        def __init__(self, net):
            super().__init__()
            self.net = net

        def forward(self, x):
            batch = x.shape[0]
            seq_len = x.shape[1]
            spatial = x[:, :, -3]
            scale = x[:, :, -2]
            mask = x[:, :, -1].bool()
            y = x[:, :, :-3]
            y = y.reshape(-1, 3, self.net.patch_size, self.net.patch_size)
            y = self.net.conv_root(y)
            y = self.net.gn_root(y)
            y = self.net.root_pool(y)
            y = self.net.block1(y)
            y = y.permute(0, 2, 3, 1)
            y = y.reshape(batch, seq_len, -1)
            y = self.net.embedding(y)
            y = self.net.transformer_encoder(y, spatial, scale, mask)
            q = self.net.head(y[:, 0])
            return q.reshape(batch, 1, -1).mean(dim=1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        MusiqCoreWrapper(net).eval(),
        patches,
        str(out_path),
        input_names=["patches"],
        output_names=["score"],
        opset_version=17,
        dynamo=False,
        dynamic_axes={"patches": {0: "batch", 1: "seq_len"}, "score": {0: "batch"}},
    )


def export_nima_core(out_path: Path) -> None:
    import torch

    metric = load_metric("nima-vgg16-ava")
    net = metric.net.eval()

    class NimaCoreWrapper(torch.nn.Module):
        def __init__(self, net):
            super().__init__()
            self.net = net

        def forward(self, x):
            x = self.net.base_model(x)[-1]
            x = self.net.global_pool(x)
            dist = self.net.classifier(x)
            scores = torch.arange(1, 11, dtype=dist.dtype, device=dist.device)
            return torch.sum(dist * scores, dim=1, keepdim=True)

    dummy = torch.zeros(1, 3, 224, 224, dtype=torch.float32)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        NimaCoreWrapper(net).eval(),
        dummy,
        str(out_path),
        input_names=["input"],
        output_names=["score"],
        opset_version=17,
        dynamo=False,
        dynamic_axes={"input": {0: "batch"}, "score": {0: "batch"}},
    )


def write_preprocessor(path: Path, max_side: int) -> None:
    payload = {
        "max_side": max_side,
        "resize_filter": "pillow_lanczos",
        "resize_rounding": "floor",
        "scale_255": False,
        "musiq_input_kind": "pyiqa_multiscale_patches",
        "musiq_patch_size": 32,
        "musiq_patch_stride": 32,
        "musiq_hse_grid_size": 10,
        "musiq_longer_side_lengths": [224, 384],
        "musiq_max_seq_len_from_original_res": -1,
        "musiq_input_name": "patches",
        "musiq_output_name": "score",
        "clipiqa_input_width": None,
        "clipiqa_input_height": None,
        "clipiqa_input_name": "input",
        "clipiqa_output_name": "score",
        "nima_model": "pyiqa-nima-vgg16-ava",
        "nima_input_width": 224,
        "nima_input_height": 224,
        "nima_resize_shorter": 224,
        "nima_mean": [0.485, 0.456, 0.406],
        "nima_std": [0.229, 0.224, 0.225],
        "nima_input_name": "input",
        "nima_output_name": "score",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--max-side", type=int, default=1024)
    parser.add_argument("--scores-out", type=Path)
    parser.add_argument("--export-dir", type=Path)
    parser.add_argument("--export-clipiqa", action="store_true")
    parser.add_argument("--export-musiq-core", action="store_true")
    parser.add_argument("--export-nima-core", action="store_true")
    args = parser.parse_args()

    paths = iter_images(args.folder, args.limit)
    if args.scores_out:
        rows = score_images(paths, args.max_side)
        args.scores_out.parent.mkdir(parents=True, exist_ok=True)
        args.scores_out.write_text(
            json.dumps(
                {
                    "schema": "pianke.quality_scores.v1",
                    "preprocess": {
                        "max_side": args.max_side,
                        "resize_filter": "pillow_lanczos",
                        "resize_rounding": "floor",
                    },
                    "items": rows,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    if args.export_dir:
        if args.export_clipiqa:
            export_clipiqa(paths, args.export_dir / "models/quality/clipiqa_plus.onnx", args.max_side)
        if args.export_musiq_core:
            export_musiq_core(paths, args.export_dir / "models/quality/musiq.onnx", args.max_side)
        if args.export_nima_core:
            export_nima_core(args.export_dir / "models/quality/nima_vgg16_ava.onnx")
        if args.export_clipiqa or args.export_musiq_core or args.export_nima_core:
            write_preprocessor(args.export_dir / "quality_preprocessor.json", args.max_side)


if __name__ == "__main__":
    main()
