"""Export Python Fast-mode golden data for Rust parity tests.

Usage:
    python scripts/export_fast_golden.py C:\photos fixtures\fast_parity\sample.json

The output intentionally includes both API-level grouping and intermediate
signals so Rust can migrate one layer at a time without guessing Python
behavior.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PIL import ImageOps  # noqa: E402

from pic_selecter.grouper import compute_infos, group_infos  # noqa: E402
from pic_selecter import grouper  # noqa: E402
from pic_selecter import fast_clustering  # noqa: E402
from pic_selecter import fast_quality  # noqa: E402


def _jsonable(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _array_summary(arr: Any) -> dict[str, Any] | None:
    if arr is None:
        return None
    a = np.asarray(arr)
    if a.size == 0:
        return {"shape": list(a.shape), "dtype": str(a.dtype), "size": 0}
    flat = a.reshape(-1)
    sample = flat[: min(16, flat.size)].tolist()
    return {
        "shape": list(a.shape),
        "dtype": str(a.dtype),
        "size": int(a.size),
        "sum": float(a.sum()),
        "mean": float(a.mean()),
        "sample": _jsonable(sample),
    }


def _info_record(info: Any, index: int) -> dict[str, Any]:
    orb_descs = getattr(info, "orb_descs", None)
    orb_kps = getattr(info, "orb_kps", None)
    quality_signals = _quality_signal_record(info)
    return {
        "index": index,
        "path": str(Path(info.path)),
        "name": Path(info.path).name,
        "companions": [str(Path(p)) for p in getattr(info, "companions", [])],
        "timestamp": getattr(info, "timestamp", None),
        "size": getattr(info, "size", None),
        "mtime": getattr(info, "mtime", None),
        "exif_summary": _jsonable(getattr(info, "exif_summary", None)),
        "hashes": {
            "phash": getattr(info, "phash", None),
            "dhash": getattr(info, "dhash", None),
            "whash": getattr(info, "whash", None),
            "ahash": getattr(info, "ahash", None),
        },
        "quality": _jsonable(getattr(info, "quality", None)),
        "quality_signals": _jsonable(quality_signals),
        "color_hist": _jsonable(getattr(info, "color_hist", None)),
        "color_hist_summary": _array_summary(getattr(info, "color_hist", None)),
        "orb": {
            "descriptors": _array_summary(orb_descs),
            "keypoints": _array_summary(orb_kps),
        },
    }


def _quality_signal_record(info: Any) -> dict[str, Any] | None:
    try:
        img = grouper._load_image_for_analysis(info.path, getattr(info, "companions", []))
        img_t = grouper._resize_for_analysis(ImageOps.exif_transpose(img))
        work = fast_quality._resize_for_analysis(img_t.convert("L"), 768)
        arr = np.asarray(work, dtype=np.float32)
        if arr.size == 0:
            arr = np.zeros((1, 1), dtype=np.float32)
        brightness_mean = float(arr.mean())
        brightness_std = float(arr.std())
        underexposed_ratio = float((arr <= 8).mean())
        overexposed_ratio = float((arr >= 247).mean())
        entropy = fast_quality._entropy(arr)
        lap = max(
            fast_quality._laplacian_variance(arr),
            fast_quality._laplacian_variance(fast_quality._center_crop(arr, 0.6)),
        )
        tenengrad = fast_quality._tenengrad(arr)
        high_ratio, motion_anisotropy = fast_quality._fft_high_freq_ratio(arr)
        edge_width = fast_quality._edge_width_marziliano(arr)
        lap_norm = min(1.0, np.log1p(max(0.0, lap)) / np.log1p(900.0))
        tenengrad_norm = min(1.0, np.log1p(max(0.0, tenengrad)) / np.log1p(2000.0))
        high_norm = min(1.0, max(0.0, high_ratio) / 0.40)
        if edge_width is None:
            edge_width_norm = None
            blur_combined = float(np.mean([lap_norm, tenengrad_norm, high_norm]))
        else:
            edge_width_norm = max(0.0, min(1.0, (10.0 - edge_width) / 7.0))
            blur_combined = float(np.mean([lap_norm, tenengrad_norm, high_norm, edge_width_norm]))
        smap = fast_quality._saliency_map(arr)
        salient_sharpness = None
        focus_ratio = None
        composition = None
        if smap is not None:
            salient_sharpness = fast_quality._salient_region_sharpness(arr, smap)
            focus_ratio = fast_quality._saliency_focus_consistency(arr, smap)
            composition = fast_quality._composition_score(arr, smap)
        nine = fast_quality._nine_grid_exposure(arr)
        horizon_tilt = fast_quality._horizon_tilt_degrees(arr)
        return {
            "work_width": int(arr.shape[1]),
            "work_height": int(arr.shape[0]),
            "brightness_mean": brightness_mean,
            "brightness_std": brightness_std,
            "contrast_score": brightness_std,
            "underexposed_ratio": underexposed_ratio,
            "overexposed_ratio": overexposed_ratio,
            "entropy": entropy,
            "lap": lap,
            "tenengrad": tenengrad,
            "high_freq_ratio": high_ratio,
            "motion_anisotropy": motion_anisotropy,
            "edge_width_pix": edge_width,
            "lap_norm": lap_norm,
            "tenengrad_norm": tenengrad_norm,
            "high_norm": high_norm,
            "edge_width_norm": edge_width_norm,
            "blur_combined": blur_combined,
            "salient_sharpness": salient_sharpness,
            "focus_ratio": focus_ratio,
            "horizon_tilt_deg": horizon_tilt,
            "composition": composition,
            "nine_grid": nine,
        }
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def _orb_pair_records(infos: list[Any]) -> list[dict[str, Any]]:
    metas = [info.exif_summary if getattr(info, "exif_summary", None) else None for info in infos]
    pairs: list[dict[str, Any]] = []
    for i in range(len(infos)):
        for j in range(i + 1, len(infos)):
            base_sim = fast_clustering._pair_base_sim(infos[i], infos[j], metas[i], metas[j])
            hash_sim = fast_clustering._hash_combined_sim(infos[i], infos[j])
            if base_sim < fast_clustering.ORB_CANDIDATE_BASE and hash_sim < 0.65:
                continue
            ta = fast_clustering._time_for_info(infos[i])
            tb = fast_clustering._time_for_info(infos[j])
            if (
                ta is not None
                and tb is not None
                and abs(ta - tb) > fast_clustering.HARD_BREAK_SECONDS
            ):
                pairs.append({
                    "i": i,
                    "j": j,
                    "base_sim": base_sim,
                    "hash_sim": hash_sim,
                    "orb_inliers": 0,
                    "final_sim": base_sim * 0.5,
                    "time_hard_split": True,
                })
                continue
            inliers = fast_clustering._orb_inliers(
                getattr(infos[i], "orb_descs", None),
                getattr(infos[j], "orb_descs", None),
                getattr(infos[i], "orb_kps", None),
                getattr(infos[j], "orb_kps", None),
            )
            pairs.append({
                "i": i,
                "j": j,
                "base_sim": base_sim,
                "hash_sim": hash_sim,
                "orb_inliers": int(inliers),
                "final_sim": fast_clustering._pair_final_sim(infos[i], infos[j], metas[i], metas[j]),
                "time_hard_split": False,
            })
    return pairs


def export_fast_golden(
    folder: Path,
    output: Path,
    strength: str,
    workers: int | None,
    include_pairs: bool,
) -> None:
    infos, skipped = compute_infos(
        str(folder),
        workers=workers,
        strength=strength,
        engine="fast",
    )
    groups = group_infos(infos, engine="fast")
    path_to_index = {info.path: i for i, info in enumerate(infos)}
    group_indices = [[path_to_index[item.path] for item in group] for group in groups]

    payload = {
        "schema": "pianke.fast_golden.v1",
        "source_folder": str(folder),
        "strength": strength,
        "count": len(infos),
        "skipped": [{"path": p, "reason": r} for p, r in skipped],
        "images": [_info_record(info, i) for i, info in enumerate(infos)],
        "orb_pairs": _orb_pair_records(infos) if include_pairs else [],
        "groups": group_indices,
        "group_paths": [[item.path for item in group] for group in groups],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Export Python Fast golden parity JSON.")
    parser.add_argument("folder", type=Path, help="Photo folder to scan.")
    parser.add_argument("output", type=Path, help="JSON file to write.")
    parser.add_argument("--strength", default="standard", choices=["standard", "advanced", "aggressive"])
    parser.add_argument("--workers", type=int, default=1, help="Worker count; default 1 for stable golden output.")
    parser.add_argument("--skip-pairs", action="store_true", help="Skip ORB pair matrix export.")
    args = parser.parse_args()

    export_fast_golden(args.folder, args.output, args.strength, args.workers, not args.skip_pairs)
    print(f"exported fast golden: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
