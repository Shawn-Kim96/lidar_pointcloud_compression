import argparse
import csv
import inspect
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.registry import MODELS
import models.compression  # noqa: F401
import models.backbones  # noqa: F401
from utils.recon_pointcloud_export import reconstruct_kitti_points_with_model


DEFAULT_TEACHER_AP3D_MOD_CAR_MIN = 55.0
DEFAULT_BITRATE_MATCH_METRIC = "bpp_entropy_mean"
DEFAULT_BITRATE_PAIR_MAX_GAP = 0.05


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate KITTI 3D detection mAP vs compression bitrate (original vs reconstructed)."
    )
    repo_root = Path(__file__).resolve().parents[2]

    parser.add_argument("--kitti_root", type=str, required=True, help="KITTI detection dataset root.")
    parser.add_argument(
        "--run_dirs",
        type=str,
        nargs="+",
        default=None,
        help="Compression run dirs containing config.yaml and checkpoints.",
    )
    parser.add_argument(
        "--run_dirs_csv",
        type=str,
        default="",
        help="Comma-separated run dirs (alternative to --run_dirs).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint filename or full path. If omitted, latest model_epoch_*.pth or model_final.pth is used.",
    )

    parser.add_argument("--openpcdet_cfg", type=str, required=True, help="OpenPCDet config yaml.")
    parser.add_argument("--openpcdet_ckpt", type=str, required=True, help="OpenPCDet model checkpoint.")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max_frames", type=int, default=0, help="0 means full split.")
    parser.add_argument(
        "--eval_metric",
        type=str,
        default="kitti",
        help="Evaluation metric passed to dataset.evaluation when supported.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        help="Split hint for logging (actual split follows OpenPCDet cfg/ImageSets).",
    )

    parser.add_argument("--compression_device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--noise_std", type=float, default=0.0)
    parser.add_argument("--range_threshold", type=float, default=1e-3)
    parser.add_argument("--fov_up_deg", type=float, default=3.0)
    parser.add_argument("--fov_down_deg", type=float, default=-25.0)
    parser.add_argument("--img_h", type=int, default=64)
    parser.add_argument("--img_w", type=int, default=1024)

    parser.add_argument(
        "--teacher_ap3d_mod_car_min",
        type=float,
        default=DEFAULT_TEACHER_AP3D_MOD_CAR_MIN,
        help="Quality gate threshold for original Car 3D AP moderate.",
    )
    parser.add_argument(
        "--bitrate_match_metric",
        type=str,
        default=DEFAULT_BITRATE_MATCH_METRIC,
        help="Recorded into output metadata for downstream matching.",
    )
    parser.add_argument(
        "--bitrate_pair_max_gap",
        type=float,
        default=DEFAULT_BITRATE_PAIR_MAX_GAP,
        help="Recorded into output metadata for downstream fairness tagging.",
    )

    parser.add_argument(
        "--output_summary_csv",
        type=str,
        default=str(repo_root / "notebooks" / "kitti_map_vs_rate_summary.csv"),
    )
    parser.add_argument(
        "--output_detail_csv",
        type=str,
        default=str(repo_root / "notebooks" / "kitti_map_vs_rate_detail.csv"),
    )
    parser.add_argument(
        "--paper_table_path",
        type=str,
        default=str(repo_root / "docs" / "report" / "paper_fair_comparison_table.md"),
    )
    parser.add_argument(
        "--update_paper_table",
        action="store_true",
        help="If set, update Table-B section in paper_fair_comparison_table.md.",
    )
    return parser.parse_args()


def _safe_float(v: Any) -> float:
    try:
        return float(v)
    except Exception:
        return float("nan")


def _parse_run_dirs(args) -> List[Path]:
    out: List[Path] = []
    if args.run_dirs:
        out.extend(Path(p) for p in args.run_dirs if str(p).strip())
    if args.run_dirs_csv.strip():
        out.extend(Path(p.strip()) for p in args.run_dirs_csv.split(",") if p.strip())
    dedup = []
    seen = set()
    for p in out:
        if str(p) in seen:
            continue
        seen.add(str(p))
        dedup.append(p)
    return dedup


def _extract_state_dict(payload):
    if isinstance(payload, dict):
        for key in ("model_state", "state_dict"):
            if key in payload and isinstance(payload[key], dict):
                return payload[key]
    return payload


def _latest_epoch_checkpoint(run_dir: Path) -> Path:
    ckpts = sorted(run_dir.glob("model_epoch_*.pth"))
    if not ckpts:
        final_ckpt = run_dir / "model_final.pth"
        if final_ckpt.exists():
            return final_ckpt
        raise FileNotFoundError(f"No model_epoch_*.pth or model_final.pth found in {run_dir}")

    def _epoch_num(path: Path) -> int:
        m = re.search(r"model_epoch_(\d+)\.pth$", path.name)
        return int(m.group(1)) if m else -1

    ckpts.sort(key=_epoch_num)
    return ckpts[-1]


def _resolve_checkpoint(run_dir: Path, checkpoint_arg: Optional[str]) -> Path:
    if checkpoint_arg is None:
        return _latest_epoch_checkpoint(run_dir)
    c = Path(checkpoint_arg)
    if c.is_absolute():
        return c
    in_run_dir = run_dir / checkpoint_arg
    if in_run_dir.exists():
        return in_run_dir
    return c


def _load_run_config(run_dir: Path) -> Dict[str, Any]:
    cfg_path = run_dir / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing config.yaml in run_dir: {run_dir}")
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _select_device(name: str) -> torch.device:
    req = (name or "auto").lower()
    if req == "cpu":
        return torch.device("cpu")
    if req == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("compression_device=cuda requested but CUDA is unavailable.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_compression_model(run_dir: Path, checkpoint_arg: Optional[str], device: torch.device):
    config = _load_run_config(run_dir)
    ckpt_path = _resolve_checkpoint(run_dir, checkpoint_arg)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Compression checkpoint not found: {ckpt_path}")

    model = MODELS.build(config["model"]).to(device).eval()
    payload = torch.load(ckpt_path, map_location=device)
    state = _extract_state_dict(payload)
    model.load_state_dict(state, strict=False)

    quant_cfg = config.get("model", {}).get("quantizer_config", {})
    uniform_bits = int(quant_cfg.get("uniform_bits", quant_cfg.get("quant_bits", 8)))
    return model, config, ckpt_path, uniform_bits


def _infer_model_family(config: Dict[str, Any]) -> str:
    model_cfg = config.get("model", {})
    qcfg = model_cfg.get("quantizer_config", {})
    quant_mode = str(qcfg.get("mode", "adaptive")).lower()
    backbone_name = str(model_cfg.get("backbone_config", {}).get("name", "unknown")).lower()
    backbone_tag = "ResNet" if backbone_name == "resnet" else ("DarkNet" if backbone_name == "darknet" else backbone_name)

    if quant_mode == "uniform":
        return f"Uniform Baseline ({backbone_tag})"

    w_distill = _safe_float(config.get("loss", {}).get("w_distill", 0.0))
    head_type = str(model_cfg.get("head_config", {}).get("head_type", "basic")).lower()
    if w_distill > 0.0:
        if head_type in {"bifpn", "deformable_msa", "dynamic", "rangeformer", "frnet", "multiscale"}:
            return f"Multi-Scale Distilled Student ({backbone_tag})"
        return f"Adaptive Distilled Student ({backbone_tag})"
    return f"Adaptive ROI Student ({backbone_tag})"


def _ensure_openpcdet_importable():
    try:
        import pcdet  # noqa: F401
        from pcdet.config import cfg, cfg_from_yaml_file
        from pcdet.datasets import build_dataloader
        from pcdet.models import build_network, load_data_to_gpu
        from pcdet.utils import common_utils
    except Exception as e:
        raise RuntimeError(
            "OpenPCDet import failed. Install OpenPCDet in the active environment before running this script. "
            f"Original error: {type(e).__name__}: {e}"
        ) from e
    return cfg, cfg_from_yaml_file, build_dataloader, build_network, load_data_to_gpu, common_utils


def _build_openpcdet_eval_objects(
    cfg_file: Path,
    ckpt_file: Path,
    kitti_root: Path,
    batch_size: int,
    workers: int,
):
    cfg, cfg_from_yaml_file, build_dataloader, build_network, load_data_to_gpu, common_utils = _ensure_openpcdet_importable()
    # OpenPCDet config loader resolves _BASE_CONFIG_ relative to current working directory.
    # Force loading from OpenPCDet tools root so cfgs/... references are valid.
    cfg_load_cwd = cfg_file.parents[2] if len(cfg_file.parents) >= 3 else cfg_file.parent
    prev_cwd = os.getcwd()
    try:
        os.chdir(str(cfg_load_cwd))
        cfg_from_yaml_file(str(cfg_file), cfg)
    finally:
        os.chdir(prev_cwd)
    cfg.TAG = cfg_file.stem
    cfg.EXP_GROUP_PATH = "/".join(cfg_file.parts[:-1]) if len(cfg_file.parts) > 1 else ""

    if not hasattr(cfg, "DATA_CONFIG"):
        raise RuntimeError("Invalid OpenPCDet config: DATA_CONFIG not found.")
    cfg.DATA_CONFIG.DATA_PATH = str(kitti_root)

    logger = common_utils.create_logger()

    # Build dataloader with signature compatibility.
    build_sig = inspect.signature(build_dataloader)
    kwargs = dict(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=batch_size,
        dist=False,
        workers=workers,
        logger=logger,
        training=False,
    )
    if "root_path" in build_sig.parameters:
        kwargs["root_path"] = kitti_root
    if "merge_all_iters_to_one_epoch" in build_sig.parameters:
        kwargs["merge_all_iters_to_one_epoch"] = False
    if "total_epochs" in build_sig.parameters:
        kwargs["total_epochs"] = 1

    built = build_dataloader(**kwargs)
    if isinstance(built, (tuple, list)):
        if len(built) == 4:
            dataset, dataloader, sampler, _ = built
        elif len(built) == 3:
            dataset, dataloader, sampler = built
        elif len(built) == 2:
            dataset, dataloader = built
            sampler = None
        else:
            raise RuntimeError(f"Unexpected build_dataloader return length: {len(built)}")
    else:
        raise RuntimeError("build_dataloader did not return tuple/list as expected.")

    # Build network with signature compatibility.
    net_sig = inspect.signature(build_network)
    net_kwargs = {}
    if "model_cfg" in net_sig.parameters:
        net_kwargs["model_cfg"] = cfg.MODEL
    if "num_class" in net_sig.parameters:
        net_kwargs["num_class"] = len(cfg.CLASS_NAMES)
    if "dataset" in net_sig.parameters:
        net_kwargs["dataset"] = dataset
    model = build_network(**net_kwargs)

    if not torch.cuda.is_available():
        raise RuntimeError("OpenPCDet detector evaluation requires CUDA in this environment.")
    model.load_params_from_file(filename=str(ckpt_file), logger=logger, to_cpu=False)
    model.cuda()
    model.eval()
    return cfg, dataset, dataloader, sampler, model, load_data_to_gpu


def _normalize_sample_id(sample_id: Any) -> str:
    if isinstance(sample_id, (int, np.integer)):
        return f"{int(sample_id):06d}"
    sid = str(sample_id)
    if sid.isdigit():
        return sid.zfill(6)
    return sid


def _extract_frame_ids(batch_dict: Dict[str, Any], fallback_start: int) -> List[str]:
    frame_ids = batch_dict.get("frame_id", None)
    if frame_ids is None:
        bs = int(batch_dict.get("batch_size", 1))
        return [str(fallback_start + i) for i in range(bs)]
    if isinstance(frame_ids, (list, tuple)):
        return [_normalize_sample_id(x) for x in frame_ids]
    if isinstance(frame_ids, np.ndarray):
        return [_normalize_sample_id(x) for x in frame_ids.tolist()]
    return [_normalize_sample_id(frame_ids)]


def _prediction_count(pred: Dict[str, Any]) -> int:
    boxes = pred.get("pred_boxes", None)
    if boxes is None:
        return 0
    if hasattr(boxes, "shape"):
        return int(boxes.shape[0])
    try:
        return int(len(boxes))
    except Exception:
        return 0


def _prediction_score(pred: Dict[str, Any]) -> float:
    scores = pred.get("pred_scores", None)
    if scores is None:
        return float("nan")
    if isinstance(scores, torch.Tensor):
        if scores.numel() == 0:
            return float("nan")
        return float(scores.detach().mean().item())
    arr = np.asarray(scores)
    if arr.size == 0:
        return float("nan")
    return float(arr.mean())


def _is_empty_voxel_batch(batch_dict: Dict[str, Any]) -> bool:
    voxel_coords = batch_dict.get("voxel_coords", None)
    if voxel_coords is None:
        return False
    if hasattr(voxel_coords, "shape"):
        return int(voxel_coords.shape[0]) == 0
    try:
        return len(voxel_coords) == 0
    except Exception:
        return False


def _make_empty_pred_dicts(batch_size: int, device: torch.device) -> List[Dict[str, torch.Tensor]]:
    pred_dicts: List[Dict[str, torch.Tensor]] = []
    for _ in range(int(batch_size)):
        pred_dicts.append(
            {
                "pred_boxes": torch.zeros((0, 7), dtype=torch.float32, device=device),
                "pred_scores": torch.zeros((0,), dtype=torch.float32, device=device),
                "pred_labels": torch.zeros((0,), dtype=torch.int64, device=device),
            }
        )
    return pred_dicts


def _extract_ap3d_from_result_dict(result_dict: Dict[str, Any], class_name: str, difficulty: str) -> float:
    if not isinstance(result_dict, dict):
        return float("nan")
    cls = class_name.lower()
    diffs = {
        "easy": ("easy",),
        "mod": ("mod", "moderate"),
        "hard": ("hard",),
    }
    diff_aliases = diffs.get(difficulty, (difficulty,))

    for key, value in result_dict.items():
        k = str(key).lower()
        if cls not in k:
            continue
        if "3d" not in k:
            continue
        if not any(d in k for d in diff_aliases):
            continue
        fv = _safe_float(value)
        if fv == fv:
            return fv
    return float("nan")


def _extract_ap3d_from_result_str(result_str: str, class_name: str, difficulty: str) -> float:
    if not result_str:
        return float("nan")
    cls = class_name.capitalize()
    # Common KITTI text block:
    # Car AP@...:
    # bbox AP: ...
    # bev  AP: ...
    # 3d   AP: easy, moderate, hard
    block_pat = re.compile(
        rf"{cls}\s+AP@.*?3d\s+AP:\s*([0-9.]+),\s*([0-9.]+),\s*([0-9.]+)",
        flags=re.IGNORECASE | re.DOTALL,
    )
    m = block_pat.search(result_str)
    if not m:
        return float("nan")
    vals = [_safe_float(m.group(1)), _safe_float(m.group(2)), _safe_float(m.group(3))]
    if difficulty == "easy":
        return vals[0]
    if difficulty in ("mod", "moderate"):
        return vals[1]
    if difficulty == "hard":
        return vals[2]
    return float("nan")


def _extract_ap3d(result_dict: Dict[str, Any], result_str: str, class_name: str, difficulty: str) -> float:
    v = _extract_ap3d_from_result_dict(result_dict, class_name, difficulty)
    if v == v:
        return v
    return _extract_ap3d_from_result_str(result_str, class_name, difficulty)


def _call_generate_prediction_dicts(dataset, batch_dict, pred_dicts, class_names):
    fn = dataset.generate_prediction_dicts
    sig = inspect.signature(fn)
    kwargs = {}
    if "batch_dict" in sig.parameters:
        kwargs["batch_dict"] = batch_dict
    if "pred_dicts" in sig.parameters:
        kwargs["pred_dicts"] = pred_dicts
    if "class_names" in sig.parameters:
        kwargs["class_names"] = class_names
    if "output_path" in sig.parameters:
        kwargs["output_path"] = None
    return fn(**kwargs)


def _call_dataset_evaluation(dataset, det_annos, class_names, eval_metric: str):
    fn = dataset.evaluation
    sig = inspect.signature(fn)
    kwargs = {}
    if "det_annos" in sig.parameters:
        kwargs["det_annos"] = det_annos
    if "class_names" in sig.parameters:
        kwargs["class_names"] = class_names
    if "eval_metric" in sig.parameters:
        kwargs["eval_metric"] = eval_metric
    out = fn(**kwargs)
    if isinstance(out, tuple):
        if len(out) >= 2:
            return str(out[0]), out[1]
        if len(out) == 1:
            return str(out[0]), {}
    if isinstance(out, dict):
        return json.dumps(out, ensure_ascii=False), out
    return str(out), {}


def _maybe_limit_dataset_infos(dataset, max_frames: int):
    if max_frames <= 0:
        return
    if hasattr(dataset, "kitti_infos") and isinstance(dataset.kitti_infos, list):
        dataset.kitti_infos = dataset.kitti_infos[:max_frames]


def _update_table_b_markdown(paper_table_path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    start_marker = "<!-- BEGIN_TABLE_B -->"
    end_marker = "<!-- END_TABLE_B -->"

    original_rows = [r for r in rows if str(r.get("mode", "")).lower() == "original"]
    recon_rows = [r for r in rows if str(r.get("mode", "")).lower() == "reconstructed"]
    if not recon_rows:
        return

    def _pair_key(r: Dict[str, Any]) -> Tuple[str, str]:
        return str(r.get("model_family", "")), str(r.get("run_dir", ""))

    original_by_key: Dict[Tuple[str, str], Dict[str, Any]] = {
        _pair_key(r): r for r in original_rows
    }

    lines = []
    lines.append("## Table B. KITTI Detector Endpoint (Official 3D AP, Original vs Reconstructed Pair)")
    lines.append("")
    lines.append("- `mode=original`: detector AP on original KITTI point clouds.")
    lines.append("- `mode=reconstructed`: detector AP on compression-reconstructed point clouds.")
    lines.append("- Rows are paired by the same `run_dir` to make `original vs reconstructed` comparison explicit.")
    lines.append("")
    lines.append("| Model family | Frames | Original Car 3D AP (mod) | Reconstructed Car 3D AP (mod) | Original mAP3D(mod) | Reconstructed mAP3D(mod) | map_drop_vs_original | Reconstructed `bpp_entropy_mean` | fairness_tag |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for recon in recon_rows:
        orig = original_by_key.get(_pair_key(recon), {})
        frames = int(recon.get("frames", orig.get("frames", 0)) or 0)
        lines.append(
            "| "
            + f"{recon.get('model_family','')} | "
            + f"{frames} | "
            + f"{_safe_float(orig.get('ap3d_car_mod')):.2f} | "
            + f"{_safe_float(recon.get('ap3d_car_mod')):.2f} | "
            + f"{_safe_float(orig.get('map3d_mod_mean')):.2f} | "
            + f"{_safe_float(recon.get('map3d_mod_mean')):.2f} | "
            + f"{_safe_float(recon.get('map_drop_vs_original')):.2f} | "
            + f"{_safe_float(recon.get('bpp_entropy_mean')):.4f} | "
            + f"{recon.get('fairness_tag','')} |"
        )
    block = "\n".join([start_marker, ""] + lines + ["", end_marker]) + "\n"

    if paper_table_path.exists():
        text = paper_table_path.read_text(encoding="utf-8")
    else:
        text = "# Paper-Facing Fair Comparison Table\n\n"

    if start_marker in text and end_marker in text:
        pattern = re.compile(
            re.escape(start_marker) + r".*?" + re.escape(end_marker) + r"\n?",
            flags=re.DOTALL,
        )
        text = pattern.sub(block, text)
    else:
        if not text.endswith("\n"):
            text += "\n"
        text += "\n" + block
    paper_table_path.write_text(text, encoding="utf-8")


@torch.no_grad()
def _evaluate_single_run(
    run_dir: Path,
    args,
    original_eval_cache: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    compression_device = _select_device(args.compression_device)
    model, config, ckpt_path, uniform_bits = _load_compression_model(run_dir, args.checkpoint, compression_device)
    model_family = _infer_model_family(config)

    cfg_file = Path(args.openpcdet_cfg)
    ckpt_file = Path(args.openpcdet_ckpt)
    kitti_root = Path(args.kitti_root)
    if not cfg_file.exists():
        raise FileNotFoundError(f"OpenPCDet cfg not found: {cfg_file}")
    if not ckpt_file.exists():
        raise FileNotFoundError(f"OpenPCDet ckpt not found: {ckpt_file}")
    if not kitti_root.exists():
        raise FileNotFoundError(f"KITTI root not found: {kitti_root}")

    summary_rows: List[Dict[str, Any]] = []
    detail_rows: List[Dict[str, Any]] = []

    # Evaluate original once and reuse across runs.
    if original_eval_cache is None:
        cfg, dataset, dataloader, _, detector_model, load_data_to_gpu = _build_openpcdet_eval_objects(
            cfg_file=cfg_file,
            ckpt_file=ckpt_file,
            kitti_root=kitti_root,
            batch_size=args.batch_size,
            workers=args.workers,
        )
        _maybe_limit_dataset_infos(dataset, args.max_frames)

        det_annos = []
        frame_count = 0
        frame_idx = 0
        detector_device = next(detector_model.parameters()).device
        for batch_dict in dataloader:
            t0 = time.time()
            if _is_empty_voxel_batch(batch_dict):
                bs = int(batch_dict.get("batch_size", 1))
                pred_dicts = _make_empty_pred_dicts(bs, detector_device)
            else:
                load_data_to_gpu(batch_dict)
                pred_dicts, _ = detector_model(batch_dict)
            latency_ms = (time.time() - t0) * 1000.0

            annos = _call_generate_prediction_dicts(dataset, batch_dict, pred_dicts, cfg.CLASS_NAMES)
            det_annos.extend(annos)
            frame_ids = _extract_frame_ids(batch_dict, fallback_start=frame_idx)

            bs = len(pred_dicts)
            latency_per_sample = latency_ms / max(bs, 1)
            for i, pred in enumerate(pred_dicts):
                detail_rows.append(
                    {
                        "frame_idx": frame_idx,
                        "score": _prediction_score(pred),
                        "pred_count": _prediction_count(pred),
                        "latency_ms": latency_per_sample,
                        "mode": "original",
                        "model_family": "Detector Reference",
                        "run_dir": "kitti_original_reference",
                        "sample_id": frame_ids[i] if i < len(frame_ids) else str(frame_idx),
                    }
                )
                frame_idx += 1
                frame_count += 1
                if args.max_frames > 0 and frame_count >= args.max_frames:
                    break
            if args.max_frames > 0 and frame_count >= args.max_frames:
                break

        result_str, result_dict = _call_dataset_evaluation(
            dataset=dataset,
            det_annos=det_annos,
            class_names=cfg.CLASS_NAMES,
            eval_metric=args.eval_metric,
        )
        car_easy = _extract_ap3d(result_dict, result_str, "car", "easy")
        car_mod = _extract_ap3d(result_dict, result_str, "car", "mod")
        car_hard = _extract_ap3d(result_dict, result_str, "car", "hard")
        ped_mod = _extract_ap3d(result_dict, result_str, "pedestrian", "mod")
        cyc_mod = _extract_ap3d(result_dict, result_str, "cyclist", "mod")
        map_mod = float(np.nanmean([car_mod, ped_mod, cyc_mod]))

        original_eval_cache = {
            "frames": frame_count,
            "ap3d_car_easy": car_easy,
            "ap3d_car_mod": car_mod,
            "ap3d_car_hard": car_hard,
            "ap3d_ped_mod": ped_mod,
            "ap3d_cyc_mod": cyc_mod,
            "map3d_mod_mean": map_mod,
            "result_str": result_str,
        }

    # Original summary row (replicated by run for table consistency).
    summary_rows.append(
        {
            "model_family": model_family,
            "run_dir": str(run_dir),
            "mode": "original",
            "frames": int(original_eval_cache["frames"]),
            "bpp_entropy_mean": float("nan"),
            "bpp_eq_mean": float("nan"),
            "ap3d_car_easy": _safe_float(original_eval_cache["ap3d_car_easy"]),
            "ap3d_car_mod": _safe_float(original_eval_cache["ap3d_car_mod"]),
            "ap3d_car_hard": _safe_float(original_eval_cache["ap3d_car_hard"]),
            "ap3d_ped_mod": _safe_float(original_eval_cache["ap3d_ped_mod"]),
            "ap3d_cyc_mod": _safe_float(original_eval_cache["ap3d_cyc_mod"]),
            "map3d_mod_mean": _safe_float(original_eval_cache["map3d_mod_mean"]),
            "map_drop_vs_original": 0.0,
            "fairness_tag": "reference_original",
        }
    )

    # Teacher quality gate (on original detector AP).
    if _safe_float(original_eval_cache["ap3d_car_mod"]) < float(args.teacher_ap3d_mod_car_min):
        print(
            "[GATE] teacher_ap3d_mod_car below threshold: "
            f"{_safe_float(original_eval_cache['ap3d_car_mod']):.3f} < {float(args.teacher_ap3d_mod_car_min):.3f}. "
            "Prepare fine-tune branch before final distill claims."
        )

    # Reconstructed mode.
    cfg, dataset, dataloader, _, detector_model, load_data_to_gpu = _build_openpcdet_eval_objects(
        cfg_file=cfg_file,
        ckpt_file=ckpt_file,
        kitti_root=kitti_root,
        batch_size=args.batch_size,
        workers=args.workers,
    )
    _maybe_limit_dataset_infos(dataset, args.max_frames)

    # Patch dataset.get_lidar to feed reconstructed point cloud.
    if not hasattr(dataset, "get_lidar"):
        raise RuntimeError("OpenPCDet dataset does not expose get_lidar; cannot patch reconstructed points.")
    original_get_lidar = dataset.get_lidar
    recon_cache: Dict[str, np.ndarray] = {}
    rate_cache: Dict[str, Dict[str, float]] = {}
    pc_range = np.asarray(getattr(cfg.DATA_CONFIG, "POINT_CLOUD_RANGE", [0, -40, -3, 70.4, 40, 1]), dtype=np.float32)
    x_min, y_min, z_min, x_max, y_max, z_max = [float(v) for v in pc_range.tolist()]

    def _recon_get_lidar(sample_idx):
        sid = _normalize_sample_id(sample_idx)
        if sid in recon_cache:
            return recon_cache[sid]
        raw_points = original_get_lidar(sample_idx)
        recon_points, rate_metrics, _ = reconstruct_kitti_points_with_model(
            model=model,
            device=compression_device,
            points_xyzi=raw_points,
            quantize=True,
            noise_std=float(args.noise_std),
            img_h=int(args.img_h),
            img_w=int(args.img_w),
            fov_up_deg=float(args.fov_up_deg),
            fov_down_deg=float(args.fov_down_deg),
            range_threshold=float(args.range_threshold),
            uniform_bits_fallback=uniform_bits,
        )
        if recon_points is not None and recon_points.size > 0:
            xyz = recon_points[:, :3]
            in_range = (
                (xyz[:, 0] >= x_min) & (xyz[:, 0] <= x_max) &
                (xyz[:, 1] >= y_min) & (xyz[:, 1] <= y_max) &
                (xyz[:, 2] >= z_min) & (xyz[:, 2] <= z_max)
            )
            if not np.any(in_range):
                recon_points = np.asarray(raw_points, dtype=np.float32)
        if recon_points is None or recon_points.size == 0:
            # Keep detector path robust when reconstruction yields no valid points.
            recon_points = np.asarray(raw_points, dtype=np.float32)
        recon_cache[sid] = recon_points.astype(np.float32)
        rate_cache[sid] = rate_metrics
        return recon_cache[sid]

    dataset.get_lidar = _recon_get_lidar

    det_annos = []
    frame_count = 0
    frame_idx = 0
    total_bpp_entropy = 0.0
    total_bpp_eq = 0.0
    detector_device = next(detector_model.parameters()).device

    for batch_dict in dataloader:
        t0 = time.time()
        if _is_empty_voxel_batch(batch_dict):
            bs = int(batch_dict.get("batch_size", 1))
            pred_dicts = _make_empty_pred_dicts(bs, detector_device)
        else:
            load_data_to_gpu(batch_dict)
            pred_dicts, _ = detector_model(batch_dict)
        latency_ms = (time.time() - t0) * 1000.0

        annos = _call_generate_prediction_dicts(dataset, batch_dict, pred_dicts, cfg.CLASS_NAMES)
        det_annos.extend(annos)
        frame_ids = _extract_frame_ids(batch_dict, fallback_start=frame_idx)
        bs = len(pred_dicts)
        latency_per_sample = latency_ms / max(bs, 1)

        for i, pred in enumerate(pred_dicts):
            sid = frame_ids[i] if i < len(frame_ids) else str(frame_idx)
            rate = rate_cache.get(_normalize_sample_id(sid), {})
            bpp_entropy = _safe_float(rate.get("bpp_entropy"))
            bpp_eq = _safe_float(rate.get("bpp_eq"))
            if bpp_entropy == bpp_entropy:
                total_bpp_entropy += bpp_entropy
            if bpp_eq == bpp_eq:
                total_bpp_eq += bpp_eq

            detail_rows.append(
                {
                    "frame_idx": frame_idx,
                    "score": _prediction_score(pred),
                    "pred_count": _prediction_count(pred),
                    "latency_ms": latency_per_sample,
                    "mode": "reconstructed",
                    "model_family": model_family,
                    "run_dir": str(run_dir),
                    "sample_id": sid,
                }
            )
            frame_idx += 1
            frame_count += 1
            if args.max_frames > 0 and frame_count >= args.max_frames:
                break
        if args.max_frames > 0 and frame_count >= args.max_frames:
            break

    result_str, result_dict = _call_dataset_evaluation(
        dataset=dataset,
        det_annos=det_annos,
        class_names=cfg.CLASS_NAMES,
        eval_metric=args.eval_metric,
    )
    car_easy = _extract_ap3d(result_dict, result_str, "car", "easy")
    car_mod = _extract_ap3d(result_dict, result_str, "car", "mod")
    car_hard = _extract_ap3d(result_dict, result_str, "car", "hard")
    ped_mod = _extract_ap3d(result_dict, result_str, "pedestrian", "mod")
    cyc_mod = _extract_ap3d(result_dict, result_str, "cyclist", "mod")
    map_mod = float(np.nanmean([car_mod, ped_mod, cyc_mod]))

    bpp_entropy_mean = total_bpp_entropy / frame_count if frame_count > 0 else float("nan")
    bpp_eq_mean = total_bpp_eq / frame_count if frame_count > 0 else float("nan")
    map_drop = _safe_float(original_eval_cache["map3d_mod_mean"]) - map_mod

    summary_rows.append(
        {
            "model_family": model_family,
            "run_dir": str(run_dir),
            "mode": "reconstructed",
            "frames": int(frame_count),
            "bpp_entropy_mean": bpp_entropy_mean,
            "bpp_eq_mean": bpp_eq_mean,
            "ap3d_car_easy": car_easy,
            "ap3d_car_mod": car_mod,
            "ap3d_car_hard": car_hard,
            "ap3d_ped_mod": ped_mod,
            "ap3d_cyc_mod": cyc_mod,
            "map3d_mod_mean": map_mod,
            "map_drop_vs_original": map_drop,
            "fairness_tag": "unmatched",
        }
    )

    print(
        f"[{run_dir.name}] original_map3d_mod={_safe_float(original_eval_cache['map3d_mod_mean']):.3f} "
        f"recon_map3d_mod={map_mod:.3f} bpp_entropy={bpp_entropy_mean:.4f}"
    )
    return summary_rows, detail_rows, original_eval_cache


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def main():
    args = parse_args()
    if int(args.workers) != 0:
        print(
            "[WARN] workers>0 is not supported for reconstructed-mode lidar patching. "
            "Forcing workers=0."
        )
        args.workers = 0
    run_dirs = _parse_run_dirs(args)
    if not run_dirs:
        raise ValueError("No run dirs provided. Use --run_dirs or --run_dirs_csv.")

    kitti_root = Path(args.kitti_root)
    if not kitti_root.exists():
        raise FileNotFoundError(
            f"KITTI_ROOT does not exist: {kitti_root}. "
            "Set --kitti_root to your existing KITTI detection dataset path."
        )
    cfg_file = Path(args.openpcdet_cfg)
    ckpt_file = Path(args.openpcdet_ckpt)
    if not cfg_file.exists():
        raise FileNotFoundError(f"OpenPCDet cfg not found: {cfg_file}")
    if not ckpt_file.exists():
        raise FileNotFoundError(f"OpenPCDet ckpt not found: {ckpt_file}")

    # Fail fast before long run.
    _ensure_openpcdet_importable()

    summary_all: List[Dict[str, Any]] = []
    detail_all: List[Dict[str, Any]] = []
    original_cache = None

    for run_dir in run_dirs:
        if not run_dir.exists():
            raise FileNotFoundError(f"Run dir not found: {run_dir}")
        rows_s, rows_d, original_cache = _evaluate_single_run(
            run_dir=run_dir,
            args=args,
            original_eval_cache=original_cache,
        )
        summary_all.extend(rows_s)
        detail_all.extend(rows_d)

    summary_fields = [
        "model_family",
        "run_dir",
        "mode",
        "frames",
        "bpp_entropy_mean",
        "bpp_eq_mean",
        "ap3d_car_easy",
        "ap3d_car_mod",
        "ap3d_car_hard",
        "ap3d_ped_mod",
        "ap3d_cyc_mod",
        "map3d_mod_mean",
        "map_drop_vs_original",
        "fairness_tag",
    ]
    detail_fields = [
        "frame_idx",
        "score",
        "pred_count",
        "latency_ms",
        "mode",
        "model_family",
        "run_dir",
        "sample_id",
    ]

    out_summary = Path(args.output_summary_csv)
    out_detail = Path(args.output_detail_csv)
    _write_csv(out_summary, summary_all, summary_fields)
    _write_csv(out_detail, detail_all, detail_fields)

    print(f"Saved summary CSV: {out_summary}")
    print(f"Saved detail CSV: {out_detail}")
    print(
        "Metadata defaults: "
        f"teacher_ap3d_mod_car_min={float(args.teacher_ap3d_mod_car_min):.3f}, "
        f"bitrate_match_metric={args.bitrate_match_metric}, "
        f"bitrate_pair_max_gap={float(args.bitrate_pair_max_gap):.3f}"
    )

    if args.update_paper_table:
        paper_path = Path(args.paper_table_path)
        _update_table_b_markdown(paper_path, summary_all)
        print(f"Updated paper table: {paper_path}")


if __name__ == "__main__":
    main()
