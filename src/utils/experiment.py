import hashlib
import json
import os
import platform
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def _now_utc_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _safe_token(value: Any) -> str:
    text = str(value)
    text = text.replace(" ", "")
    text = text.replace("/", "-")
    text = text.replace(":", "")
    text = text.replace(",", "_")
    return "".join(ch for ch in text if ch.isalnum() or ch in ("-", "_", "."))


def short_hash_dict(payload: Dict[str, Any], length: int = 8) -> str:
    data = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha1(data).hexdigest()[:length]


def get_git_rev(repo_dir: Optional[str] = None) -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir,
            stderr=subprocess.DEVNULL,
        )
        return out.decode("utf-8").strip()
    except Exception:
        return None


@dataclass(frozen=True)
class RunPaths:
    run_dir: Path
    checkpoint_path: Path
    manifest_path: Path


def make_run_id(stage: str, name: str, config: Dict[str, Any]) -> str:
    """
    Produce a stable-ish run id that is readable but also collision-resistant.
    Example: s1_ae_q8_ns0.1_lr0.001_bs4_e20_20260210T123000Z_ab12cd34
    """
    compact = _now_utc_compact()
    core = {
        "stage": stage,
        "name": name,
        # include only config keys that meaningfully describe a run
        "config": config,
    }
    h = short_hash_dict(core)

    tokens = [f"s{_safe_token(stage)}", _safe_token(name)]
    keys = (
        ("quant_bits", "q"),
        ("noise_std", "ns"),
        ("lr", "lr"),
        ("batch_size", "bs"),
        ("epochs", "e"),
        ("seed", "sd"),
        ("base_channels", "bc"),
        ("latent_channels", "lc"),
        ("num_stages", "st"),
        ("blocks_per_stage", "bp"),
        ("norm", "nm"),
        ("activation", "ac"),
        ("dropout", "do"),
        ("roi_levels", "rl"),
        ("bg_levels", "bl"),
        ("lambda_task", "lt"),
        ("beta_entropy", "be"),
        ("roi_recon_weight", "rw"),
    )
    for key, prefix in keys:
        if key in config and config[key] is not None:
            tokens.append(f"{prefix}{_safe_token(config[key])}")
    tokens.append(compact)
    tokens.append(h)
    return "_".join(tokens)


def default_run_paths(run_id: str) -> RunPaths:
    # Repo convention: store experiment artifacts under data/results/
    repo_root = Path(__file__).resolve().parents[2]
    run_dir = repo_root / "data" / "results" / "runs" / run_id
    checkpoint_path = run_dir / "checkpoint.pth"
    manifest_path = run_dir / "manifest.json"
    return RunPaths(run_dir=run_dir, checkpoint_path=checkpoint_path, manifest_path=manifest_path)


def write_manifest(
    manifest_path: Path,
    *,
    stage: str,
    run_id: str,
    command: Optional[str],
    config: Dict[str, Any],
    model_info: Dict[str, Any],
    notes: Optional[str] = None,
    metrics: Optional[Dict[str, Any]] = None,
) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "stage": stage,
        "run_id": run_id,
        "created_utc": _now_utc_compact(),
        "command": command,
        "config": config,
        "model_info": model_info,
        "metrics": metrics or {},
        "notes": notes,
        "env": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "hostname": platform.node(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "git": {
            "rev": get_git_rev(),
        },
    }
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
