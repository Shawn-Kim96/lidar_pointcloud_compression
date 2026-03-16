from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect Track 2 sweep outputs from a submission manifest and summarize best metrics."
    )
    parser.add_argument("--manifest", type=str, required=True, help="CSV written by submit_track2_range_roi_pilot.sh")
    parser.add_argument("--output_csv", type=str, default="", help="Optional explicit output CSV path")
    parser.add_argument("--output_md", type=str, default="", help="Optional explicit output Markdown path")
    return parser.parse_args()


def _resolve(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
        return path

    if path.exists():
        return path

    repo_name = Path.cwd().name
    if repo_name in path.parts:
        idx = path.parts.index(repo_name)
        suffix = Path(*path.parts[idx + 1 :])
        alt = (Path.cwd() / suffix).resolve()
        return alt

    if path.name.startswith("slurm_"):
        return (Path.cwd() / "logs" / path.name).resolve()

    return path


def _read_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _status_for(summary_path: Path, slurm_log: Path) -> str:
    if summary_path.exists():
        return "complete"
    if slurm_log.exists():
        return "no_summary"
    return "missing"


def _queue_states(job_ids: List[str]) -> Dict[str, str]:
    ids = [jid for jid in job_ids if jid]
    if not ids:
        return {}
    try:
        out = subprocess.check_output(
            ["squeue", "-j", ",".join(ids), "-h", "-o", "%i,%T"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return {}

    states: Dict[str, str] = {}
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        job_id, state = line.split(",", 1)
        states[job_id.strip()] = state.strip().lower()
    return states


def _err_status(slurm_err: Path) -> str:
    if not slurm_err.exists() or slurm_err.stat().st_size == 0:
        return ""
    try:
        text = slurm_err.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return "failed"
    markers = ("Traceback", "RuntimeError", "CUDA error", "Error:")
    return "failed" if any(mark in text for mark in markers) else "has_err"


def _load_rows(manifest_path: Path) -> List[Dict[str, str]]:
    with manifest_path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _default_output(manifest_path: Path, suffix: str) -> Path:
    stem = manifest_path.stem
    if stem.endswith("_manifest"):
        stem = stem[: -len("_manifest")]
    return manifest_path.with_name(f"{stem}_results.{suffix}")


def main() -> None:
    args = parse_args()
    manifest_path = _resolve(args.manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    rows = _load_rows(manifest_path)
    qstates = _queue_states([row.get("job_id", "") for row in rows])
    result_rows: List[Dict[str, str]] = []

    for row in rows:
        output_dir = _resolve(row["output_dir"])
        summary_path = output_dir / "summary.json"
        slurm_log = _resolve(row["slurm_log"])
        slurm_err = slurm_log.with_suffix(".err")
        status = _status_for(summary_path, slurm_log)
        if row["job_id"] in qstates:
            status = qstates[row["job_id"]]
        elif status != "complete":
            err_status = _err_status(slurm_err)
            if err_status:
                status = err_status

        result = {
            "job_id": row["job_id"],
            "track2_tag": row["track2_tag"],
            "run_name": row["run_name"],
            "head_type": row["head_type"],
            "hidden_channels": row["hidden_channels"],
            "epochs": row["epochs"],
            "lr": row["lr"],
            "weight_decay": row["weight_decay"],
            "batch_size": row["batch_size"],
            "status": status,
            "output_dir": str(output_dir),
            "summary_json": str(summary_path),
            "slurm_log": str(slurm_log),
            "slurm_err": str(slurm_err),
            "best_epoch": "",
            "raw_iou": "",
            "compressed_iou": "",
            "raw_f1": "",
            "compressed_f1": "",
        }

        if summary_path.exists():
            payload = _read_json(summary_path)
            metrics = payload.get("best_metrics", {})
            result["best_epoch"] = str(payload.get("best_epoch", metrics.get("epoch", "")))
            result["raw_iou"] = f"{float(metrics.get('raw_iou', 0.0)):.6f}"
            result["compressed_iou"] = f"{float(metrics.get('compressed_iou', 0.0)):.6f}"
            result["raw_f1"] = f"{float(metrics.get('raw_f1', 0.0)):.6f}"
            result["compressed_f1"] = f"{float(metrics.get('compressed_f1', 0.0)):.6f}"

        result_rows.append(result)

    output_csv = _resolve(args.output_csv) if args.output_csv else _default_output(manifest_path, "csv")
    output_md = _resolve(args.output_md) if args.output_md else _default_output(manifest_path, "md")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(result_rows[0].keys()) if result_rows else [
        "job_id",
        "track2_tag",
        "run_name",
        "head_type",
        "hidden_channels",
        "epochs",
        "lr",
        "weight_decay",
        "batch_size",
        "status",
        "output_dir",
        "summary_json",
        "slurm_log",
        "slurm_err",
        "best_epoch",
        "raw_iou",
        "compressed_iou",
        "raw_f1",
        "compressed_f1",
    ]

    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(result_rows)

    lines = [
        "# Track 2 Sweep Results",
        "",
        f"- Manifest: `{manifest_path}`",
        f"- Rows: `{len(result_rows)}`",
        "",
        "| job_id | run_name | head | hidden | epochs | lr | status | raw_iou | compressed_iou | raw_f1 | compressed_f1 |",
        "| --- | --- | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in result_rows:
        lines.append(
            "| {job_id} | {run_name} | {head_type} | {hidden_channels} | {epochs} | {lr} | {status} | {raw_iou} | {compressed_iou} | {raw_f1} | {compressed_f1} |".format(
                **row
            )
        )

    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[track2-collect] manifest={manifest_path}")
    print(f"[track2-collect] output_csv={output_csv}")
    print(f"[track2-collect] output_md={output_md}")


if __name__ == "__main__":
    main()
