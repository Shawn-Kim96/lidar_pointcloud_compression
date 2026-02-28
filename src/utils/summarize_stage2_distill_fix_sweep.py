#!/usr/bin/env python3
import argparse
import csv
import math
import re
from pathlib import Path
from typing import Dict, List, Optional


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize Stage2 distill-fix two-experiment sweep results.")
    p.add_argument("--input_csv", type=Path, default=Path("src/results/experiments_result.csv"))
    p.add_argument("--job_id", type=str, default="", help="Optional slurm job id filter (matches run_id=j<job>_rX).")
    p.add_argument(
        "--output_csv",
        type=Path,
        default=Path("notebooks/stage2_distill_fix_twoexp_summary.csv"),
    )
    p.add_argument(
        "--output_md",
        type=Path,
        default=Path("docs/report/stage2_distill_fix_twoexp_summary.md"),
    )
    return p.parse_args()


def to_float(v: str) -> float:
    try:
        return float(v)
    except Exception:
        return float("nan")


def fmt(v: float, nd: int = 4) -> str:
    if v != v:
        return "n/a"
    return f"{v:.{nd}f}"


def extract_case(save_dir: str) -> str:
    m = re.search(r"distill_fix2_([^/_]+(?:_[^/_]+)*)_ld", save_dir)
    if m:
        return m.group(1)
    return "unknown"


def extract_lr_tag(save_dir: str) -> str:
    m = re.search(r"_lr([^_]+)_bs", save_dir)
    return m.group(1) if m else ""


def filter_rows(rows: List[Dict[str, str]], job_id: str) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    needle = "distill_fix2_"
    run_prefix = f"j{job_id}_" if job_id else ""
    for r in rows:
        save_dir = r.get("save_dir", "")
        if needle not in save_dir:
            continue
        run_id = r.get("run_id", "")
        if run_prefix and not run_id.startswith(run_prefix):
            continue
        out.append(r)
    return out


def make_summary(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    summary: List[Dict[str, str]] = []
    for r in rows:
        save_dir = r.get("save_dir", "")
        summary.append(
            {
                "case_tag": extract_case(save_dir),
                "run_id": r.get("run_id", ""),
                "backbone": r.get("backbone", ""),
                "lr": r.get("lr", ""),
                "lambda_distill": r.get("lambda_distill", ""),
                "distill_feature_source": r.get("distill_feature_source", ""),
                "distill_align_mode": r.get("distill_align_mode", ""),
                "distill_align_hw": r.get("distill_align_hw", ""),
                "distill_teacher_score_min": r.get("distill_teacher_score_min", ""),
                "epochs": r.get("epochs", ""),
                "best_loss": r.get("best_loss", ""),
                "final_loss": r.get("final_loss", ""),
                "final_rate_proxy": r.get("final_rate_proxy", ""),
                "final_imp_mean": r.get("final_imp_mean", ""),
                "started_at": r.get("started_at", ""),
                "save_dir": save_dir,
                "log_file": r.get("log_file", ""),
            }
        )

    summary.sort(
        key=lambda x: (
            x["case_tag"],
            to_float(x["lambda_distill"]),
            to_float(x["lr"]),
            to_float(x["best_loss"]),
        )
    )
    return summary


def write_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", newline="", encoding="utf-8") as f:
            f.write("case_tag,run_id,backbone,lr,lambda_distill,distill_feature_source,distill_align_mode,distill_align_hw,distill_teacher_score_min,epochs,best_loss,final_loss,final_rate_proxy,final_imp_mean,started_at,save_dir,log_file\n")
        return
    fields = [
        "case_tag",
        "run_id",
        "backbone",
        "lr",
        "lambda_distill",
        "distill_feature_source",
        "distill_align_mode",
        "distill_align_hw",
        "distill_teacher_score_min",
        "epochs",
        "best_loss",
        "final_loss",
        "final_rate_proxy",
        "final_imp_mean",
        "started_at",
        "save_dir",
        "log_file",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def write_md(path: Path, rows: List[Dict[str, str]], job_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    lines.append("# Stage2 Distill-Fix TwoExp Sweep Summary")
    lines.append("")
    if job_id:
        lines.append(f"- Filtered slurm job id: `{job_id}`")
    lines.append(f"- Total matched runs: `{len(rows)}`")
    lines.append("")

    if not rows:
        lines.append("No matched runs found.")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    best_by_case: Dict[str, Dict[str, str]] = {}
    for r in rows:
        case = r["case_tag"]
        cur = best_by_case.get(case)
        if cur is None:
            best_by_case[case] = r
            continue
        if to_float(r.get("best_loss", "nan")) < to_float(cur.get("best_loss", "nan")):
            best_by_case[case] = r

    lines.append("## Best Run Per Case")
    lines.append("")
    lines.append("| Case | run_id | lr | lambda_distill | best_loss | final_loss | final_imp_mean |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for case in sorted(best_by_case.keys()):
        r = best_by_case[case]
        lines.append(
            "| "
            + f"{case} | {r['run_id']} | {r['lr']} | {r['lambda_distill']} | "
            + f"{fmt(to_float(r['best_loss']))} | {fmt(to_float(r['final_loss']))} | {fmt(to_float(r['final_imp_mean']))} |"
        )

    lines.append("")
    lines.append("## Full Sweep")
    lines.append("")
    lines.append("| Case | run_id | lr | lambda_distill | score_min | best_loss | final_loss | final_rate_proxy | final_imp_mean |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for r in rows:
        lines.append(
            "| "
            + f"{r['case_tag']} | {r['run_id']} | {r['lr']} | {r['lambda_distill']} | {r['distill_teacher_score_min']} | "
            + f"{fmt(to_float(r['best_loss']))} | {fmt(to_float(r['final_loss']))} | "
            + f"{fmt(to_float(r['final_rate_proxy']))} | {fmt(to_float(r['final_imp_mean']))} |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not args.input_csv.exists():
        raise FileNotFoundError(f"Missing input CSV: {args.input_csv}")

    with args.input_csv.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    filtered = filter_rows(rows, args.job_id.strip())
    summary = make_summary(filtered)
    write_csv(args.output_csv, summary)
    write_md(args.output_md, summary, args.job_id.strip())

    print(f"input_rows={len(rows)}")
    print(f"matched_rows={len(summary)}")
    print(f"output_csv={args.output_csv}")
    print(f"output_md={args.output_md}")


if __name__ == "__main__":
    main()
