import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple


def parse_args():
    parser = argparse.ArgumentParser(
        description="Match adaptive runs to uniform baselines by closest bitrate budget."
    )
    parser.add_argument(
        "--inputs",
        type=str,
        nargs="+",
        required=True,
        help="List of oracle_eval_summary_*.csv files.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="bpp_entropy_mean",
        choices=("bpp_entropy_mean", "bpp_eq_mean", "eq_bits_mean", "code_entropy_mean"),
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="notebooks/matched_bitrate_pairs.csv",
    )
    return parser.parse_args()


def _safe_float(v):
    try:
        return float(v)
    except Exception:
        return float("nan")


def _load_rows(paths: List[Path]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for p in paths:
        if not p.exists():
            continue
        with p.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append(r)
    return rows


def _split_rows(rows: List[Dict[str, str]]) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    native = [r for r in rows if str(r.get("mode", "")).strip().lower() == "native"]
    uniform = [r for r in native if str(r.get("quantizer_mode", "")).strip().lower() == "uniform"]
    adaptive = [r for r in native if str(r.get("quantizer_mode", "")).strip().lower() == "adaptive"]
    return uniform, adaptive


def main():
    args = parse_args()
    input_paths = [Path(x) for x in args.inputs]
    rows = _load_rows(input_paths)
    uniform_rows, adaptive_rows = _split_rows(rows)

    out_rows: List[Dict[str, object]] = []
    for a in adaptive_rows:
        a_metric = _safe_float(a.get(args.metric))
        if a_metric != a_metric:
            continue
        best = None
        best_diff = float("inf")
        for u in uniform_rows:
            u_metric = _safe_float(u.get(args.metric))
            if u_metric != u_metric:
                continue
            diff = abs(a_metric - u_metric)
            if diff < best_diff:
                best = u
                best_diff = diff
        if best is None:
            continue
        out_rows.append(
            {
                "metric": args.metric,
                "adaptive_run_dir": a.get("run_dir", ""),
                "adaptive_quant_bits": a.get("quant_bits", ""),
                "adaptive_metric": a_metric,
                "adaptive_all_mse": _safe_float(a.get("all_mse_mean")),
                "adaptive_roi_mse": _safe_float(a.get("roi_mse_mean")),
                "adaptive_bg_mse": _safe_float(a.get("bg_mse_mean")),
                "uniform_run_dir": best.get("run_dir", ""),
                "uniform_quant_bits": best.get("quant_bits", ""),
                "uniform_metric": _safe_float(best.get(args.metric)),
                "uniform_all_mse": _safe_float(best.get("all_mse_mean")),
                "uniform_roi_mse": _safe_float(best.get("roi_mse_mean")),
                "uniform_bg_mse": _safe_float(best.get("bg_mse_mean")),
                "metric_abs_diff": best_diff,
                "roi_mse_gain_pct_vs_uniform": (
                    (_safe_float(best.get("roi_mse_mean")) - _safe_float(a.get("roi_mse_mean")))
                    / max(_safe_float(best.get("roi_mse_mean")), 1e-12)
                    * 100.0
                ),
                "all_mse_gain_pct_vs_uniform": (
                    (_safe_float(best.get("all_mse_mean")) - _safe_float(a.get("all_mse_mean")))
                    / max(_safe_float(best.get("all_mse_mean")), 1e-12)
                    * 100.0
                ),
            }
        )

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_rows:
        with out_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
            writer.writeheader()
            writer.writerows(out_rows)
    else:
        out_path.write_text("", encoding="utf-8")

    print(f"Input summary files: {len(input_paths)}")
    print(f"Loaded rows: {len(rows)} (uniform={len(uniform_rows)}, adaptive={len(adaptive_rows)})")
    print(f"Matched pairs: {len(out_rows)}")
    print(f"Saved CSV: {out_path}")


if __name__ == "__main__":
    main()
