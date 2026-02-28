import argparse
import csv
from pathlib import Path
from typing import Dict, List, Sequence


def parse_args():
    parser = argparse.ArgumentParser(
        description="Match detector reconstructed rows to uniform references by bitrate."
    )
    parser.add_argument(
        "--inputs",
        type=str,
        nargs="+",
        required=True,
        help="List of kitti_map_vs_rate_summary.csv files.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="bpp_entropy_mean",
        choices=("bpp_entropy_mean", "bpp_eq_mean"),
    )
    parser.add_argument(
        "--max_gap",
        type=float,
        default=0.05,
        help="Primary bitrate gap threshold for fairness tag.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="notebooks/kitti_map_vs_rate_pairs.csv",
    )
    return parser.parse_args()


def _safe_float(v):
    try:
        return float(v)
    except Exception:
        return float("nan")


def _load_rows(paths: Sequence[Path]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for p in paths:
        if not p.exists():
            continue
        with p.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows.extend(reader)
    return rows


def _is_recon_row(row: Dict[str, str]) -> bool:
    return str(row.get("mode", "")).strip().lower() == "reconstructed"


def _is_uniform_family(row: Dict[str, str]) -> bool:
    fam = str(row.get("model_family", "")).strip().lower()
    return fam.startswith("uniform baseline")


def main():
    args = parse_args()
    input_paths = [Path(x) for x in args.inputs]
    rows = _load_rows(input_paths)

    recon_rows = [r for r in rows if _is_recon_row(r)]
    uniform_rows = [r for r in recon_rows if _is_uniform_family(r)]
    candidate_rows = [r for r in recon_rows if not _is_uniform_family(r)]

    out_rows = []
    for cand in candidate_rows:
        cand_metric = _safe_float(cand.get(args.metric))
        cand_eq = _safe_float(cand.get("bpp_eq_mean"))
        if cand_metric != cand_metric:
            continue

        best = None
        best_diff = float("inf")
        best_eq_tiebreak = float("inf")
        for ref in uniform_rows:
            ref_metric = _safe_float(ref.get(args.metric))
            ref_eq = _safe_float(ref.get("bpp_eq_mean"))
            if ref_metric != ref_metric:
                continue
            diff = abs(cand_metric - ref_metric)
            eq_diff = abs(cand_eq - ref_eq) if cand_eq == cand_eq and ref_eq == ref_eq else float("inf")
            if diff < best_diff:
                best = ref
                best_diff = diff
                best_eq_tiebreak = eq_diff
            elif diff == best_diff and eq_diff < best_eq_tiebreak:
                best = ref
                best_eq_tiebreak = eq_diff

        if best is None:
            continue

        ref_metric = _safe_float(best.get(args.metric))
        ref_eq = _safe_float(best.get("bpp_eq_mean"))
        ref_map = _safe_float(best.get("map3d_mod_mean"))
        cand_map = _safe_float(cand.get("map3d_mod_mean"))

        match_type = "exact_match" if abs(best_diff) <= 1e-9 else "nearest_match"
        fairness_tag = "fair" if best_diff <= float(args.max_gap) else "low-fairness"
        combined_tag = f"{match_type}:{fairness_tag}"

        out_rows.append(
            {
                "metric": args.metric,
                "max_gap": float(args.max_gap),
                "candidate_model_family": cand.get("model_family", ""),
                "candidate_run_dir": cand.get("run_dir", ""),
                "candidate_frames": cand.get("frames", ""),
                "candidate_metric": cand_metric,
                "candidate_bpp_eq_mean": cand_eq,
                "candidate_map3d_mod_mean": cand_map,
                "reference_model_family": best.get("model_family", ""),
                "reference_run_dir": best.get("run_dir", ""),
                "reference_frames": best.get("frames", ""),
                "reference_metric": ref_metric,
                "reference_bpp_eq_mean": ref_eq,
                "reference_map3d_mod_mean": ref_map,
                "metric_abs_diff": best_diff,
                "bpp_eq_abs_diff": abs(cand_eq - ref_eq) if cand_eq == cand_eq and ref_eq == ref_eq else float("nan"),
                "map3d_mod_gain_vs_uniform": cand_map - ref_map if cand_map == cand_map and ref_map == ref_map else float("nan"),
                "fairness_tag": combined_tag,
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
        with out_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "metric",
                    "max_gap",
                    "candidate_model_family",
                    "candidate_run_dir",
                    "candidate_frames",
                    "candidate_metric",
                    "candidate_bpp_eq_mean",
                    "candidate_map3d_mod_mean",
                    "reference_model_family",
                    "reference_run_dir",
                    "reference_frames",
                    "reference_metric",
                    "reference_bpp_eq_mean",
                    "reference_map3d_mod_mean",
                    "metric_abs_diff",
                    "bpp_eq_abs_diff",
                    "map3d_mod_gain_vs_uniform",
                    "fairness_tag",
                ]
            )

    print(f"Input files: {len(input_paths)}")
    print(f"Rows loaded: {len(rows)}")
    print(f"Reconstructed rows: {len(recon_rows)}")
    print(f"Uniform references: {len(uniform_rows)}")
    print(f"Candidate rows: {len(candidate_rows)}")
    print(f"Matched pairs: {len(out_rows)}")
    print(f"Saved CSV: {out_path}")


if __name__ == "__main__":
    main()
