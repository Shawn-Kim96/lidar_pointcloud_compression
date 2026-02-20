import argparse
import csv
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple


META_KEYS = [
    "stage",
    "training_mode",
    "backbone",
    "quantizer_mode",
    "quant_bits",
    "teacher_backend",
    "run_id",
    "save_dir",
    "dataset_root",
    "epochs",
    "batch_size",
    "num_workers",
    "lr",
    "roi_levels",
    "bg_levels",
    "roi_target_mode",
    "max_train_frames",
    "loss_recipe",
    "rate_loss_mode",
    "importance_loss_mode",
    "importance_pos_weight_mode",
    "importance_pos_weight",
    "importance_pos_weight_max",
    "imp_separation_margin",
    "distill_logit_loss",
    "distill_temperature",
    "distill_feature_weight",
    "distill_logit_weight",
    "importance_head_type",
    "importance_hidden_channels",
    "teacher_proxy_ckpt",
    "teacher_score_topk_ratio",
    "loss_weights",
    "started_at",
]


def _extract_meta(text: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for key in META_KEYS:
        m = re.search(rf"^{re.escape(key)}:\s*(.+)$", text, re.MULTILINE)
        if m:
            out[key] = m.group(1).strip()
    return out


def _extract_loss_weights(loss_weights: str) -> Dict[str, float]:
    result = {
        "lambda_recon": float("nan"),
        "lambda_rate": float("nan"),
        "lambda_distill": float("nan"),
        "lambda_importance": float("nan"),
        "lambda_imp_separation": float("nan"),
    }
    if not loss_weights:
        return result
    patterns = {
        "lambda_recon": r"recon=([0-9.eE+\-]+)",
        "lambda_rate": r"rate=([0-9.eE+\-]+)",
        "lambda_distill": r"distill=([0-9.eE+\-]+)",
        "lambda_importance": r"importance=([0-9.eE+\-]+)",
        "lambda_imp_separation": r"imp_separation=([0-9.eE+\-]+)",
    }
    for k, p in patterns.items():
        m = re.search(p, loss_weights)
        if m:
            result[k] = float(m.group(1))
    return result


def _parse_float(text: str) -> float:
    try:
        return float(text)
    except Exception:
        return float("nan")


def _extract_epoch_stats(text: str) -> List[Dict[str, object]]:
    """
    Parse epoch lines.
    Supports both legacy format:
      Epoch N: Loss X
    and extended format:
      Epoch N: Loss X | rate_proxy=... | eq_bits=... | code_entropy=... | imp_mean=...
    """
    values: List[Dict[str, object]] = []
    pattern = re.compile(r"^Epoch\s+(\d+):\s+Loss\s+([0-9.eE+\-]+)(?:\s+\|\s+(.*))?$")
    for raw in text.splitlines():
        line = raw.strip()
        m = pattern.match(line)
        if not m:
            continue
        metrics: Dict[str, float] = {}
        tail = m.group(3) or ""
        if tail:
            for token in tail.split("|"):
                token = token.strip()
                if "=" not in token:
                    continue
                key, val = token.split("=", 1)
                metrics[key.strip()] = _parse_float(val.strip())
        values.append(
            {
                "epoch": int(m.group(1)),
                "loss": float(m.group(2)),
                "metrics": metrics,
            }
        )
    return values


def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return float("nan")


def _format_float(v, nd=4):
    if v != v:
        return "n/a"
    return f"{v:.{nd}f}"


def _parse_started_at(value: str):
    # Example: 2026-02-12 22:18:00 PST
    if not value:
        return None
    tokens = value.split()
    if len(tokens) < 2:
        return None
    dt_text = " ".join(tokens[:2])
    try:
        return datetime.strptime(dt_text, "%Y-%m-%d %H:%M:%S")
    except Exception:
        return None


def _describe_quantizer_mode(mode: str) -> str:
    mode_l = str(mode or "").strip().lower()
    if mode_l == "adaptive":
        return (
            "Importance-aware adaptive quantization. "
            "Quantization levels vary by spatial importance map "
            "(bg_levels -> roi_levels interpolation)."
        )
    if mode_l == "uniform":
        return (
            "ROI-unaware uniform quantization. "
            "A fixed bit-depth quantizer is applied to latent values."
        )
    if not mode_l:
        return "Missing value."
    return f"Unrecognized mode '{mode}'."


def _describe_quant_bits(mode: str, bits: str) -> str:
    mode_l = str(mode or "").strip().lower()
    bits_text = str(bits or "").strip()
    if mode_l == "uniform":
        return (
            f"Uniform quantizer bit-depth ({bits_text} bits). "
            "Higher bits increase reconstruction fidelity and usually increase bitrate."
        )
    if mode_l == "adaptive":
        return (
            f"Logged as {bits_text} for compatibility. "
            "Adaptive quantization is primarily controlled by roi_levels/bg_levels, "
            "not by quant_bits directly."
        )
    return f"Bit-depth field value: {bits_text}."


def _describe_lambda_distill(weight: float) -> str:
    if weight != weight:
        return "Missing value."
    if weight <= 0.0:
        return (
            "Distillation loss disabled (teacher matching does not contribute to total loss)."
        )
    return (
        f"Distillation loss active with weight={weight:.3f}. "
        "Higher value increases emphasis on teacher feature/logit matching."
    )


def _describe_loss_recipe(recipe: str) -> str:
    r = str(recipe or "").strip().lower()
    if r == "legacy":
        return "Original objective: raw mean(level_map) rate proxy + BCE importance."
    if r == "balanced_v1":
        return "Normalized rate loss + weighted BCE importance."
    if r == "balanced_v2":
        return "Background-focused normalized rate + weighted BCE + ROI/BG separation margin."
    if not r:
        return "Missing value."
    return f"Custom recipe '{recipe}'."


def collect_runs(log_dir: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    out_files = sorted(p for p in log_dir.glob("*_r*.out") if not p.name.startswith("slurm_"))

    for out_file in out_files:
        text = out_file.read_text(errors="ignore")
        meta = _extract_meta(text)
        epoch_stats = _extract_epoch_stats(text)
        losses = [(int(s["epoch"]), float(s["loss"])) for s in epoch_stats]
        weights = _extract_loss_weights(meta.get("loss_weights", ""))

        roi_target_mode = meta.get("roi_target_mode", "").strip()
        if roi_target_mode:
            roi_target_source = "logged"
        else:
            # Backfill for legacy runs before roi_target_mode existed.
            roi_target_mode = "nearest"
            roi_target_source = "backfilled_legacy_default"

        quantizer_mode = meta.get("quantizer_mode", "").strip()
        if quantizer_mode:
            quantizer_mode_source = "logged"
        else:
            # Legacy runs in this repo used adaptive quantization by default.
            quantizer_mode = "adaptive"
            quantizer_mode_source = "backfilled_legacy_default"

        quant_bits = meta.get("quant_bits", "").strip()
        if quant_bits:
            quant_bits_source = "logged"
        else:
            # Legacy runs used default 8-bit setting where applicable.
            quant_bits = "8"
            quant_bits_source = "backfilled_legacy_default"

        loss_recipe = meta.get("loss_recipe", "").strip().lower()
        if loss_recipe:
            loss_recipe_source = "logged"
        else:
            loss_recipe = "legacy"
            loss_recipe_source = "backfilled_legacy_default"

        rate_loss_mode = meta.get("rate_loss_mode", "").strip().lower()
        if rate_loss_mode:
            rate_loss_mode_source = "logged"
        else:
            if loss_recipe == "balanced_v2":
                rate_loss_mode = "normalized_bg"
            elif loss_recipe == "balanced_v1":
                rate_loss_mode = "normalized_global"
            else:
                rate_loss_mode = "global_mean"
            rate_loss_mode_source = "backfilled_from_recipe_default"

        importance_loss_mode = meta.get("importance_loss_mode", "").strip().lower()
        if importance_loss_mode:
            importance_loss_mode_source = "logged"
        else:
            importance_loss_mode = "weighted_bce" if loss_recipe != "legacy" else "bce"
            importance_loss_mode_source = "backfilled_from_recipe_default"

        importance_head_type = meta.get("importance_head_type", "").strip().lower()
        if importance_head_type:
            importance_head_type_source = "logged"
        else:
            importance_head_type = "basic"
            importance_head_type_source = "backfilled_legacy_default"

        lambda_imp_sep = weights["lambda_imp_separation"]
        if lambda_imp_sep != lambda_imp_sep:
            lambda_imp_sep = 0.0

        first_loss = losses[0][1] if losses else float("nan")
        final_loss = losses[-1][1] if losses else float("nan")
        best_loss = min(v for _, v in losses) if losses else float("nan")
        epoch_count = len(losses)
        rel_improve_pct = ((first_loss - final_loss) / first_loss * 100.0) if losses and first_loss > 0 else float("nan")
        final_metrics = epoch_stats[-1]["metrics"] if epoch_stats else {}
        final_rate_proxy = _safe_float(final_metrics.get("rate_proxy", float("nan")))
        final_eq_bits = _safe_float(final_metrics.get("eq_bits", float("nan")))
        final_code_entropy = _safe_float(final_metrics.get("code_entropy", float("nan")))
        final_imp_mean = _safe_float(final_metrics.get("imp_mean", float("nan")))

        row = {
            "log_file": out_file.name,
            "stage": meta.get("stage", ""),
            "training_mode": meta.get("training_mode", ""),
            "backbone": meta.get("backbone", ""),
            "quantizer_mode": quantizer_mode,
            "quantizer_mode_detail": _describe_quantizer_mode(quantizer_mode),
            "quantizer_mode_source": quantizer_mode_source,
            "quant_bits": quant_bits,
            "quant_bits_detail": _describe_quant_bits(quantizer_mode, quant_bits),
            "quant_bits_source": quant_bits_source,
            "teacher_backend": meta.get("teacher_backend", ""),
            "teacher_proxy_ckpt": meta.get("teacher_proxy_ckpt", "none"),
            "teacher_score_topk_ratio": meta.get("teacher_score_topk_ratio", "0.01"),
            "run_id": meta.get("run_id", ""),
            "started_at": meta.get("started_at", ""),
            "epochs": meta.get("epochs", ""),
            "batch_size": meta.get("batch_size", ""),
            "lr": meta.get("lr", ""),
            "roi_levels": meta.get("roi_levels", ""),
            "bg_levels": meta.get("bg_levels", ""),
            "max_train_frames": meta.get("max_train_frames", "all"),
            "loss_recipe": loss_recipe,
            "loss_recipe_detail": _describe_loss_recipe(loss_recipe),
            "loss_recipe_source": loss_recipe_source,
            "rate_loss_mode": rate_loss_mode,
            "rate_loss_mode_source": rate_loss_mode_source,
            "importance_loss_mode": importance_loss_mode,
            "importance_loss_mode_source": importance_loss_mode_source,
            "importance_pos_weight_mode": meta.get("importance_pos_weight_mode", "auto"),
            "importance_pos_weight": meta.get("importance_pos_weight", "1.0"),
            "importance_pos_weight_max": meta.get("importance_pos_weight_max", "50.0"),
            "distill_logit_loss": meta.get("distill_logit_loss", "auto"),
            "distill_temperature": meta.get("distill_temperature", "1.0"),
            "distill_feature_weight": meta.get("distill_feature_weight", "1.0"),
            "distill_logit_weight": meta.get("distill_logit_weight", "1.0"),
            "importance_head_type": importance_head_type,
            "importance_head_type_source": importance_head_type_source,
            "importance_hidden_channels": meta.get("importance_hidden_channels", "32"),
            "imp_separation_margin": meta.get("imp_separation_margin", "0.05"),
            "roi_target_mode": roi_target_mode,
            "roi_target_mode_source": roi_target_source,
            "lambda_recon": weights["lambda_recon"],
            "lambda_rate": weights["lambda_rate"],
            "lambda_distill": weights["lambda_distill"],
            "lambda_distill_detail": _describe_lambda_distill(weights["lambda_distill"]),
            "lambda_importance": weights["lambda_importance"],
            "lambda_imp_separation": lambda_imp_sep,
            "first_loss": first_loss,
            "final_loss": final_loss,
            "best_loss": best_loss,
            "rel_improve_pct": rel_improve_pct,
            "final_rate_proxy": final_rate_proxy,
            "final_eq_bits": final_eq_bits,
            "final_code_entropy": final_code_entropy,
            "final_imp_mean": final_imp_mean,
            "save_dir": meta.get("save_dir", ""),
        }
        rows.append(row)

    rows.sort(
        key=lambda r: (
            _parse_started_at(r.get("started_at", "")) or datetime.min,
            str(r.get("log_file", "")),
        )
    )
    return rows


def write_csv(rows: List[Dict[str, str]], csv_path: Path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        csv_path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def render_md(rows: List[Dict[str, str]], md_path: Path, log_dir: Path):
    md_path.parent.mkdir(parents=True, exist_ok=True)
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines: List[str] = []
    lines.append("# Experiments Result Ledger")
    lines.append("")
    lines.append(f"- generated_at: `{now_str}`")
    lines.append(f"- source_logs_dir: `{log_dir}`")
    lines.append(f"- total_runs: `{len(rows)}`")
    lines.append("")
    lines.append("## Parameter Policy")
    lines.append("- New hyperparameters must be tracked for all runs.")
    lines.append("- If a parameter did not exist in legacy runs, backfill explicitly and mark source.")
    lines.append("- `roi_target_mode` backfill rule in this ledger:")
    lines.append("  - if logged in metadata -> use logged value")
    lines.append("  - if missing in legacy run -> `nearest` (`backfilled_legacy_default`)")
    lines.append("- `quantizer_mode` backfill rule in this ledger:")
    lines.append("  - if logged in metadata -> use logged value")
    lines.append("  - if missing in legacy run -> `adaptive` (`backfilled_legacy_default`)")
    lines.append("- `quant_bits` backfill rule in this ledger:")
    lines.append("  - if logged in metadata -> use logged value")
    lines.append("  - if missing in legacy run -> `8` (`backfilled_legacy_default`)")
    lines.append("- `loss_recipe` backfill rule in this ledger:")
    lines.append("  - if logged in metadata -> use logged value")
    lines.append("  - if missing in legacy run -> `legacy` (`backfilled_legacy_default`)")
    lines.append("- `rate_loss_mode`/`importance_loss_mode` backfill rule:")
    lines.append("  - if missing -> infer from `loss_recipe` default and mark `backfilled_from_recipe_default`")
    lines.append("")
    lines.append("## Column Guide (Key Fields)")
    lines.append("")
    lines.append("| column | meaning | value semantics |")
    lines.append("|---|---|---|")
    lines.append(
        "| `quantizer_mode` | Quantization method used in the run. | "
        "`adaptive`: importance-aware levels vary spatially by ROI/importance map. "
        "`uniform`: fixed bit-depth quantization across latent tensor (ROI-unaware baseline). |"
    )
    lines.append(
        "| `quant_bits` | Bit-depth field recorded in metadata. | "
        "In `uniform`, directly sets quantizer levels (`2^bits`). "
        "In `adaptive`, this field is compatibility metadata; effective granularity comes from "
        "`roi_levels` and `bg_levels`. |"
    )
    lines.append(
        "| `lambda_distill` | Loss weight for teacher-student distillation term. | "
        "`0`: distillation disabled. `>0`: distillation contributes to total loss. "
        "Larger values emphasize teacher matching more strongly. |"
    )
    lines.append(
        "| `lambda_importance` | Loss weight for importance/ROI supervision term. | "
        "Larger values force stronger alignment between predicted importance and target ROI/teacher importance. |"
    )
    lines.append(
        "| `loss_recipe` | Composite objective recipe variant. | "
        "`legacy`: old objective. `balanced_v1`: normalized rate + weighted BCE. "
        "`balanced_v2`: bg-focused normalized rate + weighted BCE + ROI/BG separation margin. |"
    )
    lines.append(
        "| `rate_loss_mode` | How rate proxy is computed from level map. | "
        "`global_mean`, `normalized_global`, or `normalized_bg` (background-focused). |"
    )
    lines.append(
        "| `distill_logit_loss` | Distillation loss type for logits. | "
        "`auto` picks `bce` for 1-channel logits, `kl` otherwise. Can force `kl`/`bce`/`mse`. |"
    )
    lines.append(
        "| `importance_head_type` | Importance head architecture variant. | "
        "`basic`, `multiscale`, `pp_lite`, `bifpn`, `deformable_msa`, `dynamic`, `rangeformer`, `frnet`. |"
    )
    lines.append(
        "| `final_eq_bits` | Final epoch effective quantization bits per latent symbol. | "
        "Adaptive: `mean(log2(level_map))`; Uniform: constant `quant_bits`. |"
    )
    lines.append(
        "| `final_code_entropy` | Final epoch empirical entropy of latent quant codes. | "
        "Shannon entropy in bits/symbol from observed code histogram. |"
    )
    lines.append("")
    lines.append("## Experiment Table")
    lines.append("")
    lines.append(
        "| log_file | stage | backbone | quantizer_mode | quant_bits | loss_recipe | rate_loss_mode | "
        "importance_loss_mode | distill_logit_loss | importance_head_type | lr | lambda_distill | "
        "lambda_importance | lambda_imp_separation | roi_target_mode | roi_target_mode_source | "
        "epochs | final_loss | final_eq_bits | final_code_entropy | rel_improve_% | save_dir |"
    )
    lines.append("|---|---:|---|---|---:|---|---|---|---|---|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---|")

    for r in rows:
        lines.append(
            f"| `{r['log_file']}` | {r['stage'] or 'n/a'} | {r['backbone'] or 'n/a'} | "
            f"{r['quantizer_mode'] or 'n/a'} | {r['quant_bits'] or 'n/a'} | "
            f"{r.get('loss_recipe', 'n/a') or 'n/a'} | {r.get('rate_loss_mode', 'n/a') or 'n/a'} | "
            f"{r.get('importance_loss_mode', 'n/a') or 'n/a'} | {r.get('distill_logit_loss', 'n/a') or 'n/a'} | "
            f"{r.get('importance_head_type', 'n/a') or 'n/a'} | "
            f"{r['lr'] or 'n/a'} | {_format_float(_safe_float(r['lambda_distill']), 3)} | "
            f"{_format_float(_safe_float(r['lambda_importance']), 3)} | "
            f"{_format_float(_safe_float(r.get('lambda_imp_separation')), 3)} | "
            f"{r['roi_target_mode'] or 'n/a'} | {r['roi_target_mode_source'] or 'n/a'} | "
            f"{r['epochs'] or 'n/a'} | {_format_float(_safe_float(r['final_loss']), 4)} | "
            f"{_format_float(_safe_float(r.get('final_eq_bits')), 3)} | "
            f"{_format_float(_safe_float(r.get('final_code_entropy')), 3)} | "
            f"{_format_float(_safe_float(r['rel_improve_pct']), 2)} | `{r['save_dir'] or ''}` |"
        )

    lines.append("")
    lines.append("## Stage Summary")
    lines.append("")
    stage_groups: Dict[str, List[Dict[str, str]]] = {}
    for r in rows:
        stage_groups.setdefault(str(r.get("stage", "n/a")), []).append(r)
    lines.append("| stage | runs | best_final_loss | best_log |")
    lines.append("|---:|---:|---:|---|")
    for stage, group in sorted(stage_groups.items(), key=lambda x: x[0]):
        group_valid = [g for g in group if _safe_float(g.get("final_loss")) == _safe_float(g.get("final_loss"))]
        if group_valid:
            best = min(group_valid, key=lambda g: _safe_float(g.get("final_loss")))
            lines.append(
                f"| {stage} | {len(group)} | {_format_float(_safe_float(best['final_loss']), 4)} | `{best['log_file']}` |"
            )
        else:
            lines.append(f"| {stage} | {len(group)} | n/a | n/a |")

    lines.append("")
    lines.append("## Notes")
    lines.append("- This ledger is intended to be updated continuously after each experiment.")
    lines.append("- Recommended command:")
    lines.append("  - `python src/utils/update_experiments_result.py`")
    lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate experiments_result markdown/csv from logs.")
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--out_md", type=str, default="src/results/experiments_result.md")
    parser.add_argument("--out_csv", type=str, default="src/results/experiments_result.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    log_dir = Path(args.log_dir)
    out_md = Path(args.out_md)
    out_csv = Path(args.out_csv)

    rows = collect_runs(log_dir)
    write_csv(rows, out_csv)
    render_md(rows, out_md, log_dir)

    print(f"Collected runs: {len(rows)}")
    print(f"Wrote markdown: {out_md}")
    print(f"Wrote csv: {out_csv}")


if __name__ == "__main__":
    main()
