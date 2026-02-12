import json
from pathlib import Path


def main():
    runs_dir = Path("data") / "results" / "runs"
    if not runs_dir.exists():
        print("No data/results/runs directory found.")
        return

    manifests = sorted(runs_dir.glob("*/manifest.json"))
    if not manifests:
        print("No manifests found under data/results/runs/*/manifest.json")
        return

    print("run_id\tstage\tq\tnoise\tlr\tbs\te\tbc\tlc\tst\tbp\tact\tdrop")
    for m in manifests:
        data = json.loads(m.read_text(encoding="utf-8"))
        cfg = data.get("config", {})
        print(
            f"{data.get('run_id','')}\t{data.get('stage','')}\t"
            f"{cfg.get('quant_bits','')}\t{cfg.get('noise_std','')}\t{cfg.get('lr','')}\t"
            f"{cfg.get('batch_size','')}\t{cfg.get('epochs','')}\t"
            f"{cfg.get('base_channels','')}\t{cfg.get('latent_channels','')}\t{cfg.get('num_stages','')}\t"
            f"{cfg.get('blocks_per_stage','')}\t{cfg.get('activation','')}\t{cfg.get('dropout','')}"
        )


if __name__ == "__main__":
    main()
