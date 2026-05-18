#!/usr/bin/env python3
"""Aggregate per-codec EER and minDCF across seeds {42,123,456,789,2024}
for the six main deep models, then emit Markdown + LaTeX tables.

Outputs:
    figures/tables/T2_per_codec_eer_seeds.md / .tex
    figures/tables/T2_per_codec_mindcf_seeds.md / .tex
    results/per_codec_seeds_summary.csv
"""

from __future__ import annotations

import csv
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1]
PRED = ROOT / "results" / "predictions"
OUT_TABLES = ROOT / "figures" / "tables"
OUT_CSV = ROOT / "results" / "per_codec_seeds_summary.csv"

SEEDS = [42, 123, 456, 789, 2024]
# Models that used a "_v2_eval" suffix on seed 42
V2_MODELS = {"wavlm_dann", "wavlm_erm_aug", "w2v2_dann", "w2v2_erm_aug"}
MODELS = [
    ("wavlm_erm", "WavLM ERM"),
    ("wavlm_erm_aug", "WavLM ERM+Aug"),
    ("wavlm_dann", "WavLM DANN"),
    ("w2v2_erm", "W2V2 ERM"),
    ("w2v2_erm_aug", "W2V2 ERM+Aug"),
    ("w2v2_dann", "W2V2 DANN"),
]
CODEC_ORDER = [f"C{i:02d}" for i in range(1, 12)] + ["NONE"]


def pred_dir(model: str, seed: int) -> Path:
    suffix = f"seed{seed}_v2_eval" if (seed == 42 and model in V2_MODELS) else f"seed{seed}_eval"
    return PRED / f"{model}_{suffix}" / "tables" / "metrics_by_codec.csv"


def load(path: Path) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            out[row["domain"]] = {
                "eer": float(row["eer"]),
                "min_dcf": float(row["min_dcf"]),
            }
    return out


def fmt(values: list[float], scale: float = 1.0, prec: int = 2) -> str:
    vs = [v * scale for v in values]
    return f"{mean(vs):.{prec}f} ± {stdev(vs):.{prec}f}" if len(vs) > 1 else f"{vs[0]:.{prec}f}"


def main() -> None:
    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    # data[model][codec][metric] = list of per-seed values
    data: dict[str, dict[str, dict[str, list[float]]]] = {}
    for model, _ in MODELS:
        data[model] = {c: {"eer": [], "min_dcf": []} for c in CODEC_ORDER}
        for seed in SEEDS:
            path = pred_dir(model, seed)
            if not path.exists():
                raise FileNotFoundError(path)
            seed_metrics = load(path)
            for codec in CODEC_ORDER:
                if codec not in seed_metrics:
                    raise KeyError(f"{codec} missing in {path}")
                data[model][codec]["eer"].append(seed_metrics[codec]["eer"])
                data[model][codec]["min_dcf"].append(seed_metrics[codec]["min_dcf"])

    # ---- write per-metric markdown + LaTeX ----
    def write_table(metric: str, scale: float, prec: int, caption: str, basename: str) -> None:
        header = ["Codec"] + [name for _, name in MODELS]
        rows = []
        for codec in CODEC_ORDER:
            row = [codec]
            for model, _ in MODELS:
                row.append(fmt(data[model][codec][metric], scale=scale, prec=prec))
            rows.append(row)

        # markdown
        md = [f"# {caption}", ""]
        md.append("| " + " | ".join(header) + " |")
        md.append("| " + " | ".join(["---"] * len(header)) + " |")
        for r in rows:
            md.append("| " + " | ".join(r) + " |")
        (OUT_TABLES / f"{basename}.md").write_text("\n".join(md) + "\n")

        # LaTeX (booktabs)
        col_spec = "l" + "c" * len(MODELS)
        tex = [
            "\\begin{table}[t]",
            "\\centering",
            "\\small",
            "\\begin{tabular}{" + col_spec + "}",
            "\\toprule",
            " & ".join(header) + " \\\\",
            "\\midrule",
        ]
        for r in rows:
            tex.append(" & ".join(r).replace("±", "$\\pm$") + " \\\\")
        tex += [
            "\\bottomrule",
            "\\end{tabular}",
            f"\\caption{{{caption}. Mean $\\pm$ standard deviation across seeds {{42, 123, 456}}.}}",
            f"\\label{{tab:{basename}}}",
            "\\end{table}",
        ]
        (OUT_TABLES / f"{basename}.tex").write_text("\n".join(tex) + "\n")

    write_table(
        metric="eer",
        scale=100.0,
        prec=2,
        caption=f"Per-codec EER (\\%) on Eval set, mean $\\pm$ std over {len(SEEDS)} seeds \\{{{', '.join(str(s) for s in SEEDS)}\\}}",
        basename="T2_per_codec_eer_seeds",
    )
    write_table(
        metric="min_dcf",
        scale=1.0,
        prec=3,
        caption=f"Per-codec minDCF on Eval set, mean $\\pm$ std over {len(SEEDS)} seeds \\{{{', '.join(str(s) for s in SEEDS)}\\}}",
        basename="T2_per_codec_mindcf_seeds",
    )

    # ---- long-form CSV with raw + summary ----
    seed_cols = [f"seed{s}" for s in SEEDS]
    with OUT_CSV.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "codec", "metric"] + seed_cols + ["mean", "std"])
        for model, _ in MODELS:
            for codec in CODEC_ORDER:
                for metric in ("eer", "min_dcf"):
                    vs = data[model][codec][metric]
                    w.writerow(
                        [model, codec, metric]
                        + [f"{v:.6f}" for v in vs]
                        + [f"{mean(vs):.6f}", f"{stdev(vs):.6f}"]
                    )

    print(f"Wrote: {OUT_TABLES / 'T2_per_codec_eer_seeds.md'}")
    print(f"Wrote: {OUT_TABLES / 'T2_per_codec_mindcf_seeds.md'}")
    print(f"Wrote: {OUT_TABLES / 'T2_per_codec_eer_seeds.tex'}")
    print(f"Wrote: {OUT_TABLES / 'T2_per_codec_mindcf_seeds.tex'}")
    print(f"Wrote: {OUT_CSV}")


if __name__ == "__main__":
    main()
