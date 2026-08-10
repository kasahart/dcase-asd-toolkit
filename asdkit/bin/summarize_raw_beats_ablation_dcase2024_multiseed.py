"""Aggregate DCASE 2024 frozen raw-BEATs ablations across random seeds."""

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from asdkit.bin.summarize_raw_beats_ablation_dcase2024 import COMPARISONS, CONDITIONS


DCASE = "dcase2024"


def _stats(frame: pd.DataFrame, group: Sequence[str], value: str) -> pd.DataFrame:
    grouped = frame.groupby(list(group), sort=False)[value]
    result = grouped.agg(["count", "mean", "std", "min", "max"]).reset_index()
    result["range"] = result["max"] - result["min"]
    return result


def _markdown(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    selected = frame.loc[:, columns]
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    rows = []
    for row in selected.itertuples(index=False, name=None):
        values = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                values.append(f"{value:.9f}")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, divider, *rows])


def _read_seed(result_dir: Path, name: str, seed: int, infer_ver: str):
    root = result_dir / name / DCASE / "raw_beats_ablation_summary" / str(seed) / infer_ver
    summary_path = root / "summary.csv"
    machine_path = root / "machine_scores.csv"
    delta_path = root / "machine_deltas.csv"
    for path in (summary_path, machine_path, delta_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing seed-{seed} result: {path}")
    summary = pd.read_csv(summary_path)
    machine = pd.read_csv(machine_path)
    delta = pd.read_csv(delta_path)
    expected_ids = list(CONDITIONS)
    if summary["ID"].tolist() != expected_ids:
        raise ValueError(f"Seed {seed} condition order differs: {summary['ID'].tolist()}")
    for frame in (summary, machine, delta):
        frame.insert(0, "seed", seed)
    return summary, machine, delta, root


def summarize(args: argparse.Namespace) -> Path:
    seeds = list(dict.fromkeys(args.seeds))
    if len(seeds) < 2:
        raise ValueError("At least two distinct seeds are required")
    result_dir = Path(args.result_dir)
    summaries = []
    machines = []
    deltas = []
    source_roots = []
    for seed in seeds:
        summary, machine, delta, root = _read_seed(
            result_dir, args.name, seed, args.infer_ver
        )
        summaries.append(summary)
        machines.append(machine)
        deltas.append(delta)
        source_roots.append(root)

    seed_scores = pd.concat(summaries, ignore_index=True)
    machine_scores = pd.concat(machines, ignore_index=True)
    seed_deltas = pd.concat(deltas, ignore_index=True)

    condition_rows = []
    for split in ("dev", "eval"):
        value = f"official_{split}"
        stats = _stats(seed_scores, ["ID"], value)
        stats.insert(1, "split", split)
        condition_rows.append(stats)
    condition_stats = pd.concat(condition_rows, ignore_index=True)

    machine_stats = _stats(
        machine_scores, ["ID", "split", "machine"], "official24"
    )

    aggregate_deltas = seed_deltas[seed_deltas["scope"] == "aggregate"].copy()
    comparison_stats = _stats(
        aggregate_deltas,
        ["comparison", "interpretation", "split"],
        "delta",
    )
    sign_counts = (
        aggregate_deltas.assign(
            positive=lambda x: (x["delta"] > 0).astype(int),
            negative=lambda x: (x["delta"] < 0).astype(int),
            zero=lambda x: (x["delta"] == 0).astype(int),
        )
        .groupby(["comparison", "interpretation", "split"], sort=False)[
            ["positive", "negative", "zero"]
        ]
        .sum()
        .reset_index()
    )
    comparison_stats = comparison_stats.merge(
        sign_counts, on=["comparison", "interpretation", "split"], validate="one_to_one"
    )

    machine_delta_source = seed_deltas[seed_deltas["scope"] == "machine"].copy()
    machine_delta_stats = _stats(
        machine_delta_source,
        ["comparison", "interpretation", "split", "machine"],
        "delta",
    )
    machine_sign_counts = (
        machine_delta_source.assign(
            positive=lambda x: (x["delta"] > 0).astype(int),
            negative=lambda x: (x["delta"] < 0).astype(int),
            zero=lambda x: (x["delta"] == 0).astype(int),
        )
        .groupby(
            ["comparison", "interpretation", "split", "machine"], sort=False
        )[["positive", "negative", "zero"]]
        .sum()
        .reset_index()
    )
    machine_delta_stats = machine_delta_stats.merge(
        machine_sign_counts,
        on=["comparison", "interpretation", "split", "machine"],
        validate="one_to_one",
    )
    sign_changes = machine_delta_stats[
        (machine_delta_stats["positive"] > 0)
        & (machine_delta_stats["negative"] > 0)
    ]

    seed_label = "seeds_" + "-".join(str(seed) for seed in seeds)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else result_dir
        / args.name
        / DCASE
        / "raw_beats_ablation_multiseed"
        / seed_label
        / args.infer_ver
    )
    outputs = {
        "condition_seed_scores.csv": seed_scores,
        "condition_multiseed_stats.csv": condition_stats,
        "machine_seed_scores.csv": machine_scores,
        "machine_multiseed_stats.csv": machine_stats,
        "comparison_seed_deltas.csv": aggregate_deltas,
        "comparison_multiseed_stats.csv": comparison_stats,
        "machine_delta_multiseed_stats.csv": machine_delta_stats,
    }
    existing = [output_dir / filename for filename in [*outputs, "REPORT.md"]]
    if not args.overwrite and any(path.exists() for path in existing):
        raise FileExistsError(
            f"Multi-seed summary already exists in {output_dir}; use a distinct "
            "output directory or pass --overwrite explicitly"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, frame in outputs.items():
        frame.to_csv(output_dir / filename, index=False)

    report = [
        "# DCASE 2024 frozen raw-BEATs multi-seed summary",
        "",
        f"- Seeds: `{seeds}`",
        "- Statistic: sample standard deviation (`ddof=1`).",
        "- Frozen BEATs extraction is shared exactly across seeds because the "
        "12-second, shuffle-disabled frontend is deterministic. The observed "
        "variation therefore measures stochastic backend behavior (SMOTE in KNN).",
        "- BEAM and VarMin are deterministic in this experiment; zero standard "
        "deviation is expected for B3-B5.",
        "",
        "## Condition statistics",
        "",
        _markdown(
            condition_stats,
            ["ID", "split", "count", "mean", "std", "min", "max", "range"],
        ),
        "",
        "## Controlled-comparison statistics",
        "",
        _markdown(
            comparison_stats,
            [
                "comparison",
                "split",
                "count",
                "mean",
                "std",
                "min",
                "max",
                "positive",
                "negative",
                "zero",
            ],
        ),
        "",
        "## Machine comparisons with seed-dependent sign",
        "",
        _markdown(
            sign_changes,
            [
                "comparison",
                "split",
                "machine",
                "mean",
                "std",
                "min",
                "max",
                "positive",
                "negative",
            ],
        ),
        "",
        "Source summaries:",
        "",
        *(f"- `{root}`" for root in source_roots),
        "",
        f"Generated: `{datetime.now(timezone.utc).isoformat()}`",
    ]
    (output_dir / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", default="./results")
    parser.add_argument("--name", default="dcase2024_raw_beats_ablation")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--infer-ver", default="last")
    parser.add_argument("--output-dir")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    output = summarize(build_parser().parse_args())
    print(f"Saved multi-seed summary to {output}")


if __name__ == "__main__":
    main()
