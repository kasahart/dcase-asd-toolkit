"""Summarize the deterministic A6 and coupled-BEAM DCASE 2024 follow-up."""

import argparse
from collections import OrderedDict
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import hmean

from asdkit.bin.summarize_raw_beats_ablation_dcase2024 import (
    CONDITIONS,
    Condition,
    _build_diagnostics,
    _markdown_table,
    _read_single_evaluation,
    _read_table_total,
    _validate_embedding,
)
from asdkit.utils.dcase_utils import MACHINE_DICT


DCASE = "dcase2024"

FOLLOWUPS: Mapping[str, Condition] = OrderedDict(
    {
        "A6": replace(
            CONDITIONS["B3"],
            varmin="K=4",
            backend_config="beam_varmin4",
            extraction_source="B1",
        ),
        "CAP": replace(
            CONDITIONS["B3"],
            backend="Coupled-BEAM",
            backend_config="beam_coupled_raw",
            extraction_source="B1",
        ),
        "CRDP": replace(
            CONDITIONS["B4"],
            backend="Coupled-BEAM",
            backend_config="beam_coupled_raw",
            extraction_source="B2",
        ),
    }
)

REPORT_CONDITIONS: Mapping[str, Condition] = OrderedDict(
    (condition_id, CONDITIONS[condition_id]) for condition_id in ("B3", "B4", "B5")
)
REPORT_CONDITIONS.update(FOLLOWUPS)  # type: ignore[attr-defined]

ROOT_IDS: Sequence[str] = tuple(
    dict.fromkeys(
        [
            *REPORT_CONDITIONS,
            *(condition.extraction_source for condition in FOLLOWUPS.values()),
        ]
    )
)

COMPARISONS: Sequence[Tuple[str, str, str, str]] = (
    ("A6-B3", "A6", "B3", "VarMin effect under AP"),
    ("B5-A6", "B5", "A6", "RDP effect under VarMin"),
    ("B3-CAP", "B3", "CAP", "independent per-band reference effect under AP"),
    (
        "B4-CRDP",
        "B4",
        "CRDP",
        "independent per-band reference effect under RDP",
    ),
    ("CRDP-CAP", "CRDP", "CAP", "RDP effect under coupled reference selection"),
)


def _followup_output_root(
    result_dir: Path, name: str, condition_id: str, seed: int, infer_ver: str
) -> Path:
    return (
        result_dir
        / name
        / DCASE
        / f"raw_beats_ablation_{condition_id}"
        / str(seed)
        / "output"
        / infer_ver
    )


def validate_shared_extractions(roots: Mapping[str, Path], machines: Sequence[str]) -> None:
    for target_id, condition in FOLLOWUPS.items():
        source_root = roots[condition.extraction_source]
        target_root = roots[target_id]
        for machine in machines:
            for split in ("train", "test"):
                source = source_root / machine / f"{split}_extract.npz"
                target = target_root / machine / f"{split}_extract.npz"
                if not source.is_file() or not target.is_file():
                    raise FileNotFoundError(
                        f"Missing shared extraction for {target_id} {machine} {split}"
                    )
                if not source.samefile(target):
                    raise ValueError(
                        f"{target_id} must share {condition.extraction_source} "
                        f"extraction for {machine} {split}"
                    )


def _build_deltas(machine_scores: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    rows: List[dict] = []
    indexed = machine_scores.set_index(["ID", "split", "machine"])[
        "official24"
    ].sort_index()
    aggregate = summary.set_index("ID")
    for comparison, left, right, interpretation in COMPARISONS:
        for split in ("dev", "eval"):
            left_values = indexed.loc[(left, split)]
            right_values = indexed.loc[(right, split)]
            if list(left_values.index) != list(right_values.index):
                raise ValueError(f"Machine order differs for {comparison} {split}")
            for machine in left_values.index:
                rows.append(
                    {
                        "comparison": comparison,
                        "interpretation": interpretation,
                        "split": split,
                        "scope": "machine",
                        "machine": machine,
                        "delta": float(left_values[machine] - right_values[machine]),
                    }
                )
            rows.append(
                {
                    "comparison": comparison,
                    "interpretation": interpretation,
                    "split": split,
                    "scope": "aggregate",
                    "machine": "__aggregate__",
                    "delta": float(
                        aggregate.loc[left, f"official_{split}"]
                        - aggregate.loc[right, f"official_{split}"]
                    ),
                }
            )
    return pd.DataFrame(rows)


def _write_report(
    path: Path,
    summary: pd.DataFrame,
    deltas: pd.DataFrame,
    diagnostics: pd.DataFrame,
) -> None:
    aggregate = (
        deltas[deltas["scope"] == "aggregate"]
        .pivot(index=["comparison", "interpretation"], columns="split", values="delta")
        .reset_index()
        .rename(columns={"dev": "dev_delta", "eval": "eval_delta"})
    )
    lines = [
        "# DCASE 2024 raw-BEATs follow-up ablation",
        "",
        "A6 fills the AP + BEAM + VarMin factorial cell. CAP and CRDP use the "
        "same all-normal memory and scaled cosine as BEAM, but select one "
        "reference after averaging its eight aligned band distances.",
        "",
        "## Aggregate scores",
        "",
        _markdown_table(
            summary,
            [
                "ID",
                "Representation",
                "Pooling",
                "Backend",
                "VarMin",
                "official_dev",
                "official_eval",
            ],
        ),
        "",
        "## Controlled comparisons",
        "",
        _markdown_table(
            aggregate,
            ["comparison", "interpretation", "dev_delta", "eval_delta"],
        ),
        "",
        "## Machine-level diagnostics",
        "",
        _markdown_table(
            diagnostics,
            [
                "comparison",
                "split",
                "improved",
                "degraded",
                "mean_machine_delta",
                "largest_positive_machine",
                "largest_positive_delta",
                "largest_negative_machine",
                "largest_negative_delta",
            ],
        ),
        "",
        "## Machine-level deltas",
        "",
    ]
    for comparison, _, _, interpretation in COMPARISONS:
        subset = deltas[
            (deltas["comparison"] == comparison) & (deltas["scope"] == "machine")
        ]
        lines.extend(
            [
                f"### {comparison}: {interpretation}",
                "",
                _markdown_table(subset, ["split", "machine", "delta"]),
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def summarize(args: argparse.Namespace) -> Path:
    result_dir = Path(args.result_dir)
    machines_by_split = {
        split: list(MACHINE_DICT[f"{DCASE}-{split}"]) for split in ("dev", "eval")
    }
    all_machines = machines_by_split["dev"] + machines_by_split["eval"]
    roots: Dict[str, Path] = {
        condition_id: _followup_output_root(
            result_dir, args.name, condition_id, args.seed, args.infer_ver
        )
        for condition_id in ROOT_IDS
    }
    validate_shared_extractions(roots, all_machines)

    summary_rows = []
    machine_rows = []
    for condition_id, condition in REPORT_CONDITIONS.items():
        root = roots[condition_id]
        backend_names = set()
        split_scores: Dict[str, List[float]] = {"dev": [], "eval": []}
        embedding_shape = None
        for split, machines in machines_by_split.items():
            for machine in machines:
                machine_root = root / machine
                backend_name, score = _read_single_evaluation(
                    machine_root / "test_evaluate.csv"
                )
                backend_names.add(backend_name)
                split_scores[split].append(score)
                observed_shape = _validate_embedding(
                    machine_root / "test_extract.npz", condition
                )
                embedding_shape = embedding_shape or observed_shape
                if embedding_shape != observed_shape:
                    raise ValueError(f"Embedding shape changed within {condition_id}")
                machine_rows.append(
                    {
                        "ID": condition_id,
                        "split": split,
                        "machine": machine,
                        "official24": score,
                    }
                )
        if len(backend_names) != 1:
            raise ValueError(f"Backend identity changed within {condition_id}")
        aggregates = {
            split: float(hmean(values)) for split, values in split_scores.items()
        }
        for split, value in aggregates.items():
            table_total = _read_table_total(root / f"{split}_official24.csv")
            if not np.isclose(value, table_total, rtol=0, atol=1e-12):
                raise ValueError(
                    f"Manual and ASDKit totals differ for {condition_id} {split}"
                )
        summary_rows.append(
            {
                "ID": condition_id,
                "Representation": condition.representation,
                "Pooling": condition.pooling,
                "gamma": condition.gamma,
                "Backend": condition.backend,
                "VarMin": condition.varmin,
                "embedding_shape": embedding_shape,
                "frontend_config": condition.frontend_config,
                "backend_config": condition.backend_config,
                "backend_name": next(iter(backend_names)),
                "official_dev": aggregates["dev"],
                "official_eval": aggregates["eval"],
            }
        )

    summary = pd.DataFrame(summary_rows)
    machine_scores = pd.DataFrame(machine_rows)
    deltas = _build_deltas(machine_scores, summary)
    diagnostics = _build_diagnostics(deltas)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else result_dir
        / args.name
        / DCASE
        / "raw_beats_ablation_followup_summary"
        / str(args.seed)
        / args.infer_ver
    )
    outputs = {
        "summary.csv": summary,
        "machine_scores.csv": machine_scores,
        "machine_deltas.csv": deltas,
        "comparison_diagnostics.csv": diagnostics,
    }
    existing = [output_dir / filename for filename in [*outputs, "REPORT.md"]]
    if not args.overwrite and any(path.exists() for path in existing):
        raise FileExistsError(f"Follow-up summary already exists in {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, frame in outputs.items():
        frame.to_csv(output_dir / filename, index=False)
    _write_report(output_dir / "REPORT.md", summary, deltas, diagnostics)
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", default="./results")
    parser.add_argument("--name", default="dcase2024_raw_beats_ablation")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--infer-ver", default="last")
    parser.add_argument("--output-dir")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    output = summarize(build_parser().parse_args())
    print(f"Saved follow-up summary to {output}")


if __name__ == "__main__":
    main()
