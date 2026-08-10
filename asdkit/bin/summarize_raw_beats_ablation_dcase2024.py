"""Summarize the controlled DCASE 2024 frozen-BEATs ablation."""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import hmean

from asdkit.utils.dcase_utils import MACHINE_DICT


DCASE = "dcase2024"
OFFICIAL_METRIC = "0_official24"


@dataclass(frozen=True)
class Condition:
    frontend_config: str
    backend_config: str
    representation: str
    pooling: str
    gamma: str
    backend: str
    varmin: str
    backend_embed_key: str
    expected_embedding_tail: Tuple[int, ...]
    extraction_source: str


CONDITIONS: Mapping[str, Condition] = {
    "B0": Condition(
        "scratch/raw_beats",
        "knn_raw_beats",
        "Global",
        "AP",
        "n/a",
        "KNN",
        "off",
        "embed",
        (768,),
        "B0",
    ),
    "B1": Condition(
        "scratch/raw_beats_freq_ap",
        "knn_raw_beats",
        "Frequency",
        "AP",
        "n/a",
        "KNN",
        "off",
        "embed",
        (8 * 768,),
        "B1",
    ),
    "B2": Condition(
        "scratch/raw_beats_freq_rdp4",
        "knn_raw_beats",
        "Frequency",
        "RDP",
        "4",
        "KNN",
        "off",
        "embed",
        (8 * 768,),
        "B2",
    ),
    "B3": Condition(
        "scratch/raw_beats_freq_ap",
        "beam_raw",
        "Frequency",
        "AP",
        "n/a",
        "BEAM",
        "off",
        "embed_freq",
        (8, 768),
        "B1",
    ),
    "B4": Condition(
        "scratch/raw_beats_freq_rdp4",
        "beam_raw",
        "Frequency",
        "RDP",
        "4",
        "BEAM",
        "off",
        "embed_freq",
        (8, 768),
        "B2",
    ),
    "B5": Condition(
        "scratch/raw_beats_freq_rdp4",
        "beam_varmin4",
        "Frequency",
        "RDP",
        "4",
        "BEAM",
        "K=4",
        "embed_freq",
        (8, 768),
        "B2",
    ),
}


COMPARISONS: Sequence[Tuple[str, str, str, str]] = (
    ("B1-B0", "B1", "B0", "frequency-preserving representation effect"),
    ("B2-B1", "B2", "B1", "RDP effect under KNN"),
    ("B3-B1", "B3", "B1", "BEAM effect under AP"),
    ("B4-B3", "B4", "B3", "RDP effect under BEAM"),
    ("B4-B2", "B4", "B2", "BEAM effect under RDP"),
    ("B5-B4", "B5", "B4", "VarMin effect"),
)


def version_for(condition_id: str) -> str:
    if condition_id not in CONDITIONS:
        raise KeyError(f"Unknown condition: {condition_id}")
    return f"raw_beats_ablation_{condition_id}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_metadata() -> Tuple[str, bool]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain", "--untracked-files=all"],
                text=True,
            ).strip()
        )
        return commit, dirty
    except (OSError, subprocess.CalledProcessError):
        return "unknown", True


def _output_root(
    result_dir: Path, name: str, condition_id: str, seed: int, infer_ver: str
) -> Path:
    return (
        result_dir
        / name
        / DCASE
        / version_for(condition_id)
        / str(seed)
        / "output"
        / infer_ver
    )


def validate_shared_extractions(
    roots: Mapping[str, Path], machines: Iterable[str]
) -> None:
    """Require shared-front-end conditions to reference identical NPZ files."""
    for condition_id, source_id in (("B3", "B1"), ("B4", "B2"), ("B5", "B2")):
        for machine in machines:
            for split in ("train", "test"):
                source = roots[source_id] / machine / f"{split}_extract.npz"
                target = roots[condition_id] / machine / f"{split}_extract.npz"
                if not source.exists() or not target.exists():
                    raise FileNotFoundError(
                        f"Missing shared extraction artifact: {source} or {target}"
                    )
                if not os.path.samefile(source, target):
                    raise ValueError(
                        f"{condition_id} must share {source_id} extraction exactly: "
                        f"{target} is not the same file as {source}"
                    )


def _read_single_evaluation(path: Path) -> Tuple[str, float]:
    frame = pd.read_csv(path)
    if len(frame) != 1:
        raise ValueError(f"Expected one primary backend in {path}, found {len(frame)}")
    if OFFICIAL_METRIC not in frame:
        raise KeyError(f"{OFFICIAL_METRIC} is missing from {path}")
    backend_name = str(frame.iloc[0]["backend"])
    score = float(frame.iloc[0][OFFICIAL_METRIC])
    if not np.isfinite(score):
        raise ValueError(f"Non-finite official score in {path}")
    return backend_name, score


def _read_table_total(path: Path) -> float:
    frame = pd.read_csv(path)
    if len(frame) != 1 or "total" not in frame:
        raise ValueError(f"Expected one backend and a total column in {path}")
    total = float(frame.iloc[0]["total"])
    if not np.isfinite(total):
        raise ValueError(f"Non-finite total in {path}")
    return total


def _validate_embedding(path: Path, condition: Condition) -> str:
    with np.load(path, allow_pickle=True) as archive:
        if condition.backend_embed_key not in archive:
            raise KeyError(f"{condition.backend_embed_key} is missing from {path}")
        embedding = archive[condition.backend_embed_key]
        if embedding.shape[1:] != condition.expected_embedding_tail:
            raise ValueError(
                f"Unexpected {condition.backend_embed_key} shape in {path}: "
                f"{embedding.shape}; expected [N, {condition.expected_embedding_tail}]"
            )
        if not np.isfinite(embedding).all():
            raise ValueError(f"Non-finite embedding in {path}")
    return "[" + ",".join(str(value) for value in condition.expected_embedding_tail) + "]"


def _build_delta_rows(
    machine_scores: pd.DataFrame, summary: pd.DataFrame
) -> pd.DataFrame:
    rows: List[dict] = []
    indexed = machine_scores.set_index(["ID", "split", "machine"])["official24"]
    summary_indexed = summary.set_index("ID")
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
                        summary_indexed.loc[left, f"official_{split}"]
                        - summary_indexed.loc[right, f"official_{split}"]
                    ),
                }
            )
    return pd.DataFrame(rows)


def _build_diagnostics(delta_frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    machine_rows = delta_frame[delta_frame["scope"] == "machine"]
    for (comparison, interpretation, split), group in machine_rows.groupby(
        ["comparison", "interpretation", "split"], sort=False
    ):
        values = group["delta"].to_numpy(dtype=float)
        largest_positive = group.loc[group["delta"].idxmax()]
        largest_negative = group.loc[group["delta"].idxmin()]
        rows.append(
            {
                "comparison": comparison,
                "interpretation": interpretation,
                "split": split,
                "improved": int(np.sum(values > 0)),
                "degraded": int(np.sum(values < 0)),
                "unchanged": int(np.sum(values == 0)),
                "mean_machine_delta": float(values.mean()),
                "largest_positive_machine": largest_positive["machine"],
                "largest_positive_delta": float(largest_positive["delta"]),
                "largest_negative_machine": largest_negative["machine"],
                "largest_negative_delta": float(largest_negative["delta"]),
            }
        )
    return pd.DataFrame(rows)


def _markdown_table(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    selected = frame.loc[:, columns].copy()
    for column in selected.select_dtypes(include=[np.number]).columns:
        selected[column] = selected[column].map(lambda value: f"{value:.9f}")
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    rows = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in selected.itertuples(index=False, name=None)
    ]
    return "\n".join([header, divider, *rows])


def _write_report(
    output_path: Path,
    summary: pd.DataFrame,
    deltas: pd.DataFrame,
    diagnostics: pd.DataFrame,
    commit: str,
    dirty: bool,
    checkpoint_hash: str,
) -> None:
    lines = [
        "# DCASE 2024 frozen raw-BEATs ablation",
        "",
        f"- Git commit: `{commit}` (dirty: `{str(dirty).lower()}`)",
        f"- BEATs checkpoint SHA-256: `{checkpoint_hash}`",
        "- Fixed controls: DCASE 2024, seed 0, 16 kHz, first/mono channel, "
        "12-second crop, tile padding, frozen Original BEATs_iter3, identical "
        "machine order and ASDKit official24 evaluation.",
        "- B0 uses a 768-dimensional global embedding. B1/B2 KNN use the "
        "flattened 8x768=6144-dimensional frequency representation; therefore "
        "B1-B0 includes frequency retention and increased dimensionality.",
        "- DCASE 2024 reports separate official dev and eval aggregates; no "
        "invented combined score is reported.",
        "",
        "## Aggregate scores",
        "",
        _markdown_table(
            summary,
            [
                "ID",
                "Representation",
                "Pooling",
                "gamma",
                "Backend",
                "VarMin",
                "embedding_shape",
                "official_dev",
                "official_eval",
            ],
        ),
        "",
        "## Controlled comparisons",
        "",
        "Positive delta means the left-hand configuration scored higher. These "
        "comparisons support interpretation only for the stated controlled "
        "implementation difference.",
        "",
    ]
    aggregate = deltas[deltas["scope"] == "aggregate"].pivot(
        index=["comparison", "interpretation"], columns="split", values="delta"
    )
    aggregate = aggregate.reset_index().rename(
        columns={"dev": "dev_delta", "eval": "eval_delta"}
    )
    lines.extend(
        [
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
                    "unchanged",
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
    )
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
    output_path.write_text("\n".join(lines), encoding="utf-8")


def summarize(args: argparse.Namespace) -> Path:
    result_dir = Path(args.result_dir)
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"BEATs checkpoint not found: {checkpoint}")
    machines_by_split = {
        split: list(MACHINE_DICT[f"{DCASE}-{split}"]) for split in ("dev", "eval")
    }
    all_machines = machines_by_split["dev"] + machines_by_split["eval"]
    roots = {
        condition_id: _output_root(
            result_dir, args.name, condition_id, args.seed, args.infer_ver
        )
        for condition_id in CONDITIONS
    }
    validate_shared_extractions(roots, all_machines)

    summary_rows = []
    machine_rows = []
    metadata_rows = []
    commit, dirty = _git_metadata()
    checkpoint_hash = _sha256(checkpoint)
    generated_at = datetime.now(timezone.utc).isoformat()

    for condition_id, condition in CONDITIONS.items():
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
                if embedding_shape is None:
                    embedding_shape = observed_shape
                elif embedding_shape != observed_shape:
                    raise ValueError(f"Embedding shape changed within {condition_id}")
                machine_rows.append(
                    {
                        "ID": condition_id,
                        "split": split,
                        "machine": machine,
                        "official24": score,
                    }
                )
                evaluation_path = machine_root / "test_evaluate.csv"
                metadata_rows.append(
                    {
                        "ID": condition_id,
                        "git_commit": commit,
                        "git_dirty": dirty,
                        "dcase": DCASE,
                        "machine": machine,
                        "split": split,
                        "seed": args.seed,
                        "checkpoint_path": str(checkpoint),
                        "checkpoint_sha256": checkpoint_hash,
                        "frontend_config": condition.frontend_config,
                        "backend_config": condition.backend_config,
                        "pooling": condition.pooling,
                        "gamma": condition.gamma,
                        "varmin_k": 4 if condition_id == "B5" else "off",
                        "timestamp": datetime.fromtimestamp(
                            evaluation_path.stat().st_mtime, timezone.utc
                        ).isoformat(),
                        "evaluation_path": str(evaluation_path),
                        "extraction_source_version": version_for(
                            condition.extraction_source
                        ),
                    }
                )
        if len(backend_names) != 1:
            raise ValueError(
                f"Backend identity changed within {condition_id}: {backend_names}"
            )
        aggregates = {
            split: float(hmean(values)) for split, values in split_scores.items()
        }
        for split, aggregate in aggregates.items():
            table_total = _read_table_total(root / f"{split}_official24.csv")
            if not np.isclose(aggregate, table_total, rtol=0, atol=1e-12):
                raise ValueError(
                    f"Manual and ASDKit totals differ for {condition_id} {split}: "
                    f"{aggregate} != {table_total}"
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
    deltas = _build_delta_rows(machine_scores, summary)
    diagnostics = _build_diagnostics(deltas)
    metadata = pd.DataFrame(metadata_rows)

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else result_dir
        / args.name
        / DCASE
        / "raw_beats_ablation_summary"
        / str(args.seed)
        / args.infer_ver
    )
    output_files = {
        "summary.csv": summary,
        "machine_scores.csv": machine_scores,
        "machine_deltas.csv": deltas,
        "comparison_diagnostics.csv": diagnostics,
        "reproducibility_metadata.csv": metadata,
    }
    existing = [output_dir / name for name in [*output_files, "REPORT.md"]]
    if not args.overwrite and any(path.exists() for path in existing):
        raise FileExistsError(
            f"Ablation summary already exists in {output_dir}; use a distinct name "
            "or pass --overwrite explicitly"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, frame in output_files.items():
        frame.to_csv(output_dir / filename, index=False)
    _write_report(
        output_dir / "REPORT.md",
        summary,
        deltas,
        diagnostics,
        commit,
        dirty,
        checkpoint_hash,
    )
    (output_dir / "generated_at.txt").write_text(
        generated_at + "\n", encoding="utf-8"
    )
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", default="./results")
    parser.add_argument("--name", default="dcase2024_raw_beats_ablation")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--infer-ver", default="last")
    parser.add_argument(
        "--checkpoint", default="pretrained_models/beats/BEATs_iter3.pt"
    )
    parser.add_argument("--output-dir")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = summarize(args)
    print(f"Saved DCASE 2024 raw-BEATs ablation summary to {output_dir}")


if __name__ == "__main__":
    main()
