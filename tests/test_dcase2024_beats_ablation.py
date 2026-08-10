import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from hydra import compose, initialize_config_dir
from torch import nn

from asdkit.backends import BEAMVarianceMin, Knn
from asdkit.bin.extract import hydra_to_pydantic as extract_hydra_to_pydantic
from asdkit.bin.score import hydra_to_pydantic as score_hydra_to_pydantic
from asdkit.bin.summarize_raw_beats_ablation_dcase2024 import (
    COMPARISONS,
    CONDITIONS,
    validate_shared_extractions,
)
from asdkit.bin.summarize_raw_beats_ablation_dcase2024_multiseed import _stats
from asdkit.bin.summarize_raw_beats_ablation_dcase2024_followup import (
    COMPARISONS as FOLLOWUP_COMPARISONS,
    FOLLOWUPS,
    ROOT_IDS as FOLLOWUP_ROOT_IDS,
)
from asdkit.frontends.pretrained_feature.beats import (
    BEATsFrequencyPoolingFrozenModel,
    BEATsFrozenModel,
)
from asdkit.utils.common.instantiate_util import instantiate_tgt


class _DummyBEATs(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))

    def _sequence(self, wave):
        return torch.arange(
            wave.shape[0] * 3 * 8 * 768, dtype=wave.dtype, device=wave.device
        ).reshape(wave.shape[0], 3 * 8, 768)

    def extract_features(self, wave):
        return self._sequence(wave), None

    def extract_features_with_grid(self, wave):
        return self._sequence(wave), None, (3, 8)


def _overrides(experiment):
    return [
        f"experiments={experiment}",
        "seed=0",
        "dcase=dcase2024",
        "name=test",
        "version=test",
        "infer_ver=last",
        "machine=bearing",
    ]


@pytest.mark.parametrize("condition_id", list(CONDITIONS))
def test_all_six_ablation_configs_compose(monkeypatch, condition_id):
    monkeypatch.setattr(
        BEATsFrozenModel,
        "construct_model",
        lambda self, **kwargs: _DummyBEATs(),
    )
    condition = CONDITIONS[condition_id]

    extract_config_dir = str(Path("config/extract").resolve())
    with initialize_config_dir(version_base=None, config_dir=extract_config_dir):
        hydra_cfg = compose(
            config_name="main",
            overrides=_overrides(condition.frontend_config)
            + [
                "datamodule.train.collator.sec=dcase2024",
                "+datamodule.train.collator.pad_mode=tile",
                "datamodule.train.dataset.audio_channel=first",
            ],
        )
    extract_cfg = extract_hydra_to_pydantic(hydra_cfg)
    frontend = instantiate_tgt(extract_cfg.scratch_frontend)
    collator = instantiate_tgt(extract_cfg.datamodule.train.collator)

    assert collator.crop_len == 12 * 16000
    assert collator.pad_mode == "tile"
    assert extract_cfg.datamodule.train.dataset["audio_channel"] == "first"
    if condition_id == "B0":
        assert type(frontend) is BEATsFrozenModel
    else:
        assert isinstance(frontend, BEATsFrequencyPoolingFrozenModel)
        assert frontend.pooling == ("mean" if condition_id in {"B1", "B3"} else "rdp")
        assert frontend.gamma == 4

    score_config_dir = str(Path("config/score").resolve())
    with initialize_config_dir(version_base=None, config_dir=score_config_dir):
        score_hydra_cfg = compose(
            config_name="main", overrides=_overrides(condition.backend_config)
        )
    score_cfg = score_hydra_to_pydantic(score_hydra_cfg)
    assert len(score_cfg.backend) == 1
    backend = instantiate_tgt(score_cfg.backend[0])

    if condition_id in {"B0", "B1", "B2"}:
        assert isinstance(backend, Knn)
        assert backend.embed_key == "embed"
        assert backend.metric == "cosine"
        assert backend.n_neighbors_so == 1
        assert backend.n_neighbors_ta == 1
        assert backend.sep_section is False
        assert backend.smote is not None
        assert backend.smote.sampling_strategy == 0.2
        assert backend.smote.k_neighbors == 2
    else:
        assert isinstance(backend, BEAMVarianceMin)
        assert backend.embed_key == "embed_freq"
        assert backend.sep_section is False
        assert backend.use_rescaling is (condition_id == "B5")
        assert backend.rescaler.k == 4
        assert backend.rescaler.validation == "train_all"
        assert backend.rescaler.scope == "per_band"


def test_ablation_frontend_backend_input_shapes(monkeypatch):
    monkeypatch.setattr(
        BEATsFrozenModel,
        "construct_model",
        lambda self, **kwargs: _DummyBEATs(),
    )
    batch = {"wave": torch.zeros(2, 12 * 16000)}
    global_output = BEATsFrozenModel(model_cfg={}).extract(batch)
    ap_output = BEATsFrequencyPoolingFrozenModel(
        model_cfg={}, pooling="mean", gamma=4
    ).extract(batch)
    rdp_output = BEATsFrequencyPoolingFrozenModel(
        model_cfg={}, pooling="rdp", gamma=4
    ).extract(batch)

    assert global_output["embed"].shape == (2, 768)
    for output in (ap_output, rdp_output):
        assert output["embed_freq"].shape == (2, 8, 768)
        assert output["embed"].shape == (2, 8 * 768)
    assert CONDITIONS["B1"].backend_embed_key == "embed"
    assert CONDITIONS["B2"].backend_embed_key == "embed"
    assert CONDITIONS["B3"].backend_embed_key == "embed_freq"
    assert CONDITIONS["B4"].backend_embed_key == "embed_freq"
    assert CONDITIONS["B5"].backend_embed_key == "embed_freq"


def test_knn_config_exactly_matches_existing_raw_beats_entry():
    config_dir = str(Path("config/score").resolve())
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        dedicated = compose(
            config_name="main", overrides=_overrides("knn_raw_beats")
        )
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        existing = compose(config_name="main", overrides=_overrides("default"))
    dedicated_backend = dict(dedicated.backend[0])
    assert dedicated_backend.pop("random_state") == 0
    assert dedicated_backend == dict(existing.backend[2])


def test_knn_ablation_seed_controls_smote_random_state():
    config_dir = str(Path("config/score").resolve())
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        configured = compose(
            config_name="main",
            overrides=_overrides("knn_raw_beats") + ["seed=17"],
        )
    cfg = score_hydra_to_pydantic(configured)
    backend = instantiate_tgt(cfg.backend[0])
    assert backend.random_state == 17
    assert backend.smote is not None
    assert backend.smote.random_state == 17


def test_multiseed_summary_uses_sample_standard_deviation():
    frame = pd.DataFrame(
        {"ID": ["B0", "B0"], "official_dev": [1.0, 3.0]}
    )
    stats = _stats(frame, ["ID"], "official_dev").iloc[0]
    assert stats["count"] == 2
    assert stats["mean"] == 2.0
    assert stats["std"] == pytest.approx(np.sqrt(2.0))
    assert stats["range"] == 2.0


def test_followup_conditions_and_comparisons_are_fixed():
    assert list(FOLLOWUPS) == ["A6", "CAP", "CRDP"]
    assert FOLLOWUPS["A6"].frontend_config == "scratch/raw_beats_freq_ap"
    assert FOLLOWUPS["A6"].backend_config == "beam_varmin4"
    assert FOLLOWUPS["CAP"].extraction_source == "B1"
    assert FOLLOWUPS["CRDP"].extraction_source == "B2"
    assert set(FOLLOWUP_ROOT_IDS) == {
        "B1",
        "B2",
        "B3",
        "B4",
        "B5",
        "A6",
        "CAP",
        "CRDP",
    }
    assert [item[0] for item in FOLLOWUP_COMPARISONS] == [
        "A6-B3",
        "B5-A6",
        "B3-CAP",
        "B4-CRDP",
        "CRDP-CAP",
    ]


def test_coupled_beam_config_composes():
    config_dir = str(Path("config/score").resolve())
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        configured = compose(
            config_name="main", overrides=_overrides("beam_coupled_raw")
        )
    cfg = score_hydra_to_pydantic(configured)
    assert len(cfg.backend) == 1
    backend = instantiate_tgt(cfg.backend[0])
    assert isinstance(backend, BEAMVarianceMin)
    assert backend.embed_key == "embed_freq"
    assert backend.use_rescaling is False
    assert backend.neighbor_mode == "coupled"


def test_required_comparisons_and_shared_extraction_sources_are_fixed():
    assert [comparison[0] for comparison in COMPARISONS] == [
        "B1-B0",
        "B2-B1",
        "B3-B1",
        "B4-B3",
        "B4-B2",
        "B5-B4",
    ]
    assert CONDITIONS["B3"].extraction_source == "B1"
    assert CONDITIONS["B4"].extraction_source == "B2"
    assert CONDITIONS["B5"].extraction_source == "B2"


def test_b4_b5_and_b3_share_exact_extraction_files(tmp_path):
    roots = {condition_id: tmp_path / condition_id for condition_id in CONDITIONS}
    machine = "bearing"
    for source_id in ("B0", "B1", "B2"):
        machine_dir = roots[source_id] / machine
        machine_dir.mkdir(parents=True)
        for split in ("train", "test"):
            np.savez(machine_dir / f"{split}_extract.npz", marker=source_id)
    for target_id, source_id in (("B3", "B1"), ("B4", "B2"), ("B5", "B2")):
        target_dir = roots[target_id] / machine
        target_dir.mkdir(parents=True)
        for split in ("train", "test"):
            os.symlink(
                roots[source_id] / machine / f"{split}_extract.npz",
                target_dir / f"{split}_extract.npz",
            )

    validate_shared_extractions(roots, [machine])
    assert os.path.samefile(
        roots["B4"] / machine / "test_extract.npz",
        roots["B5"] / machine / "test_extract.npz",
    )
