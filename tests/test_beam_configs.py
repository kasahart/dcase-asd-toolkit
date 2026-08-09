from pathlib import Path

import numpy as np
import pandas as pd
from hydra import compose, initialize_config_dir

from asdkit.backends import BEAMVarianceMin
from asdkit.bin.score import hydra_to_pydantic as score_hydra_to_pydantic
from asdkit.utils.asdkit_utils.evaluate import get_as_name as get_evaluation_as_name
from asdkit.utils.asdkit_utils.score import add_score
from asdkit.utils.common.instantiate_util import instantiate_tgt


def _common_overrides(experiment):
    return [
        f"experiments={experiment}",
        "seed=0",
        "dcase=dcase2023",
        "name=test",
        "version=test",
        "infer_ver=last",
        "machine=bearing",
    ]


def test_score_configs_compose_and_backend_instantiates():
    config_dir = str(Path("config/score").resolve())
    for experiment, expected_rescaling in [
        ("beam_raw", False),
        ("beam_varmin4", True),
    ]:
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            hydra_cfg = compose(
                config_name="main", overrides=_common_overrides(experiment)
            )
        cfg = score_hydra_to_pydantic(hydra_cfg)
        backend = instantiate_tgt(cfg.backend[0])
        assert isinstance(backend, BEAMVarianceMin)
        assert backend.use_rescaling is expected_rescaling


def test_score_pipeline_accepts_beam_outputs():
    rng = np.random.default_rng(23)
    train_embed = rng.normal(size=(6, 2, 3)).astype(np.float32)
    test_embed = rng.normal(size=(3, 2, 3)).astype(np.float32)
    train = {
        "embed_freq": train_embed,
        "path": np.array([f"train-{i}" for i in range(6)]),
        "section": np.zeros(6, dtype=np.int64),
        "is_normal": np.ones(6, dtype=np.int64),
    }
    test = {
        "embed_freq": test_embed,
        "path": np.array([f"test-{i}" for i in range(3)]),
        "section": np.zeros(3, dtype=np.int64),
        "is_normal": np.zeros(3, dtype=np.int64),
    }
    frames = {
        "train": pd.DataFrame({"path": train["path"]}),
        "test": pd.DataFrame({"path": test["path"]}),
    }
    result = add_score(
        backend_cfg={
            "tgt_class": "asdkit.backends.BEAMVarianceMin",
            "rescale_k": 4,
            "chunk_size": 2,
        },
        extract_dict_dict={"train": train, "test": test},
        score_df_dict=frames,
    )
    for split, length in [("train", 6), ("test", 3)]:
        score_columns = [column for column in result[split] if column.startswith("AS-")]
        diagnostic_columns = [
            column
            for column in result[split]
            if column.startswith("diagnostic-")
        ]
        assert len(score_columns) == 1
        assert score_columns[0].endswith("-main")
        assert len(diagnostic_columns) == 2
        assert any(column.endswith("-raw") for column in diagnostic_columns)
        assert any(column.endswith("-rescale_delta") for column in diagnostic_columns)
        assert get_evaluation_as_name(result[split]) == score_columns
        all_score_columns = score_columns + diagnostic_columns
        assert result[split][all_score_columns].shape == (length, 3)
        assert np.isfinite(result[split][all_score_columns].to_numpy()).all()
