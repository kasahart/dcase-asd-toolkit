from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from torch import nn

from asdkit.bin.extract import hydra_to_pydantic as extract_hydra_to_pydantic
from asdkit.frontends.pretrained_feature.beats import BEATsFrozenModel
from asdkit.utils.common.instantiate_util import instantiate_tgt


class _ConfigDummyBEATs(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))


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


def test_extract_configs_compose_and_frontend_instantiates(monkeypatch):
    monkeypatch.setattr(
        BEATsFrozenModel,
        "construct_model",
        lambda self, **kwargs: _ConfigDummyBEATs(),
    )
    config_dir = str(Path("config/extract").resolve())
    for experiment in [
        "scratch/raw_beats_freq_ap",
        "scratch/raw_beats_freq_rdp4",
    ]:
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            hydra_cfg = compose(
                config_name="main", overrides=_common_overrides(experiment)
            )
        cfg = extract_hydra_to_pydantic(hydra_cfg)
        frontend = instantiate_tgt(cfg.scratch_frontend)
        assert frontend.emit_patch_sequence is False


def test_dcase2026_recipe_overrides_compose_original_duration(monkeypatch):
    monkeypatch.setattr(
        BEATsFrozenModel,
        "construct_model",
        lambda self, **kwargs: _ConfigDummyBEATs(),
    )
    overrides = [
        "experiments=scratch/raw_beats_freq_rdp4",
        "seed=0",
        "dcase=dcase2026",
        "name=test",
        "version=test",
        "infer_ver=last",
        "machine=ToyCar",
        "datamodule.train.collator.sec=all",
        "datamodule.train.dataloader.batch_size=1",
        "scratch_frontend.gamma=8",
    ]
    config_dir = str(Path("config/extract").resolve())
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        hydra_cfg = compose(config_name="main", overrides=overrides)
    cfg = extract_hydra_to_pydantic(hydra_cfg)

    frontend = instantiate_tgt(cfg.scratch_frontend)
    train_collator = instantiate_tgt(cfg.datamodule.train.collator)
    test_collator = instantiate_tgt(cfg.datamodule.test.collator)

    assert frontend.pooling == "rdp"
    assert frontend.gamma == 8
    assert train_collator.crop_len == "all"
    assert test_collator.crop_len == "all"
    assert cfg.datamodule.train.dataloader["batch_size"] == 1
    assert cfg.datamodule.test.dataloader["batch_size"] == 1
    assert cfg.datamodule.train.dataset["audio_channel"] == "first"
    assert cfg.datamodule.test.dataset["audio_channel"] == "first"
