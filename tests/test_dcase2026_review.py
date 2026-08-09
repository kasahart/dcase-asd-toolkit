from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf

from asdkit.bin import table
from asdkit.datasets import torch_dataset
from asdkit.datasets.collators import DCASEWaveCollator
from asdkit.frontends.auroc import AUROC
from asdkit.utils.asdkit_utils.extract import restore_dataset_args
from asdkit.utils.asdkit_utils.visualize.plot import get_cfg_list_of_dict, get_u_idx


def test_auroc_compute_without_updates_returns_none():
    assert AUROC().compute() is None


@pytest.mark.parametrize("audio_channel", [1, "second", "diff", "unknown"])
def test_mono_audio_rejects_invalid_channel_requests(monkeypatch, audio_channel):
    monkeypatch.setattr(
        torch_dataset.torchaudio,
        "load",
        lambda _: (torch.zeros(1, 16), 16000),
    )

    with pytest.raises(ValueError):
        torch_dataset.torch_mono_wav_load("mono.wav", audio_channel=audio_channel)


def test_dcase2026_table_uses_single_section_metrics(tmp_path, monkeypatch):
    monkeypatch.setitem(table.MACHINE_DICT, "dcase2026-dev", ["machine"])
    machine_dir = tmp_path / "machine"
    machine_dir.mkdir()
    pd.DataFrame({"backend": ["model"], "0_auc": [0.75]}).to_csv(
        machine_dir / "test_evaluate.csv", index=False
    )

    result = table.get_table_df(tmp_path, "dcase2026", "dev", "auc")

    assert result is not None
    assert result.loc[0, "machine"] == pytest.approx(0.75)


@pytest.mark.parametrize("audio_channel", ["first", "second", "mean", "diff"])
def test_train_and_validation_audio_channels_are_aligned(audio_channel):
    config_path = (
        Path(__file__).parents[1] / "config/train/datamodule/audiofeat.yaml"
    )
    cfg = OmegaConf.create({"datamodule": OmegaConf.load(config_path)})

    cfg.datamodule.train.dataset.audio_channel = audio_channel

    assert cfg.datamodule.valid.dataset.audio_channel == audio_channel


@pytest.mark.parametrize("audio_channel", ["first", "second", "mean", "diff"])
def test_extraction_audio_channels_are_aligned(audio_channel):
    config_path = (
        Path(__file__).parents[1] / "config/extract/datamodule/default.yaml"
    )
    cfg = OmegaConf.create({"datamodule": OmegaConf.load(config_path)})

    cfg.datamodule.train.dataset.audio_channel = audio_channel

    assert cfg.datamodule.test.dataset.audio_channel == audio_channel


@pytest.mark.parametrize("audio_channel", ["first", "second", "mean", "diff"])
def test_restore_dataset_args_uses_training_audio_channel(audio_channel):
    cfg = SimpleNamespace(
        datamodule=SimpleNamespace(
            train=SimpleNamespace(dataset={"audio_channel": "first"}),
            test=SimpleNamespace(dataset={"audio_channel": "mean"}),
        )
    )
    past_cfg = SimpleNamespace(
        datamodule=SimpleNamespace(
            train=SimpleNamespace(dataset={"audio_channel": audio_channel})
        )
    )

    restore_dataset_args(cfg, past_cfg)

    assert cfg.datamodule.train.dataset["audio_channel"] == audio_channel
    assert cfg.datamodule.test.dataset["audio_channel"] == audio_channel


@pytest.mark.parametrize("duration_sec", [6, 10, 12, 16])
def test_sec_all_preserves_original_waveform(duration_sec):
    collator = DCASEWaveCollator(
        label_dict_path={}, sec="all", sr=16000, shuffle=False
    )
    wave = torch.arange(duration_sec * 16000, dtype=torch.float32)

    output = collator.crop_wave(wave)

    assert output.data_ptr() == wave.data_ptr()
    torch.testing.assert_close(output, wave)


def test_unknown_evaluation_samples_are_visualized():
    is_test = np.array([0, 1, 1])
    is_target = np.array([0, -1, 1])
    is_normal = np.array([1, -1, 0])

    unknown_idx = get_u_idx(
        is_test, is_target, is_normal, "test_unknown_unknown"
    )
    plot_cfg = get_cfg_list_of_dict(is_test, is_target, is_normal)[1]

    np.testing.assert_array_equal(unknown_idx, [False, True, False])
    np.testing.assert_array_equal(
        plot_cfg["test_unknown_unknown"]["u_idx"], unknown_idx
    )
