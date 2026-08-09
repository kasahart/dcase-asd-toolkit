import os
from pathlib import Path

import pytest
import torch
from torch import nn

from asdkit.frontends.pretrained_feature.beats import (
    BEATsFrequencyPoolingFrozenModel,
    BEATsFrozenModel,
)


class _DummyBEATs(nn.Module):
    def __init__(self, time_patches=3, frequency_patches=8, dimension=768):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.time_patches = time_patches
        self.frequency_patches = frequency_patches
        self.dimension = dimension

    def _sequence(self, wave):
        length = self.time_patches * self.frequency_patches
        return torch.arange(
            wave.shape[0] * length * self.dimension,
            device=wave.device,
            dtype=wave.dtype,
        ).reshape(wave.shape[0], length, self.dimension)

    def extract_features(self, wave):
        return self._sequence(wave), None

    def extract_features_with_grid(self, wave):
        return (
            self._sequence(wave),
            None,
            (self.time_patches, self.frequency_patches),
        )


def test_existing_beats_frozen_model_still_returns_global_embedding(monkeypatch):
    monkeypatch.setattr(
        BEATsFrozenModel, "construct_model", lambda self, **kwargs: _DummyBEATs()
    )
    frontend = BEATsFrozenModel(model_cfg={})
    output = frontend.extract({"wave": torch.zeros(2, 100)})
    assert output["embed"].shape == (2, 768)
    assert set(output) == {"embed"}


@pytest.mark.parametrize("pooling", ["mean", "rdp"])
def test_frequency_frontend_compact_output(monkeypatch, pooling):
    monkeypatch.setattr(
        BEATsFrozenModel, "construct_model", lambda self, **kwargs: _DummyBEATs()
    )
    frontend = BEATsFrequencyPoolingFrozenModel(
        model_cfg={}, pooling=pooling, gamma=4, emit_patch_sequence=False
    )
    output = frontend.extract({"wave": torch.zeros(2, 100)})
    assert output["embed_freq"].shape == (2, 8, 768)
    assert output["embed"].shape == (2, 8 * 768)
    assert "patch_sequence" not in output


def test_frequency_frontend_can_emit_patch_sequence_for_debugging(monkeypatch):
    monkeypatch.setattr(
        BEATsFrozenModel, "construct_model", lambda self, **kwargs: _DummyBEATs()
    )
    frontend = BEATsFrequencyPoolingFrozenModel(
        model_cfg={}, pooling="mean", emit_patch_sequence=True
    )
    output = frontend.extract({"wave": torch.zeros(1, 100)})
    assert output["patch_sequence"].shape == (1, 3 * 8, 768)


def test_frequency_frontend_rejects_unknown_pooling():
    with pytest.raises(ValueError, match="pooling"):
        BEATsFrequencyPoolingFrozenModel(model_cfg={}, pooling="max")


def test_actual_beats_iter3_frequency_shape_when_checkpoint_exists():
    checkpoint = Path(
        os.environ.get(
            "BEATS_ITER3_CHECKPOINT", "pretrained_models/beats/BEATs_iter3.pt"
        )
    )
    if not checkpoint.exists():
        pytest.skip(
            "BEATs_iter3 checkpoint is not available locally; set "
            "BEATS_ITER3_CHECKPOINT to an existing file"
        )
    frontend = BEATsFrequencyPoolingFrozenModel(
        model_cfg={"ckpt_path": str(checkpoint)},
        pooling="mean",
        emit_patch_sequence=False,
    )
    frontend.model.eval()

    original_extract = frontend.model.extract_features_with_grid
    observed_grid_shapes = []

    def extract_and_record_grid(*args, **kwargs):
        result = original_extract(*args, **kwargs)
        observed_grid_shapes.append(result[2])
        return result

    frontend.model.extract_features_with_grid = extract_and_record_grid
    time_patches = {}
    with torch.no_grad():
        for duration_sec in [6, 10, 12, 16]:
            batch = {"wave": torch.zeros(1, duration_sec * 16000)}
            duration_grids = []
            for pooling, gamma in [("mean", 4), ("rdp", 4), ("rdp", 8)]:
                frontend.pooling = pooling
                frontend.gamma = gamma
                output = frontend.extract(batch)
                duration_grids.append(observed_grid_shapes[-1])

                assert output["embed_freq"].ndim == 3
                assert output["embed_freq"].shape == (1, 8, 768)
                assert output["embed"].shape == (1, 8 * 768)
                assert torch.isfinite(output["embed_freq"]).all()
                assert torch.isfinite(output["embed"]).all()
                assert "patch_sequence" not in output

            assert len(set(duration_grids)) == 1
            time_patches[duration_sec] = duration_grids[0][0]
            assert duration_grids[0][1] == 8

    assert list(time_patches.values()) == sorted(time_patches.values())
    assert len(set(time_patches.values())) == 4
