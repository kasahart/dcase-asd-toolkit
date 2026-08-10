import pytest
import torch
from torch import nn

from asdkit.models.pretrained_models.beats.BEATs import BEATs


class _CountingPatchEmbedding(nn.Conv2d):
    def __init__(self):
        super().__init__(1, 4, kernel_size=2, stride=2, bias=False)
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        return super().forward(x)


class _IdentityEncoder(nn.Module):
    def forward(self, x, padding_mask=None):
        return x, []


def _tiny_beats():
    model = BEATs.__new__(BEATs)
    nn.Module.__init__(model)
    model.patch_embedding = _CountingPatchEmbedding()
    model.layer_norm = nn.Identity()
    model.post_extract_proj = None
    model.dropout_input = nn.Identity()
    model.encoder = _IdentityEncoder()
    model.predictor = None
    model.preprocess = lambda source, **kwargs: source
    return model


def test_beats_grid_api_captures_shape_without_second_patch_pass():
    model = _tiny_beats()
    sequence, padding_mask, grid_shape = model.extract_features_with_grid(
        torch.zeros(2, 6, 4)
    )
    assert padding_mask is None
    assert grid_shape == (3, 2)
    assert sequence.shape == (2, 6, 4)
    assert model.patch_embedding.calls == 1

    longer_sequence, _, longer_grid = model.extract_features_with_grid(
        torch.zeros(2, 10, 4)
    )
    assert longer_grid == (5, 2)
    assert longer_sequence.shape == (2, 10, 4)


def test_existing_fbank_api_return_shape_is_unchanged():
    model = _tiny_beats()
    output = model.extract_features_from_fbank(torch.zeros(2, 6, 4))
    assert isinstance(output, tuple)
    assert len(output) == 2
    assert output[0].shape == (2, 6, 4)


def test_patch_padding_mask_repeats_each_time_mask_across_frequency():
    model = _tiny_beats()
    padding_mask = torch.tensor(
        [
            [False, False, False, False, False, True],
            [False, False, False, False, True, True],
        ]
    )

    _, sequence_padding_mask, grid_shape = model.extract_features_with_grid(
        torch.zeros(2, 6, 4), padding_mask=padding_mask
    )

    assert grid_shape == (3, 2)
    torch.testing.assert_close(
        sequence_padding_mask,
        torch.tensor(
            [
                [False, False, False, False, False, False],
                [False, False, False, False, True, True],
            ]
        ),
    )


def test_patch_padding_mask_uses_convolution_windows_for_remainder_frames():
    model = _tiny_beats()
    model.patch_embedding = nn.Conv2d(
        1, 4, kernel_size=16, stride=16, bias=False
    )
    padding_mask = torch.tensor([[False] * 17 + [True] * 18])

    _, sequence_padding_mask, grid_shape = model.extract_features_with_grid(
        torch.zeros(1, 35, 32), padding_mask=padding_mask
    )

    assert grid_shape == (2, 2)
    torch.testing.assert_close(
        sequence_padding_mask,
        torch.tensor([[False, False, False, False]]),
    )


def test_grid_api_rejects_finetuned_predictor_without_patch_execution():
    model = _tiny_beats()
    model.predictor = nn.Linear(4, 3)

    with pytest.raises(ValueError, match="sequence-output BEATs"):
        model.extract_features_with_grid(torch.zeros(2, 6, 4))

    assert model.patch_embedding.calls == 0
