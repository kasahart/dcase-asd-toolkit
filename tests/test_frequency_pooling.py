import pytest
import torch

from asdkit.models.pooling import (
    frequency_pooling,
    relative_deviation_pooling,
    sequence_to_time_frequency,
)


def test_rdp_gamma_zero_matches_time_mean():
    x = torch.randn(3, 7, 4, 5, dtype=torch.float32)
    pooled = relative_deviation_pooling(x, gamma=0)
    torch.testing.assert_close(pooled, x.mean(dim=1))
    assert pooled.dtype == x.dtype


def test_rdp_constant_sequence_is_finite_and_matches_mean():
    x = torch.full((2, 6, 3, 4), 2.5)
    pooled, weights = relative_deviation_pooling(x, gamma=8, return_weights=True)
    assert torch.isfinite(pooled).all()
    assert torch.isfinite(weights).all()
    torch.testing.assert_close(pooled, x.mean(dim=1))


def test_rdp_emphasizes_transient_patch():
    x = torch.tensor([0.0, 0.0, 0.0, 10.0]).reshape(1, 4, 1, 1)
    pooled, weights = relative_deviation_pooling(x, gamma=4, return_weights=True)
    assert weights[0, 3, 0] > weights[0, 0, 0]
    assert pooled.item() > x.mean().item()


def test_rdp_valid_time_mask():
    x = torch.tensor([1.0, 3.0, 100.0]).reshape(1, 3, 1, 1)
    mask = torch.tensor([[True, True, False]])
    pooled = frequency_pooling(x, mode="mean", valid_time_mask=mask)
    torch.testing.assert_close(pooled, torch.tensor([[[2.0]]]))


def test_rdp_valid_time_mask_can_differ_by_frequency():
    x = torch.tensor([1.0, 10.0, 3.0, 30.0]).reshape(1, 2, 2, 1)
    mask = torch.tensor([[[True, False], [True, True]]])
    pooled = frequency_pooling(x, mode="mean", valid_time_mask=mask)
    torch.testing.assert_close(pooled, torch.tensor([[[2.0], [30.0]]]))


def test_rdp_rejects_band_without_valid_time_patch():
    x = torch.ones(1, 2, 2, 1)
    mask = torch.tensor([[[True, False], [True, False]]])
    with pytest.raises(ValueError, match="needs a valid time patch"):
        relative_deviation_pooling(x, valid_time_mask=mask)


def test_rdp_rejects_bad_inputs():
    with pytest.raises(ValueError, match="gamma"):
        relative_deviation_pooling(torch.ones(1, 2, 1, 1), gamma=-1)
    with pytest.raises(ValueError, match="NaN or Inf"):
        relative_deviation_pooling(torch.tensor([[[[float("nan")]]]]))
    with pytest.raises(ValueError, match="shape"):
        relative_deviation_pooling(torch.ones(2, 3, 4))


def test_rdp_preserves_float64_dtype():
    x = torch.randn(2, 3, 4, 5, dtype=torch.float64)
    assert relative_deviation_pooling(x).dtype == torch.float64


def test_time_frequency_reshape_preserves_flatten_order():
    original = torch.arange(2 * 3 * 4 * 2).reshape(2, 3, 4, 2)
    sequence = original.reshape(2, 3 * 4, 2)
    restored = sequence_to_time_frequency(sequence, (3, 4))
    torch.testing.assert_close(restored, original)
    assert not torch.equal(restored, original.transpose(1, 2))


def test_time_frequency_reshape_checks_length():
    with pytest.raises(AssertionError, match="does not match"):
        sequence_to_time_frequency(torch.ones(1, 11, 3), (3, 4))
