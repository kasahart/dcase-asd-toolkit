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


@pytest.mark.parametrize("mode", ["mean", "rdp"])
def test_zero_band_preserves_uniform_input_gradients(mode):
    x = torch.zeros(1, 4, 2, 3, requires_grad=True)

    pooled = frequency_pooling(x, mode=mode, gamma=4)
    pooled.sum().backward()

    torch.testing.assert_close(pooled, torch.zeros_like(pooled))
    torch.testing.assert_close(x.grad, torch.full_like(x, 0.25))


def test_rdp_float16_constant_sequence_has_finite_gradient():
    x = torch.full((1, 4, 2, 3), 2.5, dtype=torch.float16, requires_grad=True)

    pooled, weights = relative_deviation_pooling(
        x, gamma=8, return_weights=True
    )
    pooled.sum().backward()

    assert torch.isfinite(pooled).all()
    assert torch.isfinite(weights).all()
    assert torch.isfinite(x.grad).all()
    torch.testing.assert_close(pooled, x.detach()[:, 0])


def test_rdp_large_gamma_uses_finite_log_space_weights():
    x = torch.tensor([0.0, 0.0, 0.0, 10.0]).reshape(1, 4, 1, 1)

    pooled, weights = relative_deviation_pooling(
        x, gamma=200, return_weights=True
    )

    assert torch.isfinite(pooled).all()
    assert torch.isfinite(weights).all()
    torch.testing.assert_close(weights.sum(dim=1), torch.ones(1, 1))
    assert weights[0, 3, 0] > weights[0, 0, 0]


@pytest.mark.parametrize("mode", ["mean", "rdp"])
def test_low_precision_pooling_accumulates_in_float32(mode):
    x = torch.full(
        (1, 8, 2, 3), 10_000.0, dtype=torch.float16, requires_grad=True
    )

    pooled = frequency_pooling(x, mode=mode, gamma=4)
    pooled.float().sum().backward()

    assert pooled.dtype == x.dtype
    assert torch.isfinite(pooled).all()
    assert torch.isfinite(x.grad).all()
    torch.testing.assert_close(
        pooled.float(), torch.full((1, 2, 3), 10_000.0)
    )


@pytest.mark.parametrize("mode", ["mean", "rdp"])
def test_bfloat16_pooling_avoids_float32_range_overflow(mode):
    x = torch.full(
        (1, 2, 1, 1), 2e38, dtype=torch.bfloat16, requires_grad=True
    )

    pooled = frequency_pooling(x, mode=mode, gamma=4)
    pooled.float().sum().backward()

    assert pooled.dtype == x.dtype
    assert torch.isfinite(pooled).all()
    assert torch.isfinite(x.grad).all()
    torch.testing.assert_close(pooled, x.detach()[:, 0])


@pytest.mark.parametrize(
    ("dtype", "value"),
    [(torch.float32, 2e38), (torch.float64, 1e308)],
)
@pytest.mark.parametrize("mode", ["mean", "rdp"])
def test_full_precision_pooling_uses_overflow_safe_mean(dtype, value, mode):
    x = torch.full((1, 2, 1, 1), value, dtype=dtype, requires_grad=True)

    pooled = frequency_pooling(x, mode=mode, gamma=4)
    pooled.sum().backward()

    assert torch.isfinite(pooled).all()
    assert torch.isfinite(x.grad).all()
    torch.testing.assert_close(pooled, x.detach()[:, 0])


@pytest.mark.parametrize("mode", ["mean", "rdp"])
def test_float32_pooling_avoids_rounded_weight_overflow(mode):
    value = torch.finfo(torch.float32).max
    x = torch.full((1, 43, 1, 1), value, requires_grad=True)

    pooled = frequency_pooling(x, mode=mode, gamma=4)
    pooled.sum().backward()

    assert torch.isfinite(pooled).all()
    assert torch.isfinite(x.grad).all()
    torch.testing.assert_close(pooled, x.detach()[:, 0])


@pytest.mark.parametrize(
    ("dtype", "value"),
    [(torch.float32, 3e38), (torch.float64, 1e308)],
)
def test_rdp_scales_deviations_before_vector_norm(dtype, value):
    x = torch.tensor(
        [[[[value, value]], [[-value, -value]]]],
        dtype=dtype,
    )

    pooled, weights = relative_deviation_pooling(
        x, gamma=4, return_weights=True
    )

    assert torch.isfinite(pooled).all()
    assert torch.isfinite(weights).all()
    torch.testing.assert_close(pooled, torch.zeros_like(pooled))
    torch.testing.assert_close(weights, torch.full_like(weights, 0.5))


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


def test_rdp_masks_values_before_band_scaling():
    x = torch.tensor([1e-30, 2e-30, 3e38]).reshape(1, 3, 1, 1).requires_grad_()
    mask = torch.tensor([[True, True, False]])

    pooled, weights = relative_deviation_pooling(
        x, gamma=4, eps=1e-35, valid_time_mask=mask, return_weights=True
    )
    expected, expected_weights = relative_deviation_pooling(
        x[:, :2], gamma=4, eps=1e-35, return_weights=True
    )
    pooled.sum().backward()

    assert torch.isfinite(pooled).all()
    assert torch.isfinite(weights).all()
    assert torch.isfinite(x.grad).all()
    torch.testing.assert_close(pooled, expected)
    torch.testing.assert_close(weights[:, :2], expected_weights)
    torch.testing.assert_close(weights[:, 2], torch.zeros_like(weights[:, 2]))


def test_rdp_rejects_band_without_valid_time_patch():
    x = torch.ones(1, 2, 2, 1)
    mask = torch.tensor([[[True, False], [True, False]]])
    with pytest.raises(ValueError, match="needs a valid time patch"):
        relative_deviation_pooling(x, valid_time_mask=mask)


@pytest.mark.parametrize("mode", ["mean", "rdp"])
def test_frequency_pooling_rejects_empty_time_axis(mode):
    x = torch.empty(1, 0, 2, 3)
    with pytest.raises(ValueError, match="at least one time patch"):
        frequency_pooling(x, mode=mode)


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
    with pytest.raises(ValueError) as error:
        sequence_to_time_frequency(torch.ones(1, 11, 3), (3, 4))

    message = str(error.value)
    assert "L=11" in message
    assert "T_p=3" in message
    assert "F_p=4" in message
    assert "expected_length=12" in message
