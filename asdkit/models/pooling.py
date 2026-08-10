"""Encoder-independent pooling utilities for time-frequency embeddings."""

import math
from typing import Optional, Tuple, Union

import torch


def sequence_to_time_frequency(
    x_seq: torch.Tensor, grid_shape: Tuple[int, int]
) -> torch.Tensor:
    """Restore a row-major ``(time, frequency)`` patch sequence to a grid."""
    if x_seq.ndim != 3:
        raise ValueError(
            f"x_seq must have shape [B, L, D], but got {tuple(x_seq.shape)}"
        )
    if len(grid_shape) != 2:
        raise ValueError(f"grid_shape must be (T, F), but got {grid_shape}")
    time_patches, frequency_patches = grid_shape
    if time_patches <= 0 or frequency_patches <= 0:
        raise ValueError(f"grid dimensions must be positive, but got {grid_shape}")
    expected_length = time_patches * frequency_patches
    if x_seq.shape[1] != expected_length:
        raise ValueError(
            "Patch sequence length does not match its grid: "
            f"L={x_seq.shape[1]}, T_p={time_patches}, "
            f"F_p={frequency_patches}, expected_length={expected_length}"
        )
    return x_seq.reshape(
        x_seq.shape[0], time_patches, frequency_patches, x_seq.shape[2]
    )


def _expand_valid_time_mask(
    valid_time_mask: Optional[torch.Tensor], x_tf: torch.Tensor
) -> torch.Tensor:
    batch, time, frequency, _ = x_tf.shape
    if time == 0:
        raise ValueError("x_tf must contain at least one time patch")
    if valid_time_mask is None:
        return torch.ones(
            (batch, time, frequency), dtype=torch.bool, device=x_tf.device
        )
    if valid_time_mask.dtype != torch.bool:
        raise TypeError("valid_time_mask must have dtype torch.bool")
    if valid_time_mask.device != x_tf.device:
        raise ValueError("valid_time_mask and x_tf must be on the same device")
    if valid_time_mask.shape == (batch, time):
        mask = valid_time_mask.unsqueeze(-1).expand(-1, -1, frequency)
    elif valid_time_mask.shape == (batch, time, frequency):
        mask = valid_time_mask
    else:
        raise ValueError(
            "valid_time_mask must have shape [B, T] or [B, T, F], but got "
            f"{tuple(valid_time_mask.shape)}"
        )
    if not mask.any(dim=1).all():
        raise ValueError("Every sample and frequency band needs a valid time patch")
    return mask


def relative_deviation_pooling(
    x_tf: torch.Tensor,
    gamma: float = 4.0,
    eps: float = 1e-8,
    valid_time_mask: Optional[torch.Tensor] = None,
    return_weights: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Apply Relative Deviation Pooling over time independently per band.

    Args:
        x_tf: Floating-point tensor with shape ``[B, T, F, D]``.
        gamma: Non-negative deviation emphasis exponent.
        eps: Threshold below which the maximum deviation is treated as zero.
        valid_time_mask: Optional boolean ``[B, T]`` or ``[B, T, F]`` mask.
        return_weights: Also return the normalized ``[B, T, F]`` weights.
    """
    if x_tf.ndim != 4:
        raise ValueError(f"x_tf must have shape [B, T, F, D], got {tuple(x_tf.shape)}")
    if not x_tf.is_floating_point():
        raise TypeError(f"x_tf must be floating point, got {x_tf.dtype}")
    if not torch.isfinite(x_tf).all():
        raise ValueError("x_tf contains NaN or Inf")
    if isinstance(gamma, bool) or not isinstance(gamma, (int, float)):
        raise TypeError("gamma must be a finite non-negative number")
    gamma = float(gamma)
    if not math.isfinite(gamma) or gamma < 0:
        raise ValueError(f"gamma must be finite and non-negative, got {gamma}")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError(f"eps must be finite and positive, got {eps}")

    mask = _expand_valid_time_mask(valid_time_mask, x_tf)
    compute_dtype = (
        torch.float64
        if x_tf.dtype == torch.bfloat16
        else torch.float32
        if x_tf.dtype == torch.float16
        else x_tf.dtype
    )
    x_compute = x_tf.to(dtype=compute_dtype)
    mask_value = mask.unsqueeze(-1).to(dtype=compute_dtype)
    valid_count = mask_value.sum(dim=1)
    mean_weight = mask_value / valid_count.unsqueeze(1)
    valid_x = x_compute.masked_fill(~mask.unsqueeze(-1), 0)
    band_scale = valid_x.abs().amax(dim=(1, 3), keepdim=True)
    safe_band_scale = torch.where(
        band_scale > 0, band_scale, torch.ones_like(band_scale)
    )
    x_scaled = valid_x / safe_band_scale
    mean_scaled = (x_scaled * mean_weight).sum(dim=1)

    distance = torch.linalg.vector_norm(
        x_scaled - mean_scaled.unsqueeze(1), dim=-1
    )
    distance = distance.masked_fill(~mask, 0)
    max_distance = distance.amax(dim=1, keepdim=True)
    scaled_eps = eps / safe_band_scale.squeeze(-1)
    has_deviation = max_distance > scaled_eps
    safe_max_distance = torch.where(
        has_deviation, max_distance, torch.ones_like(max_distance)
    )
    normalized_distance = torch.where(
        has_deviation,
        distance / safe_max_distance,
        torch.zeros_like(distance),
    )
    weight_dtype = (
        torch.float64
        if gamma > torch.finfo(torch.float32).max
        else compute_dtype
    )
    log_weight = torch.log1p(normalized_distance.to(dtype=weight_dtype)) * gamma
    log_weight = log_weight.masked_fill(~mask, -torch.inf)
    weight_compute = torch.softmax(log_weight, dim=1).to(dtype=compute_dtype)
    pooled_scaled = (x_scaled * weight_compute.unsqueeze(-1)).sum(dim=1)
    # A convex combination is mathematically in [-1, 1], but rounded weights
    # can sum to slightly more than one and overflow when rescaled by finfo.max.
    pooled = (
        pooled_scaled.clamp(min=-1, max=1) * band_scale.squeeze(1)
    ).to(x_tf.dtype)
    weight = weight_compute.to(dtype=x_tf.dtype)

    if not torch.isfinite(pooled).all() or not torch.isfinite(weight).all():
        raise FloatingPointError("RDP produced a non-finite result")
    if return_weights:
        return pooled, weight
    return pooled


def frequency_pooling(
    x_tf: torch.Tensor,
    mode: str = "rdp",
    gamma: float = 4.0,
    eps: float = 1e-8,
    valid_time_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Pool ``[B, T, F, D]`` over time while retaining frequency bands."""
    if mode not in {"mean", "rdp"}:
        raise ValueError(f"pooling mode must be 'mean' or 'rdp', got {mode!r}")
    if mode == "rdp":
        pooled = relative_deviation_pooling(
            x_tf,
            gamma=gamma,
            eps=eps,
            valid_time_mask=valid_time_mask,
        )
        assert isinstance(pooled, torch.Tensor)
        return pooled

    if x_tf.ndim != 4:
        raise ValueError(f"x_tf must have shape [B, T, F, D], got {tuple(x_tf.shape)}")
    if not x_tf.is_floating_point():
        raise TypeError(f"x_tf must be floating point, got {x_tf.dtype}")
    if not torch.isfinite(x_tf).all():
        raise ValueError("x_tf contains NaN or Inf")
    mask = _expand_valid_time_mask(valid_time_mask, x_tf)
    compute_dtype = (
        torch.float64
        if x_tf.dtype == torch.bfloat16
        else torch.float32
        if x_tf.dtype == torch.float16
        else x_tf.dtype
    )
    x_compute = x_tf.to(dtype=compute_dtype)
    mask_value = mask.unsqueeze(-1).to(dtype=compute_dtype)
    mean_weight = mask_value / mask_value.sum(dim=1).unsqueeze(1)
    valid_x = x_compute.masked_fill(~mask.unsqueeze(-1), 0)
    band_scale = valid_x.abs().amax(dim=(1, 3), keepdim=True)
    safe_band_scale = torch.where(
        band_scale > 0, band_scale, torch.ones_like(band_scale)
    )
    mean_scaled = ((valid_x / safe_band_scale) * mean_weight).sum(dim=1)
    return (
        mean_scaled.clamp(min=-1, max=1) * band_scale.squeeze(1)
    ).to(x_tf.dtype)
