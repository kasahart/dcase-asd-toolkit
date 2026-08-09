"""Band-wise Equalized Anomaly Measure with variance-minimum rescaling."""

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .base import BaseBackend

logger = logging.getLogger(__name__)


def _validate_band_embeddings(name: str, embeddings: np.ndarray) -> np.ndarray:
    array = np.asarray(embeddings)
    if array.ndim != 3:
        raise ValueError(f"{name} must have shape [N, F, D], got {array.shape}")
    if not np.issubdtype(array.dtype, np.floating):
        raise TypeError(f"{name} must have a floating dtype, got {array.dtype}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or Inf")
    return array.astype(np.float32, copy=False)


def l2_normalize(embeddings: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """L2-normalize the last axis, mapping near-zero vectors safely to zero."""
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError(f"eps must be finite and positive, got {eps}")
    array = np.asarray(embeddings)
    if not np.issubdtype(array.dtype, np.floating):
        raise TypeError(f"embeddings must have a floating dtype, got {array.dtype}")
    if not np.isfinite(array).all():
        raise ValueError("embeddings contains NaN or Inf")
    norm = np.linalg.norm(array, axis=-1, keepdims=True)
    return np.divide(array, norm, out=np.zeros_like(array), where=norm > eps)


def cosine_distance(
    query: np.ndarray, reference: np.ndarray, eps: float = 1e-12
) -> np.ndarray:
    """Calculate exact ``0.5 * (1 - cosine)`` pairwise band distances.

    Two-dimensional inputs ``[N, D]`` produce ``[Q, R]``. Three-dimensional
    inputs ``[N, F, D]`` produce ``[Q, R, F]`` and compare aligned bands only.
    Cosine similarity involving a near-zero vector is defined as zero.
    """
    query_array = np.asarray(query)
    reference_array = np.asarray(reference)
    squeeze_band = query_array.ndim == reference_array.ndim == 2
    if squeeze_band:
        query_array = query_array[:, None, :]
        reference_array = reference_array[:, None, :]
    if query_array.ndim != 3 or reference_array.ndim != 3:
        raise ValueError("query and reference must both be [N, D] or [N, F, D]")
    if query_array.shape[1:] != reference_array.shape[1:]:
        raise ValueError(
            "query and reference band shapes differ: "
            f"{query_array.shape[1:]} != {reference_array.shape[1:]}"
        )
    query_norm = l2_normalize(query_array, eps=eps)
    reference_norm = l2_normalize(reference_array, eps=eps)
    similarity = np.einsum("qfd,rfd->qrf", query_norm, reference_norm)
    distance = 0.5 * (1.0 - np.clip(similarity, -1.0, 1.0))
    return distance[:, :, 0] if squeeze_band else distance


def _cosine_distance_normalized(
    query_normalized: np.ndarray, reference_normalized: np.ndarray
) -> np.ndarray:
    similarity = np.einsum(
        "qfd,rfd->qrf", query_normalized, reference_normalized, optimize=True
    )
    return 0.5 * (1.0 - np.clip(similarity, -1.0, 1.0))


def minimum_band_scores(
    distances: np.ndarray,
    local_density: Optional[np.ndarray] = None,
    alpha: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Minimize raw or Eq. 5 adjusted distances over references."""
    distance = np.asarray(distances)
    if distance.ndim != 3:
        raise ValueError(f"distances must have shape [Q, R, F], got {distance.shape}")
    if (local_density is None) != (alpha is None):
        raise ValueError(
            "local_density and alpha must either both be set or both be None"
        )
    if local_density is None:
        return distance.min(axis=1)
    density = np.asarray(local_density)
    alpha_array = np.asarray(alpha)
    if density.shape != distance.shape[1:]:
        raise ValueError(
            f"local_density must have shape {distance.shape[1:]}, got {density.shape}"
        )
    if alpha_array.shape != (distance.shape[2],):
        raise ValueError(
            f"alpha must have shape {(distance.shape[2],)}, got {alpha_array.shape}"
        )
    adjusted = distance - alpha_array[None, None, :] * density[None, :, :]
    return adjusted.min(axis=1)


def _iter_slices(length: int, chunk_size: int):
    for start in range(0, length, chunk_size):
        yield slice(start, min(start + chunk_size, length))


def compute_local_density(
    reference_normalized: np.ndarray,
    k: int = 4,
    chunk_size: int = 128,
) -> np.ndarray:
    """Mean distance to K nearest *other* references for every band."""
    reference = _validate_band_embeddings("reference", reference_normalized)
    reference_count, frequency_count, _ = reference.shape
    if reference_count < 2:
        raise ValueError("Variance-minimum rescaling needs at least 2 references")
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError(f"k must be a positive integer, got {k!r}")
    if (
        isinstance(chunk_size, bool)
        or not isinstance(chunk_size, int)
        or chunk_size <= 0
    ):
        raise ValueError(f"chunk_size must be a positive integer, got {chunk_size!r}")
    effective_k = min(k, reference_count - 1)
    if effective_k != k:
        logger.warning(
            "Only %d references are available; using effective_k=%d instead of K=%d.",
            reference_count,
            effective_k,
            k,
        )

    density = np.empty((reference_count, frequency_count), dtype=np.float32)
    for query_slice in _iter_slices(reference_count, chunk_size):
        distance = _cosine_distance_normalized(reference[query_slice], reference)
        global_indices = np.arange(query_slice.start, query_slice.stop)
        distance[np.arange(len(global_indices)), global_indices, :] = np.inf
        nearest = np.partition(distance, effective_k - 1, axis=1)[:, :effective_k, :]
        density[query_slice] = nearest.mean(axis=1)
    if not np.isfinite(density).all():
        raise FloatingPointError("Local-density calculation produced NaN or Inf")
    return density


def estimate_train_all_alpha(
    reference_normalized: np.ndarray,
    local_density: np.ndarray,
    chunk_size: int = 128,
    eps: float = 1e-12,
) -> np.ndarray:
    """Estimate per-band alpha with TrainAll leave-one-out validation."""
    reference = _validate_band_embeddings("reference", reference_normalized)
    reference_count, frequency_count, _ = reference.shape
    density = np.asarray(local_density, dtype=np.float32)
    if density.shape != (reference_count, frequency_count):
        raise ValueError(
            f"local_density must have shape {(reference_count, frequency_count)}, "
            f"got {density.shape}"
        )
    if reference_count < 2:
        raise ValueError("TrainAll leave-one-out needs at least 2 references")
    if not np.isfinite(density).all():
        raise ValueError("local_density contains NaN or Inf")

    raw_distance = np.empty((reference_count, frequency_count), dtype=np.float32)
    selected_density = np.empty_like(raw_distance)
    bands = np.arange(frequency_count)[None, :]
    for query_slice in _iter_slices(reference_count, chunk_size):
        distance = _cosine_distance_normalized(reference[query_slice], reference)
        global_indices = np.arange(query_slice.start, query_slice.stop)
        distance[np.arange(len(global_indices)), global_indices, :] = np.inf
        nearest_indices = distance.argmin(axis=1)
        raw_distance[query_slice] = np.take_along_axis(
            distance, nearest_indices[:, None, :], axis=1
        )[:, 0, :]
        selected_density[query_slice] = density[nearest_indices, bands]

    raw_centered = raw_distance.astype(np.float64) - raw_distance.mean(
        axis=0, dtype=np.float64
    )
    density_centered = selected_density.astype(np.float64) - selected_density.mean(
        axis=0, dtype=np.float64
    )
    numerator = np.mean(raw_centered * density_centered, axis=0)
    denominator = np.mean(density_centered * density_centered, axis=0)
    alpha = np.zeros(frequency_count, dtype=np.float64)
    stable = denominator > eps
    alpha[stable] = numerator[stable] / denominator[stable]
    if not stable.all():
        logger.warning(
            "VarMin alpha denominator <= eps for bands %s; alpha is set to zero.",
            np.flatnonzero(~stable).tolist(),
        )
    if not np.isfinite(alpha).all():
        raise FloatingPointError("Alpha estimation produced NaN or Inf")
    return alpha.astype(np.float32)


class VarianceMinRescaler:
    """Fit local density and per-band variance-minimizing weights."""

    def __init__(
        self,
        k: int = 4,
        validation: str = "train_all",
        scope: str = "per_band",
        chunk_size: int = 128,
        eps: float = 1e-12,
    ):
        if validation != "train_all":
            raise NotImplementedError(
                f"rescale_validation={validation!r} is not implemented"
            )
        if scope != "per_band":
            raise NotImplementedError(f"rescale_scope={scope!r} is not implemented")
        self.k = k
        self.validation = validation
        self.scope = scope
        self.chunk_size = chunk_size
        self.eps = eps

    def fit(self, reference_normalized: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        density = compute_local_density(
            reference_normalized, k=self.k, chunk_size=self.chunk_size
        )
        alpha = estimate_train_all_alpha(
            reference_normalized,
            density,
            chunk_size=self.chunk_size,
            eps=self.eps,
        )
        return density, alpha


@dataclass
class _BandMemory:
    embeddings: np.ndarray
    paths: Optional[np.ndarray]
    local_density: np.ndarray
    alpha: np.ndarray


class BEAMVarianceMin(BaseBackend):
    """BEAM backend with optional per-band variance-minimum score rescaling.

    The canonical reproduction assumes TrainAll leave-one-out calibration and a
    separate alpha for each frequency band, applies rescaling before the uniform
    frequency mean, and minimizes the adjusted distance over all references. The
    available paper description does not fully disambiguate the per-band scope,
    so this is an explicit reproduction assumption rather than a paper-guaranteed
    detail.
    """

    diagnostic_score_keys = {"raw", "rescale_delta"}

    def __init__(
        self,
        embed_key: str = "embed_freq",
        sep_section: bool = False,
        use_rescaling: bool = True,
        rescale_k: int = 4,
        rescale_validation: str = "train_all",
        rescale_scope: str = "per_band",
        chunk_size: int = 128,
        eps: float = 1e-12,
    ):
        if (
            isinstance(chunk_size, bool)
            or not isinstance(chunk_size, int)
            or chunk_size <= 0
        ):
            raise ValueError(
                f"chunk_size must be a positive integer, got {chunk_size!r}"
            )
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError(f"eps must be finite and positive, got {eps}")
        self.embed_key = embed_key
        self.sep_section = sep_section
        self.use_rescaling = use_rescaling
        self.chunk_size = chunk_size
        self.eps = eps
        self.rescaler = VarianceMinRescaler(
            k=rescale_k,
            validation=rescale_validation,
            scope=rescale_scope,
            chunk_size=chunk_size,
            eps=eps,
        )
        self.memory: Dict[Any, _BandMemory] = {}
        self.band_shape: Optional[Tuple[int, int]] = None

    def _get_sections(self, extract_dict: dict, length: int) -> np.ndarray:
        if "section" not in extract_dict:
            if self.sep_section:
                raise KeyError("section is required when sep_section=True")
            return np.zeros(length, dtype=np.int64)
        section = np.asarray(extract_dict["section"])
        if section.shape != (length,):
            raise ValueError(f"section must have shape [{length}], got {section.shape}")
        return section if self.sep_section else np.zeros_like(section)

    @staticmethod
    def _get_paths(extract_dict: dict, length: int) -> Optional[np.ndarray]:
        if "path" not in extract_dict:
            return None
        paths = np.asarray(extract_dict["path"])
        if paths.shape != (length,):
            raise ValueError(f"path must have shape [{length}], got {paths.shape}")
        return paths

    def fit(self, train_dict: dict) -> None:
        embeddings = _validate_band_embeddings(
            self.embed_key, train_dict[self.embed_key]
        )
        length, frequency_count, dimension = embeddings.shape
        self.band_shape = (frequency_count, dimension)
        sections = self._get_sections(train_dict, length)
        paths = self._get_paths(train_dict, length)
        if paths is None:
            logger.warning(
                "Training paths are unavailable; path-based self-match exclusion "
                "will not be possible in anomaly_score()."
            )
        if "is_normal" in train_dict:
            is_normal = np.asarray(train_dict["is_normal"])
            if is_normal.shape != (length,):
                raise ValueError(
                    f"is_normal must have shape [{length}], got {is_normal.shape}"
                )
            normal = is_normal == 1
        else:
            normal = np.ones(length, dtype=bool)

        self.memory.clear()
        for section_value in np.unique(sections):
            selected = (sections == section_value) & normal
            if not selected.any():
                raise ValueError(
                    f"No normal references are available for section {section_value!r}"
                )
            reference = l2_normalize(embeddings[selected], eps=self.eps)
            if self.use_rescaling:
                local_density, alpha = self.rescaler.fit(reference)
            else:
                local_density = np.zeros(reference.shape[:2], dtype=np.float32)
                alpha = np.zeros(reference.shape[1], dtype=np.float32)
            self.memory[section_value] = _BandMemory(
                embeddings=reference,
                paths=None if paths is None else paths[selected],
                local_density=local_density,
                alpha=alpha,
            )

    def _score_section(
        self,
        query: np.ndarray,
        query_paths: Optional[np.ndarray],
        memory: _BandMemory,
    ) -> Tuple[np.ndarray, np.ndarray]:
        query_normalized = l2_normalize(query, eps=self.eps)
        query_count, frequency_count, _ = query.shape
        raw_band = np.empty((query_count, frequency_count), dtype=np.float32)
        main_band = np.empty_like(raw_band)
        if query_paths is not None and memory.paths is None:
            logger.warning(
                "Reference paths are unavailable; self matches cannot be excluded."
            )
        if query_paths is None:
            logger.warning(
                "Query paths are unavailable; self matches cannot be excluded."
            )

        for query_slice in _iter_slices(query_count, self.chunk_size):
            distance = _cosine_distance_normalized(
                query_normalized[query_slice], memory.embeddings
            )
            if query_paths is not None and memory.paths is not None:
                self_match = query_paths[query_slice, None] == memory.paths[None, :]
                distance = np.where(self_match[:, :, None], np.inf, distance)
            if np.isinf(distance).all(axis=1).any():
                raise ValueError(
                    "Self-match exclusion removed every reference for at least one query"
                )
            raw_band[query_slice] = minimum_band_scores(distance)
            if self.use_rescaling:
                main_band[query_slice] = minimum_band_scores(
                    distance,
                    local_density=memory.local_density,
                    alpha=memory.alpha,
                )
            else:
                main_band[query_slice] = raw_band[query_slice]

        if not np.isfinite(raw_band).all() or not np.isfinite(main_band).all():
            raise FloatingPointError("BEAM scoring produced NaN or Inf")
        return raw_band, main_band

    def anomaly_score(self, test_dict: dict) -> Dict[str, np.ndarray]:
        if not self.memory or self.band_shape is None:
            raise RuntimeError("fit() must be called before anomaly_score()")
        embeddings = _validate_band_embeddings(
            self.embed_key, test_dict[self.embed_key]
        )
        length = len(embeddings)
        if embeddings.shape[1:] != self.band_shape:
            raise ValueError(
                f"Expected band shape {self.band_shape}, got {embeddings.shape[1:]}"
            )
        sections = self._get_sections(test_dict, length)
        paths = self._get_paths(test_dict, length)
        raw_score = np.empty(length, dtype=np.float32)
        main_score = np.empty(length, dtype=np.float32)

        for section_value in np.unique(sections):
            if section_value not in self.memory:
                raise KeyError(f"No fitted memory for section {section_value!r}")
            selected = sections == section_value
            raw_band, main_band = self._score_section(
                embeddings[selected],
                None if paths is None else paths[selected],
                self.memory[section_value],
            )
            raw_score[selected] = raw_band.mean(axis=1)
            main_score[selected] = main_band.mean(axis=1)

        return {
            "main": main_score,
            "raw": raw_score,
            "rescale_delta": raw_score - main_score,
        }
