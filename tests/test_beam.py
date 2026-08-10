import logging

import numpy as np
import pytest

from asdkit.backends.beam import (
    BEAMVarianceMin,
    compute_local_density,
    cosine_distance,
    estimate_train_all_alpha,
    l2_normalize,
    minimum_band_scores,
)


def _dict(embeddings, paths, is_normal=None):
    length = len(embeddings)
    result = {
        "embed_freq": np.asarray(embeddings, dtype=np.float32),
        "path": np.asarray(paths),
        "section": np.zeros(length, dtype=np.int64),
    }
    if is_normal is not None:
        result["is_normal"] = np.asarray(is_normal)
    return result


def test_exact_cosine_distance():
    query = np.array([[1.0, 0.0]], dtype=np.float32)
    reference = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float32)
    np.testing.assert_allclose(cosine_distance(query, reference), [[0, 0.5, 1]])


def test_cosine_distance_handles_zero_vectors_safely():
    distance = cosine_distance(
        np.zeros((1, 2), dtype=np.float32),
        np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32),
    )
    np.testing.assert_allclose(distance, [[0.5, 0.5]])
    assert np.isfinite(distance).all()


def test_beam_chooses_different_neighbor_per_band():
    references = np.array(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.0, 1.0], [1.0, 0.0]],
        ],
        dtype=np.float32,
    )
    query = np.array([[[1.0, 0.0], [1.0, 0.0]]], dtype=np.float32)
    backend = BEAMVarianceMin(use_rescaling=False, chunk_size=1)
    backend.fit(_dict(references, ["r0", "r1"], [1, 1]))
    raw = backend.anomaly_score(_dict(query, ["query"]))["raw"]

    tied_global = cosine_distance(query.reshape(1, -1), references.reshape(2, -1))
    np.testing.assert_allclose(raw, [0.0])
    assert tied_global.min() > raw[0]


def test_coupled_beam_uses_one_reference_for_all_bands():
    references = np.array(
        [
            [[1.0, 0.0], [-1.0, 0.0]],
            [[-1.0, 0.0], [1.0, 0.0]],
        ],
        dtype=np.float32,
    )
    query = np.array([[[1.0, 0.0], [1.0, 0.0]]], dtype=np.float32)
    train = _dict(references, ["r0", "r1"], [1, 1])
    test = _dict(query, ["query"])

    independent = BEAMVarianceMin(use_rescaling=False, neighbor_mode="per_band")
    coupled = BEAMVarianceMin(use_rescaling=False, neighbor_mode="coupled")
    independent.fit(train)
    coupled.fit(train)

    np.testing.assert_allclose(independent.anomaly_score(test)["main"], [0.0])
    np.testing.assert_allclose(coupled.anomaly_score(test)["main"], [0.5])


def test_coupled_beam_rejects_unknown_neighbor_mode():
    with pytest.raises(ValueError, match="neighbor_mode"):
        BEAMVarianceMin(neighbor_mode="frequency_shuffle")


def test_local_density_excludes_self():
    references = np.array(
        [[[1.0, 0.0]], [[0.5, np.sqrt(3) / 2]], [[-1.0, 0.0]]],
        dtype=np.float32,
    )
    density = compute_local_density(l2_normalize(references), k=1, chunk_size=2)
    np.testing.assert_allclose(density[:, 0], [0.25, 0.25, 0.75], atol=1e-6)
    assert np.all(density > 0)


def test_local_density_reduces_k_with_warning(caplog):
    references = l2_normalize(
        np.array([[[1.0, 0.0]], [[0.0, 1.0]], [[-1.0, 0.0]]], dtype=np.float32)
    )
    with caplog.at_level(logging.WARNING):
        density = compute_local_density(references, k=4)
    assert density.shape == (3, 1)
    assert "effective_k=2" in caplog.text


def test_local_density_rejects_single_reference():
    with pytest.raises(ValueError, match="at least 2"):
        compute_local_density(np.ones((1, 1, 2), dtype=np.float32), k=4)


def test_train_scores_exclude_path_self_match():
    references = np.array([[[1.0, 0.0]], [[0.0, 1.0]], [[-1.0, 0.0]]], dtype=np.float32)
    train = _dict(references, ["a", "b", "c"], [1, 1, 1])
    backend = BEAMVarianceMin(use_rescaling=False)
    backend.fit(train)
    scores = backend.anomaly_score(train)["raw"]
    assert np.all(scores > 0)


def test_self_match_exclusion_rejects_query_without_remaining_reference():
    train = _dict(np.array([[[1.0, 0.0]]], dtype=np.float32), ["only"], [1])
    backend = BEAMVarianceMin(use_rescaling=False)
    backend.fit(train)
    with pytest.raises(ValueError, match="removed every reference"):
        backend.anomaly_score(train)


def test_fit_uses_only_normal_references():
    references = np.array([[[1.0, 0.0]], [[0.0, 1.0]], [[1.0, 1.0]]], dtype=np.float32)
    train = _dict(references, ["normal-a", "normal-b", "anomaly"], [1, 1, 0])
    backend = BEAMVarianceMin(use_rescaling=False)
    backend.fit(train)
    score = backend.anomaly_score(_dict(references[2:], ["external-query"]))["raw"]
    assert score[0] > 0


def test_sep_section_builds_independent_memories():
    train = _dict(
        np.array([[[1.0, 0.0]], [[0.0, 1.0]]], dtype=np.float32),
        ["section-0", "section-1"],
        [1, 1],
    )
    train["section"] = np.array([0, 1])
    query = _dict(np.array([[[0.0, 1.0]]], dtype=np.float32), ["external-query"])
    query["section"] = np.array([0])
    backend = BEAMVarianceMin(use_rescaling=False, sep_section=True)
    backend.fit(train)
    score = backend.anomaly_score(query)["raw"]
    np.testing.assert_allclose(score, [0.5])


def test_variance_min_alpha_minimizes_closed_form_objective():
    rng = np.random.default_rng(7)
    references = l2_normalize(rng.normal(size=(15, 2, 4)).astype(np.float32))
    density = compute_local_density(references, k=4, chunk_size=4)
    alpha = estimate_train_all_alpha(references, density, chunk_size=3)

    distance = cosine_distance(references, references)
    diagonal = np.arange(len(references))
    distance[diagonal, diagonal, :] = np.inf
    nearest = distance.argmin(axis=1)
    bands = np.arange(references.shape[1])[None, :]
    raw = np.take_along_axis(distance, nearest[:, None, :], axis=1)[:, 0, :]
    selected_density = density[nearest, bands]
    optimum_variance = np.var(raw - alpha * selected_density, axis=0)
    np.testing.assert_array_less(
        optimum_variance,
        np.var(raw - (alpha + 0.1) * selected_density, axis=0) + 1e-10,
    )
    np.testing.assert_array_less(
        optimum_variance,
        np.var(raw - (alpha - 0.1) * selected_density, axis=0) + 1e-10,
    )


def test_adjusted_minimum_can_change_reference():
    distances = np.array([[[0.1], [0.2]]], dtype=np.float32)
    density = np.array([[0.0], [1.0]], dtype=np.float32)
    alpha = np.array([0.2], dtype=np.float32)
    raw = minimum_band_scores(distances)
    adjusted = minimum_band_scores(distances, density, alpha)
    np.testing.assert_allclose(raw, [[0.1]])
    np.testing.assert_allclose(adjusted, [[0.0]])


def test_zero_alpha_denominator_is_safe(caplog):
    references = l2_normalize(
        np.array([[[1.0, 0.0]], [[0.0, 1.0]], [[-1.0, 0.0]]], dtype=np.float32)
    )
    with caplog.at_level(logging.WARNING):
        alpha = estimate_train_all_alpha(references, np.ones((3, 1), dtype=np.float32))
    np.testing.assert_array_equal(alpha, [0.0])
    assert "denominator <= eps" in caplog.text


def test_chunk_size_does_not_change_scores():
    rng = np.random.default_rng(11)
    references = rng.normal(size=(12, 3, 5)).astype(np.float32)
    queries = rng.normal(size=(7, 3, 5)).astype(np.float32)
    train = _dict(references, [f"train-{i}" for i in range(12)], np.ones(12))
    test = _dict(queries, [f"test-{i}" for i in range(7)])
    small = BEAMVarianceMin(chunk_size=1, rescale_k=4)
    large = BEAMVarianceMin(chunk_size=128, rescale_k=4)
    small.fit(train)
    large.fit(train)
    small_scores = small.anomaly_score(test)
    large_scores = large.anomaly_score(test)
    for key in ["main", "raw", "rescale_delta"]:
        np.testing.assert_allclose(small_scores[key], large_scores[key], atol=1e-6)


def test_backend_api_shapes_and_finite_scores():
    references = np.array(
        [
            [[1.0, 0.0], [1.0, 1.0]],
            [[0.0, 1.0], [1.0, -1.0]],
            [[-1.0, 0.0], [-1.0, 1.0]],
            [[0.0, -1.0], [-1.0, -1.0]],
            [[1.0, 1.0], [0.0, 1.0]],
        ],
        dtype=np.float32,
    )
    train = _dict(references, [f"r{i}" for i in range(5)], [1, 1, 1, 1, 1])
    backend = BEAMVarianceMin(rescale_k=4, chunk_size=2)
    backend.fit(train)
    scores = backend.anomaly_score(_dict(references[:2], ["q0", "q1"]))
    assert set(scores) == {"main", "raw", "rescale_delta"}
    for value in scores.values():
        assert value.shape == (2,)
        assert np.isfinite(value).all()
    np.testing.assert_allclose(scores["rescale_delta"], scores["raw"] - scores["main"])


def test_backend_rejects_non_finite_embeddings():
    train = _dict(np.ones((2, 1, 2), dtype=np.float32), ["a", "b"], [1, 1])
    train["embed_freq"][0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN or Inf"):
        BEAMVarianceMin().fit(train)


def test_rescaling_requires_supported_modes():
    with pytest.raises(NotImplementedError):
        BEAMVarianceMin(rescale_validation="test_all")
    with pytest.raises(NotImplementedError):
        BEAMVarianceMin(rescale_scope="clip")
