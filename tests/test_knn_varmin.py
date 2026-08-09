import logging

import numpy as np
import pytest

from asdkit.backends import KnnVarianceMin
from asdkit.backends.beam import cosine_distance


def _dict(embeddings, paths, is_normal=None, is_target=None):
    length = len(embeddings)
    result = {
        "embed": np.asarray(embeddings, dtype=np.float32),
        "path": np.asarray(paths),
        "section": np.zeros(length, dtype=np.int64),
    }
    if is_normal is not None:
        result["is_normal"] = np.asarray(is_normal)
    if is_target is not None:
        result["is_target"] = np.asarray(is_target)
    return result


def _circle(count):
    angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
    return np.stack([np.cos(angles), np.sin(angles)], axis=1).astype(np.float32)


def test_exact_scaled_cosine_for_2d_embeddings():
    references = _circle(5)
    train = _dict(references, [f"r{i}" for i in range(5)], np.ones(5))
    query = np.array([[1.0, 0.0]], dtype=np.float32)
    backend = KnnVarianceMin()
    backend.fit(train)
    raw = backend.anomaly_score(_dict(query, ["query"]))["raw"]
    np.testing.assert_allclose(raw, cosine_distance(query, references).min(axis=1))


def test_shared_source_target_normal_memory_and_normal_filtering():
    references = _circle(6)
    train = _dict(
        np.concatenate([references, [[1.0, 1.0]]]),
        [f"normal-{i}" for i in range(6)] + ["anomaly"],
        [1, 1, 1, 1, 1, 1, 0],
        [0, 1, 0, 1, 0, 1, 1],
    )
    backend = KnnVarianceMin()
    backend.fit(train)

    source_query = _dict(references[0:1], ["source-query"])
    target_query = _dict(references[1:2], ["target-query"])
    np.testing.assert_allclose(backend.anomaly_score(source_query)["raw"], [0.0])
    np.testing.assert_allclose(backend.anomaly_score(target_query)["raw"], [0.0])

    anomaly_query = _dict(np.array([[1.0, 1.0]], dtype=np.float32), ["query"])
    assert backend.anomaly_score(anomaly_query)["raw"][0] > 0


def test_k_shortage_uses_other_references_with_warning(caplog):
    references = np.array(
        [[1.0, 0.0], [0.5, np.sqrt(3) / 2], [-1.0, 0.0]], dtype=np.float32
    )
    backend = KnnVarianceMin(rescale_k=4)
    with caplog.at_level(logging.WARNING):
        backend.fit(_dict(references, ["a", "b", "c"], np.ones(3)))
    memory = backend._engine.memory[0]
    expected = cosine_distance(references, references)
    np.fill_diagonal(expected, np.inf)
    expected.sort(axis=1)
    np.testing.assert_allclose(memory.local_density[:, 0], expected[:, :2].mean(axis=1))
    assert "effective_k=2" in caplog.text


def test_train_all_loo_alpha_matches_closed_form():
    rng = np.random.default_rng(101)
    references = rng.normal(size=(12, 5)).astype(np.float32)
    backend = KnnVarianceMin(rescale_k=4, chunk_size=3)
    backend.fit(_dict(references, [f"r{i}" for i in range(12)], np.ones(12)))
    memory = backend._engine.memory[0]

    distance = cosine_distance(memory.embeddings[:, 0], memory.embeddings[:, 0])
    np.fill_diagonal(distance, np.inf)
    nearest = distance.argmin(axis=1)
    raw = distance[np.arange(len(distance)), nearest]
    selected_density = memory.local_density[nearest, 0]
    expected_alpha = np.mean(
        (raw - raw.mean()) * (selected_density - selected_density.mean())
    ) / np.mean((selected_density - selected_density.mean()) ** 2)
    np.testing.assert_allclose(memory.alpha, [expected_alpha], rtol=1e-5, atol=1e-6)


def test_zero_density_variance_sets_alpha_to_zero(caplog):
    references = _circle(5)
    backend = KnnVarianceMin(rescale_k=4)
    with caplog.at_level(logging.WARNING):
        backend.fit(_dict(references, [f"r{i}" for i in range(5)], np.ones(5)))
    np.testing.assert_array_equal(backend._engine.memory[0].alpha, [0.0])
    assert "denominator <= eps" in caplog.text


def test_adjusted_neighbor_can_differ_and_negative_score_is_preserved():
    references = _circle(5)
    backend = KnnVarianceMin()
    backend.fit(_dict(references, [f"r{i}" for i in range(5)], np.ones(5)))
    memory = backend._engine.memory[0]
    memory.local_density[:, 0] = np.array([0.0, 1.0, 0.0, 0.0, 0.0])
    memory.alpha[0] = 1.0

    query = _dict(np.array([[1.0, 0.0]], dtype=np.float32), ["query"])
    scores = backend.anomaly_score(query)
    distance = cosine_distance(query["embed"], memory.embeddings[:, 0])[0]
    adjusted = distance - memory.alpha[0] * memory.local_density[:, 0]
    assert distance.argmin() == 0
    assert adjusted.argmin() == 1
    np.testing.assert_allclose(scores["raw"], [0.0], atol=1e-6)
    assert scores["main"][0] < 0
    assert scores["rescale_delta"][0] > 0


def test_training_scores_exclude_self_by_path():
    references = _circle(5)
    train = _dict(references, [f"r{i}" for i in range(5)], np.ones(5))
    backend = KnnVarianceMin()
    backend.fit(train)
    assert np.all(backend.anomaly_score(train)["raw"] > 0)


def test_chunk_size_does_not_change_2d_scores():
    rng = np.random.default_rng(103)
    references = rng.normal(size=(12, 7)).astype(np.float32)
    queries = rng.normal(size=(5, 7)).astype(np.float32)
    train = _dict(references, [f"r{i}" for i in range(12)], np.ones(12))
    test = _dict(queries, [f"q{i}" for i in range(5)])
    small = KnnVarianceMin(chunk_size=1)
    large = KnnVarianceMin(chunk_size=128)
    small.fit(train)
    large.fit(train)
    for key in ["main", "raw", "rescale_delta"]:
        np.testing.assert_allclose(
            small.anomaly_score(test)[key], large.anomaly_score(test)[key], atol=1e-6
        )


def test_rejects_non_2d_and_unsupported_protocols():
    with pytest.raises(ValueError, match=r"\[N, D\]"):
        KnnVarianceMin().fit(
            _dict(np.ones((5, 1, 2), dtype=np.float32), [f"r{i}" for i in range(5)])
        )
    with pytest.raises(NotImplementedError, match="scaled_cosine"):
        KnnVarianceMin(distance="euclidean")
    with pytest.raises(NotImplementedError, match="variance_minimum"):
        KnnVarianceMin(rescaling="ratio")
