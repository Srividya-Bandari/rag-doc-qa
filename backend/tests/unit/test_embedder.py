"""Unit tests for embedder — OpenAI client is mocked, no network calls."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from services.embedder import _l2_normalize, embed_chunks


# ── Helper ────────────────────────────────────────────────────────────────────

def _fake_embedding(dim: int = 8) -> list[float]:
    rng = np.random.default_rng(42)
    return rng.random(dim).tolist()


def _mock_openai_response(texts: list[str], dim: int = 8) -> MagicMock:
    response = MagicMock()
    response.data = [
        MagicMock(embedding=_fake_embedding(dim)) for _ in texts
    ]
    return response


# ── L2 normalisation ──────────────────────────────────────────────────────────

def test_l2_normalize_unit_length() -> None:
    vectors = [[3.0, 4.0], [1.0, 0.0]]
    normed = _l2_normalize(vectors)
    norms = np.linalg.norm(normed, axis=1)
    np.testing.assert_allclose(norms, np.ones(len(vectors)), atol=1e-6)


def test_l2_normalize_zero_vector_stays_zero() -> None:
    vectors = [[0.0, 0.0, 0.0]]
    normed = _l2_normalize(vectors)
    assert normed.shape == (1, 3)
    # zero-vector norm treated as 1 → output stays zero
    np.testing.assert_array_equal(normed[0], [0.0, 0.0, 0.0])


# ── Batching ──────────────────────────────────────────────────────────────────

def test_embed_chunks_calls_api_in_batches() -> None:
    """With batch_size=3 and 7 chunks, exactly 3 API calls should be made."""
    chunks = [{"chunk_text": f"text {i}"} for i in range(7)]

    with (
        patch("services.embedder.openai.OpenAI") as mock_openai_cls,
        patch("services.embedder.get_settings") as mock_settings,
    ):
        mock_settings.return_value = MagicMock(
            openai_api_key="test-key",
            embedding_model="text-embedding-3-small",
            embedding_batch_size=3,
        )
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        # Each batch returns embeddings of dim 8
        def side_effect(model, input):  # noqa: A002
            return _mock_openai_response(input, dim=8)

        mock_client.embeddings.create.side_effect = side_effect

        result = embed_chunks(chunks)

    assert mock_client.embeddings.create.call_count == 3  # ceil(7/3)
    assert result.shape == (7, 8)


def test_embed_chunks_output_is_l2_normalized() -> None:
    chunks = [{"chunk_text": f"text {i}"} for i in range(4)]

    with (
        patch("services.embedder.openai.OpenAI") as mock_openai_cls,
        patch("services.embedder.get_settings") as mock_settings,
    ):
        mock_settings.return_value = MagicMock(
            openai_api_key="test-key",
            embedding_model="text-embedding-3-small",
            embedding_batch_size=64,
        )
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.embeddings.create.side_effect = lambda model, input: _mock_openai_response(input, dim=8)  # noqa: A002

        result = embed_chunks(chunks)

    norms = np.linalg.norm(result, axis=1)
    np.testing.assert_allclose(norms, np.ones(len(chunks)), atol=1e-6)


def test_embed_chunks_preserves_order() -> None:
    """Embeddings returned must stay aligned to input chunk order.

    Strategy: assign each chunk an axis-aligned unit vector [0..0, 1, 0..0].
    After L2 normalization these remain axis-aligned, so we can identify
    which chunk maps to which row by its argmax.
    """
    n = 5
    chunks = [{"chunk_text": f"chunk {i}"} for i in range(n)]
    # Each chunk i → standard basis vector e_i (dim = n)
    fixed_embeddings = [
        [1.0 if j == i else 0.0 for j in range(n)]
        for i in range(n)
    ]

    with (
        patch("services.embedder.openai.OpenAI") as mock_openai_cls,
        patch("services.embedder.get_settings") as mock_settings,
    ):
        mock_settings.return_value = MagicMock(
            openai_api_key="test-key",
            embedding_model="text-embedding-3-small",
            embedding_batch_size=64,
        )
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        def ordered_side_effect(model, input):  # noqa: A002
            start = int(input[0].split()[-1])
            response = MagicMock()
            response.data = [
                MagicMock(embedding=fixed_embeddings[start + j])
                for j in range(len(input))
            ]
            return response

        mock_client.embeddings.create.side_effect = ordered_side_effect
        result = embed_chunks(chunks)

    # Row i should point along axis i → argmax of each row == i
    for i in range(n):
        assert np.argmax(result[i]) == i, (
            f"Row {i} argmax is {np.argmax(result[i])}, expected {i} — order is wrong"
        )
