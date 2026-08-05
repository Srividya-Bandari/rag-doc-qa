"""Unit tests for vector_store — no network calls, uses tmp_path fixture."""

import json
from unittest.mock import MagicMock

import numpy as np
import pytest

from services.vector_store import load_document_index, save_document_index, search_index


def _random_unit_vectors(n: int, dim: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vecs = rng.random((n, dim)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


def _make_metadata(n: int, doc_id: str = "doc-1") -> list[dict]:
    return [
        {"document_id": doc_id, "chunk_id": i, "page_number": 1, "chunk_text": f"chunk {i}"}
        for i in range(n)
    ]


@pytest.fixture()
def patch_index_dir(tmp_path, monkeypatch):
    """Redirect FAISS index writes to a temp dir by patching the module-level import."""
    import services.vector_store as vs_module

    mock_settings = MagicMock()
    mock_settings.faiss_index_dir = tmp_path
    monkeypatch.setattr(vs_module, "get_settings", lambda: mock_settings)
    return tmp_path


def test_save_creates_index_and_meta_files(patch_index_dir) -> None:
    vectors = _random_unit_vectors(5, 8)
    metadata = _make_metadata(5)
    save_document_index("test-doc", vectors, metadata)

    assert (patch_index_dir / "test-doc.index").exists()
    assert (patch_index_dir / "test-doc.meta.json").exists()


def test_meta_json_is_valid_and_aligned(patch_index_dir) -> None:
    n = 4
    vectors = _random_unit_vectors(n, 8)
    metadata = _make_metadata(n, "doc-x")
    save_document_index("doc-x", vectors, metadata)

    raw = json.loads((patch_index_dir / "doc-x.meta.json").read_text())
    assert len(raw) == n
    assert raw[0]["chunk_id"] == 0
    assert raw[-1]["chunk_id"] == n - 1


def test_load_roundtrip(patch_index_dir) -> None:
    vectors = _random_unit_vectors(6, 8)
    metadata = _make_metadata(6, "doc-rt")
    save_document_index("doc-rt", vectors, metadata)

    index, loaded_meta = load_document_index("doc-rt")
    assert index.ntotal == 6
    assert len(loaded_meta) == 6
    assert loaded_meta[2]["chunk_id"] == 2


def test_load_missing_document_raises(patch_index_dir) -> None:
    with pytest.raises(FileNotFoundError):
        load_document_index("does-not-exist")


def test_search_returns_top_k_results(patch_index_dir) -> None:
    dim = 8
    vectors = _random_unit_vectors(10, dim)
    metadata = _make_metadata(10)
    save_document_index("doc-search", vectors, metadata)

    index, _ = load_document_index("doc-search")
    query = _random_unit_vectors(1, dim, seed=99)
    scores, indices = search_index(index, query, top_k=3)

    assert len(scores) == 3
    assert len(indices) == 3
    # Scores from IndexFlatIP on unit vectors are cosine similarities: [-1, 1]
    assert all(-1.0 <= s <= 1.0 + 1e-5 for s in scores)
    # Results are sorted descending
    assert scores[0] >= scores[1] >= scores[2]



def test_save_creates_index_and_meta_files(patch_index_dir) -> None:
    vectors = _random_unit_vectors(5, 8)
    metadata = _make_metadata(5)
    save_document_index("test-doc", vectors, metadata)

    assert (patch_index_dir / "test-doc.index").exists()
    assert (patch_index_dir / "test-doc.meta.json").exists()


def test_meta_json_is_valid_and_aligned(patch_index_dir) -> None:
    n = 4
    vectors = _random_unit_vectors(n, 8)
    metadata = _make_metadata(n, "doc-x")
    save_document_index("doc-x", vectors, metadata)

    raw = json.loads((patch_index_dir / "doc-x.meta.json").read_text())
    assert len(raw) == n
    assert raw[0]["chunk_id"] == 0
    assert raw[-1]["chunk_id"] == n - 1


def test_load_roundtrip(patch_index_dir) -> None:
    vectors = _random_unit_vectors(6, 8)
    metadata = _make_metadata(6, "doc-rt")
    save_document_index("doc-rt", vectors, metadata)

    index, loaded_meta = load_document_index("doc-rt")
    assert index.ntotal == 6
    assert len(loaded_meta) == 6
    assert loaded_meta[2]["chunk_id"] == 2


def test_load_missing_document_raises(patch_index_dir) -> None:
    with pytest.raises(FileNotFoundError):
        load_document_index("does-not-exist")


def test_search_returns_top_k_results(patch_index_dir) -> None:
    dim = 8
    vectors = _random_unit_vectors(10, dim)
    metadata = _make_metadata(10)
    save_document_index("doc-search", vectors, metadata)

    index, _ = load_document_index("doc-search")
    query = _random_unit_vectors(1, dim, seed=99)
    scores, indices = search_index(index, query, top_k=3)

    assert len(scores) == 3
    assert len(indices) == 3
    # Scores from IndexFlatIP on unit vectors are cosine similarities: [-1, 1]
    assert all(-1.0 <= s <= 1.0 + 1e-5 for s in scores)
    # Results are sorted descending
    assert scores[0] >= scores[1] >= scores[2]
