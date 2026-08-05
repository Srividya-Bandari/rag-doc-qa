"""Integration tests for the ingestion pipeline.

These tests run the full PDF → chunk → embed → FAISS → persist pipeline
using a fixture PDF and a mocked OpenAI embeddings client.
No network calls are made.
"""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import openai
import pytest
from fastapi.testclient import TestClient

from main import app

FIXTURE_PDF = Path(__file__).parent.parent / "fixtures" / "sample.pdf"


def _mock_embeddings_client(batch_size_hint: int = 64, dim: int = 16):
    """Return a mock OpenAI client whose embeddings.create returns random unit vectors."""
    rng = np.random.default_rng(0)

    def fake_create(model, input):  # noqa: A002
        vecs = rng.random((len(input), dim)).astype(float)
        response = MagicMock()
        response.data = [MagicMock(embedding=v.tolist()) for v in vecs]
        return response

    mock_client = MagicMock()
    mock_client.embeddings.create.side_effect = fake_create
    return mock_client


@pytest.fixture()
def patch_openai_and_index_dir(tmp_path, monkeypatch):
    """Patch OpenAI and redirect FAISS writes to tmp_path."""
    import routers.upload as upload_module
    import services.embedder as embedder_module
    import services.vector_store as vs_module

    mock_settings = MagicMock()
    mock_settings.faiss_index_dir = tmp_path
    mock_settings.openai_api_key = "test-key"
    mock_settings.embedding_model = "text-embedding-3-small"
    mock_settings.embedding_batch_size = 64
    mock_settings.chunk_size_tokens = 512
    mock_settings.chunk_overlap_tokens = 50
    mock_settings.max_upload_size_mb = 10

    monkeypatch.setattr(vs_module, "get_settings", lambda: mock_settings)
    monkeypatch.setattr(upload_module, "get_settings", lambda: mock_settings)

    mock_client = _mock_embeddings_client()
    monkeypatch.setattr(embedder_module.openai, "OpenAI", lambda **kw: mock_client)
    monkeypatch.setattr(embedder_module, "get_settings", lambda: mock_settings)

    return tmp_path, mock_client


def test_upload_creates_index_and_meta(patch_openai_and_index_dir) -> None:
    tmp_path, _ = patch_openai_and_index_dir
    client = TestClient(app)

    with FIXTURE_PDF.open("rb") as f:
        response = client.post(
            "/api/upload",
            files={"file": ("sample.pdf", f, "application/pdf")},
        )

    assert response.status_code == 200
    body = response.json()
    doc_id = body["document_id"]
    assert doc_id  # non-empty UUID

    # Both persisted files must exist
    assert (tmp_path / f"{doc_id}.index").exists()
    assert (tmp_path / f"{doc_id}.meta.json").exists()


def test_upload_status_contains_chunk_count(patch_openai_and_index_dir) -> None:
    tmp_path, _ = patch_openai_and_index_dir
    client = TestClient(app)

    with FIXTURE_PDF.open("rb") as f:
        response = client.post(
            "/api/upload",
            files={"file": ("sample.pdf", f, "application/pdf")},
        )

    assert response.status_code == 200
    status = response.json()["status"]
    assert status.startswith("indexed:")
    chunk_count = int(status.split(":")[1].split("_")[0])
    assert chunk_count > 0


def test_upload_rejects_non_pdf(patch_openai_and_index_dir) -> None:
    client = TestClient(app)
    response = client.post(
        "/api/upload",
        files={"file": ("doc.txt", b"hello", "text/plain")},
    )
    assert response.status_code == 400


def test_upload_rejects_empty_file(patch_openai_and_index_dir) -> None:
    client = TestClient(app)
    response = client.post(
        "/api/upload",
        files={"file": ("empty.pdf", b"", "application/pdf")},
    )
    assert response.status_code == 400


def test_upload_rejects_invalid_pdf_bytes(patch_openai_and_index_dir) -> None:
    client = TestClient(app)
    response = client.post(
        "/api/upload",
        files={"file": ("broken.pdf", b"not-a-real-pdf", "application/pdf")},
    )
    assert response.status_code == 400
    assert "Invalid or unreadable PDF" in response.json()["detail"]


def test_upload_rejects_file_over_size_limit(tmp_path, monkeypatch) -> None:
    import routers.upload as upload_module
    import services.embedder as embedder_module
    import services.vector_store as vs_module

    mock_settings = MagicMock()
    mock_settings.faiss_index_dir = tmp_path
    mock_settings.openai_api_key = "test-key"
    mock_settings.embedding_model = "text-embedding-3-small"
    mock_settings.embedding_batch_size = 64
    mock_settings.chunk_size_tokens = 512
    mock_settings.chunk_overlap_tokens = 50
    mock_settings.max_upload_size_mb = 1

    monkeypatch.setattr(vs_module, "get_settings", lambda: mock_settings)
    monkeypatch.setattr(upload_module, "get_settings", lambda: mock_settings)
    monkeypatch.setattr(embedder_module, "get_settings", lambda: mock_settings)
    monkeypatch.setattr(embedder_module.openai, "OpenAI", lambda **kw: _mock_embeddings_client())

    client = TestClient(app)
    oversized_bytes = b"%PDF-1.4\n" + (b"x" * (1024 * 1024 + 1))
    response = client.post(
        "/api/upload",
        files={"file": ("large.pdf", oversized_bytes, "application/pdf")},
    )
    assert response.status_code == 413


def test_upload_maps_openai_failure_to_502(tmp_path, monkeypatch) -> None:
    import routers.upload as upload_module
    import services.embedder as embedder_module
    import services.vector_store as vs_module

    mock_settings = MagicMock()
    mock_settings.faiss_index_dir = tmp_path
    mock_settings.openai_api_key = "test-key"
    mock_settings.embedding_model = "text-embedding-3-small"
    mock_settings.embedding_batch_size = 64
    mock_settings.chunk_size_tokens = 512
    mock_settings.chunk_overlap_tokens = 50
    mock_settings.max_upload_size_mb = 10

    monkeypatch.setattr(vs_module, "get_settings", lambda: mock_settings)
    monkeypatch.setattr(upload_module, "get_settings", lambda: mock_settings)
    monkeypatch.setattr(embedder_module, "get_settings", lambda: mock_settings)

    client = TestClient(app)
    failing_client = MagicMock()
    failing_client.embeddings.create.side_effect = openai.OpenAIError("boom")
    monkeypatch.setattr(embedder_module.openai, "OpenAI", lambda **kw: failing_client)

    with FIXTURE_PDF.open("rb") as f:
        response = client.post(
            "/api/upload",
            files={"file": ("sample.pdf", f, "application/pdf")},
        )

    assert response.status_code == 502
    assert "OpenAI request failed" in response.json()["detail"]
