"""Integration test for the full RAG pipeline: upload PDF → ask question.

Both OpenAI calls (embeddings + chat) are mocked. FAISS writes go to tmp_path.
No network calls are made.
"""

import re
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import openai
import pytest
from fastapi.testclient import TestClient

from main import app

FIXTURE_PDF = Path(__file__).parent.parent / "fixtures" / "sample.pdf"
DIM = 16


# ── Mock helpers ───────────────────────────────────────────────────────────────

def _mock_embedding_response(texts: list[str]) -> MagicMock:
    rng = np.random.default_rng(42)
    response = MagicMock()
    response.data = [
        MagicMock(embedding=rng.random(DIM).tolist()) for _ in texts
    ]
    return response


def _mock_chat_response(answer: str = "Mocked answer (S1 | page 1).") -> MagicMock:
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = answer
    return response


def _mock_stream_response(answer: str = "Mocked answer (S1 | page 1)."):
    for token in ["Mocked ", "answer ", "(S1 | page 1)."]:
        chunk = MagicMock()
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta.content = token
        yield chunk


# ── Shared fixture ─────────────────────────────────────────────────────────────

@pytest.fixture()
def full_pipeline(tmp_path, monkeypatch):
    """Patch OpenAI (embeddings + chat) and FAISS index dir."""
    import services.embedder as embedder_module
    import services.qa_service as qa_module
    import services.vector_store as vs_module

    mock_settings = MagicMock()
    mock_settings.faiss_index_dir = tmp_path
    mock_settings.openai_api_key = "test-key"
    mock_settings.embedding_model = "text-embedding-3-small"
    mock_settings.chat_model = "gpt-4o-mini"
    mock_settings.embedding_batch_size = 64
    mock_settings.chunk_size_tokens = 512
    mock_settings.chunk_overlap_tokens = 50
    mock_settings.top_k = 5
    mock_settings.similarity_threshold = 0.0  # accept all results in tests
    mock_settings.max_chat_history_messages = 8

    monkeypatch.setattr(vs_module, "get_settings", lambda: mock_settings)
    monkeypatch.setattr(embedder_module, "get_settings", lambda: mock_settings)
    monkeypatch.setattr(qa_module, "get_settings", lambda: mock_settings)

    # Single mock OpenAI client shared across both embedder and qa_service
    mock_openai_client = MagicMock()
    mock_openai_client.embeddings.create.side_effect = (
        lambda model, input: _mock_embedding_response(input)
    )
    def fake_chat_create(**kwargs):
        if kwargs.get("stream"):
            return _mock_stream_response()
        return _mock_chat_response()

    mock_openai_client.chat.completions.create.side_effect = fake_chat_create

    monkeypatch.setattr(embedder_module.openai, "OpenAI", lambda **kw: mock_openai_client)
    monkeypatch.setattr(qa_module.openai, "OpenAI", lambda **kw: mock_openai_client)

    return tmp_path, mock_openai_client


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_ask_returns_answer_and_sources(full_pipeline) -> None:
    client = TestClient(app)

    with FIXTURE_PDF.open("rb") as f:
        upload = client.post("/api/upload", files={"file": ("sample.pdf", f, "application/pdf")})
    assert upload.status_code == 200
    doc_id = upload.json()["document_id"]

    ask = client.post("/api/ask", json={"document_id": doc_id, "question": "What is AI?"})
    assert ask.status_code == 200
    body = ask.json()
    assert "answer" in body
    assert isinstance(body["sources"], list)


def test_ask_sources_have_correct_shape(full_pipeline) -> None:
    client = TestClient(app)

    with FIXTURE_PDF.open("rb") as f:
        upload = client.post("/api/upload", files={"file": ("sample.pdf", f, "application/pdf")})
    doc_id = upload.json()["document_id"]

    ask = client.post("/api/ask", json={"document_id": doc_id, "question": "What is ML?"})
    assert ask.status_code == 200

    for source in ask.json()["sources"]:
        assert re.match(r"^S\d+$", source["source_id"]), f"Bad source_id: {source['source_id']}"
        assert isinstance(source["chunk_id"], int)
        assert isinstance(source["page_number"], int)
        assert isinstance(source["chunk_text"], str)
        assert source["chunk_text"].strip()


def test_ask_unknown_document_returns_404(full_pipeline) -> None:
    client = TestClient(app)
    ask = client.post("/api/ask", json={"document_id": "no-such-doc", "question": "anything"})
    assert ask.status_code == 404


def test_ask_stream_returns_sse_events(full_pipeline) -> None:
    client = TestClient(app)

    with FIXTURE_PDF.open("rb") as f:
        upload = client.post("/api/upload", files={"file": ("sample.pdf", f, "application/pdf")})
    doc_id = upload.json()["document_id"]

    with client.stream(
        "POST",
        "/api/ask/stream",
        json={"document_id": doc_id, "question": "What is AI?"},
    ) as response:
        body = "".join(response.iter_text())

    assert response.status_code == 200
    assert "event: chunk" in body
    assert "event: done" in body
    assert '"source_id": "S1"' in body


def test_ask_passes_chat_history_to_llm(full_pipeline, monkeypatch) -> None:
    client = TestClient(app)

    with FIXTURE_PDF.open("rb") as f:
        upload = client.post("/api/upload", files={"file": ("sample.pdf", f, "application/pdf")})
    doc_id = upload.json()["document_id"]

    captured_messages = {}

    def fake_chat_completion(**kwargs):
        captured_messages["messages"] = kwargs["messages"]
        return _mock_chat_response("History-aware answer (S1 | page 1).")

    import services.qa_service as qa_module

    mock_client = MagicMock()
    mock_client.embeddings.create.side_effect = lambda model, input: _mock_embedding_response(input)
    mock_client.chat.completions.create.side_effect = fake_chat_completion
    monkeypatch.setattr(qa_module.openai, "OpenAI", lambda **kw: mock_client)

    ask = client.post(
        "/api/ask",
        json={
            "document_id": doc_id,
            "question": "What about the follow-up?",
            "chat_history": [
                {"role": "user", "content": "What is AI?"},
                {"role": "assistant", "content": "AI is ... (S1 | page 1)."},
            ],
        },
    )

    assert ask.status_code == 200
    assert captured_messages["messages"][1]["content"] == "What is AI?"
    assert captured_messages["messages"][2]["content"] == "AI is ... (S1 | page 1)."


def test_ask_below_threshold_returns_not_found_answer(tmp_path, monkeypatch) -> None:
    """When threshold is very high, no chunks qualify → fallback answer returned.

    We stub _embed_query to return a fixed zero-direction vector that is
    guaranteed to score below 0.9999 against any randomly generated chunk vector.
    """
    import services.embedder as embedder_module
    import services.qa_service as qa_module
    import services.vector_store as vs_module

    mock_settings = MagicMock()
    mock_settings.faiss_index_dir = tmp_path
    mock_settings.openai_api_key = "test-key"
    mock_settings.embedding_model = "text-embedding-3-small"
    mock_settings.chat_model = "gpt-4o-mini"
    mock_settings.embedding_batch_size = 64
    mock_settings.chunk_size_tokens = 512
    mock_settings.chunk_overlap_tokens = 50
    mock_settings.top_k = 5
    mock_settings.similarity_threshold = 0.9999  # impossibly high — nothing will pass
    mock_settings.max_chat_history_messages = 8

    monkeypatch.setattr(vs_module, "get_settings", lambda: mock_settings)
    monkeypatch.setattr(embedder_module, "get_settings", lambda: mock_settings)
    monkeypatch.setattr(qa_module, "get_settings", lambda: mock_settings)

    mock_openai_client = MagicMock()
    mock_openai_client.embeddings.create.side_effect = (
        lambda model, input: _mock_embedding_response(input)
    )
    mock_openai_client.chat.completions.create.return_value = _mock_chat_response()

    monkeypatch.setattr(embedder_module.openai, "OpenAI", lambda **kw: mock_openai_client)
    monkeypatch.setattr(qa_module.openai, "OpenAI", lambda **kw: mock_openai_client)

    # Return a unit vector guaranteed to be orthogonal to all stored chunk vectors
    # (chunk vectors are random in DIM=16 dims; e0 = [1, 0, ...] is almost always low-similarity)
    orthogonal_query = np.zeros((1, DIM), dtype=np.float32)
    orthogonal_query[0, DIM - 1] = 1.0  # last-axis unit vector, unlikely to match random chunks
    monkeypatch.setattr(qa_module, "_embed_query", lambda q: orthogonal_query)

    client = TestClient(app)
    with FIXTURE_PDF.open("rb") as f:
        upload = client.post("/api/upload", files={"file": ("sample.pdf", f, "application/pdf")})
    doc_id = upload.json()["document_id"]

    ask = client.post("/api/ask", json={"document_id": doc_id, "question": "anything"})
    assert ask.status_code == 200
    body = ask.json()
    assert "not available" in body["answer"].lower()
    assert body["sources"] == []


def test_ask_maps_openai_failure_to_502(full_pipeline, monkeypatch) -> None:
    client = TestClient(app)

    with FIXTURE_PDF.open("rb") as f:
        upload = client.post("/api/upload", files={"file": ("sample.pdf", f, "application/pdf")})
    doc_id = upload.json()["document_id"]

    import services.qa_service as qa_module

    failing_client = MagicMock()
    failing_client.embeddings.create.side_effect = openai.OpenAIError("ask failed")
    monkeypatch.setattr(qa_module.openai, "OpenAI", lambda **kw: failing_client)

    ask = client.post("/api/ask", json={"document_id": doc_id, "question": "What is AI?"})
    assert ask.status_code == 502
    assert "OpenAI request failed" in ask.json()["detail"]
