"""Unit tests for chunker — no network calls needed."""

from services.chunker import chunk_pages


def _make_pages(texts: list[str]) -> list[dict]:
    return [{"page_number": i + 1, "text": t} for i, t in enumerate(texts)]


def test_chunk_ids_are_sequential() -> None:
    pages = _make_pages(["word " * 20, "word " * 20])
    chunks = chunk_pages(pages, "doc-1")
    ids = [c["chunk_id"] for c in chunks]
    assert ids == list(range(len(ids)))


def test_page_number_carried_into_chunks() -> None:
    pages = _make_pages(["alpha " * 10, "beta " * 10])
    chunks = chunk_pages(pages, "doc-1")
    page_nums = {c["page_number"] for c in chunks}
    assert 1 in page_nums
    assert 2 in page_nums


def test_document_id_carried_into_chunks() -> None:
    pages = _make_pages(["hello world " * 10])
    chunks = chunk_pages(pages, "test-doc-id")
    assert all(c["document_id"] == "test-doc-id" for c in chunks)


def test_empty_pages_returns_empty_chunks() -> None:
    assert chunk_pages([], "doc-1") == []


def test_chunk_text_is_non_empty() -> None:
    pages = _make_pages(["The quick brown fox " * 30])
    chunks = chunk_pages(pages, "doc-1")
    assert all(c["chunk_text"].strip() for c in chunks)
