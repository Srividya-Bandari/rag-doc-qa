"""Unit tests for qa_service — all OpenAI and FAISS calls are mocked."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from services.qa_service import (
    _apply_threshold,
    _build_llm_messages,
    _build_context,
    _deduplicate,
    _overlap_ratio,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _chunk(chunk_id: int, page: int, text: str) -> dict:
    return {
        "document_id": "doc-1",
        "chunk_id": chunk_id,
        "page_number": page,
        "chunk_text": text,
    }


def _scores_indices(pairs: list[tuple[float, int]]):
    scores = np.array([s for s, _ in pairs], dtype=np.float32)
    indices = np.array([i for _, i in pairs], dtype=np.int64)
    return scores, indices


# ── Threshold filter ───────────────────────────────────────────────────────────

def test_threshold_keeps_results_above_cutoff() -> None:
    metadata = [_chunk(0, 1, "alpha"), _chunk(1, 2, "beta"), _chunk(2, 3, "gamma")]
    scores, indices = _scores_indices([(0.9, 0), (0.7, 1), (0.5, 2)])
    result = _apply_threshold(scores, indices, metadata, threshold=0.75)
    assert len(result) == 1
    assert result[0][1]["chunk_id"] == 0


def test_threshold_keeps_all_above_cutoff() -> None:
    metadata = [_chunk(i, i + 1, f"text {i}") for i in range(3)]
    scores, indices = _scores_indices([(0.9, 0), (0.85, 1), (0.8, 2)])
    result = _apply_threshold(scores, indices, metadata, threshold=0.75)
    assert len(result) == 3


def test_threshold_returns_empty_when_none_qualify() -> None:
    metadata = [_chunk(0, 1, "alpha")]
    scores, indices = _scores_indices([(0.4, 0)])
    result = _apply_threshold(scores, indices, metadata, threshold=0.75)
    assert result == []


def test_threshold_skips_faiss_sentinel_index() -> None:
    """FAISS returns -1 for unfilled slots — these must be skipped."""
    metadata = [_chunk(0, 1, "alpha")]
    scores = np.array([0.9, 0.0], dtype=np.float32)
    indices = np.array([0, -1], dtype=np.int64)
    result = _apply_threshold(scores, indices, metadata, threshold=0.5)
    assert len(result) == 1


# ── Overlap ratio ──────────────────────────────────────────────────────────────

def test_overlap_ratio_identical_text() -> None:
    assert _overlap_ratio("the quick brown fox", "the quick brown fox") == 1.0


def test_overlap_ratio_no_overlap() -> None:
    assert _overlap_ratio("apple banana", "cherry mango") == 0.0


def test_overlap_ratio_partial_overlap() -> None:
    ratio = _overlap_ratio("the quick brown fox", "the slow green fox")
    assert 0.0 < ratio < 1.0


def test_overlap_ratio_empty_strings() -> None:
    assert _overlap_ratio("", "something") == 0.0


# ── Deduplication ──────────────────────────────────────────────────────────────

def test_deduplicate_removes_duplicate_chunk_ids() -> None:
    candidates = [
        (0.9, _chunk(5, 1, "unique text about topic A")),
        (0.85, _chunk(5, 1, "unique text about topic A")),  # same chunk_id
    ]
    result = _deduplicate(candidates)
    assert len(result) == 1
    assert result[0][0] == pytest.approx(0.9)  # highest score kept


def test_deduplicate_removes_high_overlap_chunks() -> None:
    base_text = "the quick brown fox jumps over the lazy dog near the river"
    near_duplicate = "the quick brown fox jumps over the lazy dog near the stream"
    candidates = [
        (0.9, _chunk(1, 1, base_text)),
        (0.85, _chunk(2, 2, near_duplicate)),
    ]
    result = _deduplicate(candidates)
    assert len(result) == 1
    assert result[0][1]["chunk_id"] == 1


def test_deduplicate_keeps_distinct_chunks() -> None:
    candidates = [
        (0.9, _chunk(1, 1, "machine learning is a subset of artificial intelligence")),
        (0.8, _chunk(2, 2, "neural networks are inspired by the human brain")),
        (0.75, _chunk(3, 3, "transformers revolutionized natural language processing")),
    ]
    result = _deduplicate(candidates)
    assert len(result) == 3


def test_deduplicate_preserves_relevance_order() -> None:
    candidates = [
        (0.9, _chunk(1, 3, "first result text content alpha")),
        (0.8, _chunk(2, 1, "second result text content beta")),
        (0.75, _chunk(3, 2, "third result text content gamma")),
    ]
    result = _deduplicate(candidates)
    returned_scores = [s for s, _ in result]
    assert returned_scores == sorted(returned_scores, reverse=True)


# ── Context construction ───────────────────────────────────────────────────────

def test_build_context_labels_are_sequential() -> None:
    candidates = [
        (0.9, _chunk(1, 2, "first chunk text")),
        (0.8, _chunk(2, 5, "second chunk text")),
    ]
    context, sources = _build_context(candidates)
    assert sources[0].source_id == "S1"
    assert sources[1].source_id == "S2"


def test_build_context_preserves_relevance_order_not_page_order() -> None:
    """S1 must be the highest-scoring chunk regardless of its page number."""
    candidates = [
        (0.9, _chunk(3, 10, "high score chunk on late page")),
        (0.8, _chunk(1, 1,  "lower score chunk on first page")),
    ]
    _, sources = _build_context(candidates)
    assert sources[0].page_number == 10   # highest score → S1
    assert sources[1].page_number == 1    # lower score → S2


def test_build_context_format_contains_label_and_page() -> None:
    candidates = [(0.9, _chunk(1, 4, "some important content"))]
    context, _ = _build_context(candidates)
    assert "[S1 | Page 4]" in context
    assert "some important content" in context


def test_build_context_sources_carry_correct_metadata() -> None:
    candidates = [
        (0.88, _chunk(7, 3, "relevant paragraph")),
    ]
    _, sources = _build_context(candidates)
    assert sources[0].chunk_id == 7
    assert sources[0].page_number == 3
    assert sources[0].chunk_text == "relevant paragraph"


def test_build_llm_messages_includes_recent_chat_history() -> None:
    history = [
        MagicMock(role="user", content="Older question"),
        MagicMock(role="assistant", content="Older answer"),
        MagicMock(role="user", content="Follow-up question"),
    ]

    with patch("services.qa_service.get_settings") as mock_settings:
        mock_settings.return_value = MagicMock(max_chat_history_messages=2)
        messages = _build_llm_messages("ctx", "Current question", history)

    assert messages[0]["role"] == "system"
    assert messages[1]["content"] == "Older answer"
    assert messages[2]["content"] == "Follow-up question"
    assert messages[-1]["content"].startswith("Context:\nctx")


def test_build_llm_messages_uses_current_question_in_final_user_turn() -> None:
    messages = _build_llm_messages("context block", "What is the answer?", [])
    assert messages[-1]["role"] == "user"
    assert "Current question: What is the answer?" in messages[-1]["content"]
