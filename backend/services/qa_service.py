import json
from collections.abc import Iterator

import numpy as np
import openai

from config import get_settings
from models.schemas import AskResponse, ChatMessage, Source
from services.vector_store import load_document_index, search_index


# ── Query embedding ────────────────────────────────────────────────────────────

def _embed_query(question: str) -> np.ndarray:
    """Return an L2-normalized float32 query vector of shape (1, dim)."""
    settings = get_settings()
    client = openai.OpenAI(api_key=settings.openai_api_key)
    response = client.embeddings.create(
        model=settings.embedding_model,
        input=[question],
    )
    vec = np.array(response.data[0].embedding, dtype=np.float32)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec.reshape(1, -1)


def _get_openai_client() -> openai.OpenAI:
    settings = get_settings()
    return openai.OpenAI(api_key=settings.openai_api_key)


# ── Post-retrieval filters ─────────────────────────────────────────────────────

def _apply_threshold(
    scores: np.ndarray,
    indices: np.ndarray,
    metadata: list[dict],
    threshold: float,
) -> list[tuple[float, dict]]:
    """Keep only results whose cosine similarity meets the threshold."""
    return [
        (float(scores[i]), metadata[indices[i]])
        for i in range(len(scores))
        if indices[i] != -1 and float(scores[i]) >= threshold
    ]


def _overlap_ratio(a: str, b: str) -> float:
    """Rough word-level overlap ratio between two strings."""
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    if not words_a or not words_b:
        return 0.0
    return len(words_a & words_b) / min(len(words_a), len(words_b))


def _deduplicate(candidates: list[tuple[float, dict]]) -> list[tuple[float, dict]]:
    """Remove duplicate chunk_ids and chunks with >80% text overlap.

    Input is already sorted by score descending (FAISS order).
    First occurrence wins — highest-scoring chunk is kept.
    """
    seen_chunk_ids: set[int] = set()
    selected: list[tuple[float, dict]] = []

    for score, chunk in candidates:
        if chunk["chunk_id"] in seen_chunk_ids:
            continue
        if any(_overlap_ratio(chunk["chunk_text"], kept["chunk_text"]) > 0.8
               for _, kept in selected):
            continue
        seen_chunk_ids.add(chunk["chunk_id"])
        selected.append((score, chunk))

    return selected


# ── Context + prompt construction ─────────────────────────────────────────────

def _build_context(candidates: list[tuple[float, dict]]) -> tuple[str, list[Source]]:
    """Format context in FAISS relevance order and assign S1..Sn labels."""
    sources: list[Source] = []
    context_parts: list[str] = []

    for rank, (_, chunk) in enumerate(candidates, start=1):
        label = f"S{rank}"
        sources.append(Source(
            source_id=label,
            chunk_id=chunk["chunk_id"],
            page_number=chunk["page_number"],
            chunk_text=chunk["chunk_text"],
        ))
        context_parts.append(f"[{label} | Page {chunk['page_number']}]\n{chunk['chunk_text']}")

    return "\n\n".join(context_parts), sources


def _trim_chat_history(chat_history: list[ChatMessage]) -> list[ChatMessage]:
    settings = get_settings()
    if settings.max_chat_history_messages <= 0:
        return []
    return chat_history[-settings.max_chat_history_messages :]


_SYSTEM_PROMPT = (
    "You are a precise document assistant. "
    "Answer the user's question using ONLY the context passages provided below. "
    "Each passage is labelled with a source tag like [S1 | Page N]. "
    "Cite the relevant source(s) inline in your answer using the exact format (S1 | page N). "
    "If the answer cannot be found in the provided context, respond with: "
    "'The answer is not available in the provided document.'"
)


def _build_llm_messages(
    context: str,
    question: str,
    chat_history: list[ChatMessage] | None = None,
) -> list[dict[str, str]]:
    history = _trim_chat_history(chat_history or [])
    messages: list[dict[str, str]] = [{"role": "system", "content": _SYSTEM_PROMPT}]
    messages.extend({"role": turn.role, "content": turn.content} for turn in history)
    messages.append(
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nCurrent question: {question}",
        }
    )
    return messages


def _call_llm(
    context: str,
    question: str,
    chat_history: list[ChatMessage] | None = None,
) -> str:
    settings = get_settings()
    client = _get_openai_client()
    response = client.chat.completions.create(
        model=settings.chat_model,
        messages=_build_llm_messages(context, question, chat_history),
        temperature=0,
    )
    return response.choices[0].message.content.strip()


def _stream_llm(
    context: str,
    question: str,
    chat_history: list[ChatMessage] | None = None,
) -> Iterator[str]:
    settings = get_settings()
    client = _get_openai_client()
    stream = client.chat.completions.create(
        model=settings.chat_model,
        messages=_build_llm_messages(context, question, chat_history),
        temperature=0,
        stream=True,
    )

    for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        if delta:
            yield delta


def _retrieve_context(document_id: str, question: str) -> tuple[str, list[Source]]:
    settings = get_settings()
    index, metadata = load_document_index(document_id)
    query_vector = _embed_query(question)
    scores, indices = search_index(index, query_vector, top_k=settings.top_k)
    candidates = _apply_threshold(scores, indices, metadata, settings.similarity_threshold)
    if not candidates:
        return "", []
    candidates = _deduplicate(candidates)
    if not candidates:
        return "", []
    return _build_context(candidates)


def _sse(event: str, payload: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


# ── Public entry point ─────────────────────────────────────────────────────────

def answer_question(
    document_id: str,
    question: str,
    chat_history: list[ChatMessage] | None = None,
) -> AskResponse:
    """Full RAG query pipeline.

    1. Embed + normalize the question.
    2. FAISS top-k search.
    3. Similarity threshold filter.
    4. Deduplication and overlap removal (relevance order preserved).
    5. Context construction with [S# | Page N] labels.
    6. OpenAI ChatCompletion with document-grounded system prompt.
    7. Return answer + structured sources.
    """
    context, sources = _retrieve_context(document_id, question)
    if not sources:
        return AskResponse(
            answer="The answer is not available in the provided document.",
            sources=[],
        )
    answer = _call_llm(context, question, chat_history)

    return AskResponse(answer=answer, sources=sources)


def stream_answer_question(
    document_id: str,
    question: str,
    chat_history: list[ChatMessage] | None = None,
) -> Iterator[str]:
    context, sources = _retrieve_context(document_id, question)

    def event_stream() -> Iterator[str]:
        if not sources:
            fallback = "The answer is not available in the provided document."
            yield _sse("done", {"answer": fallback, "sources": []})
            return

        answer_parts: list[str] = []
        try:
            for delta in _stream_llm(context, question, chat_history):
                answer_parts.append(delta)
                yield _sse("chunk", {"delta": delta})
        except openai.OpenAIError as exc:
            yield _sse("error", {"detail": f"OpenAI request failed: {exc}"})
            return

        answer = "".join(answer_parts).strip() or "The answer is not available in the provided document."
        yield _sse(
            "done",
            {
                "answer": answer,
                "sources": [source.model_dump() for source in sources],
            },
        )

    return event_stream()
