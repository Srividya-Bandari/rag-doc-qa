import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import get_settings


def _make_splitter() -> RecursiveCharacterTextSplitter:
    settings = get_settings()
    enc = tiktoken.get_encoding("cl100k_base")

    def _token_len(text: str) -> int:
        return len(enc.encode(text))

    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size_tokens,
        chunk_overlap=settings.chunk_overlap_tokens,
        length_function=_token_len,
        separators=["\n\n", "\n", " ", ""],
    )


def chunk_pages(pages: list[dict], document_id: str) -> list[dict]:
    """Split per-page text into token-accurate overlapping chunks.

    Input:  list of {"page_number": int, "text": str}
    Output: list of {"document_id": str, "chunk_id": int,
                     "page_number": int, "chunk_text": str}
    """
    splitter = _make_splitter()
    chunks: list[dict] = []
    chunk_id = 0

    for page in pages:
        page_chunks = splitter.split_text(page["text"])
        for chunk_text in page_chunks:
            chunks.append({
                "document_id": document_id,
                "chunk_id": chunk_id,
                "page_number": page["page_number"],
                "chunk_text": chunk_text,
            })
            chunk_id += 1

    return chunks

