import json
from pathlib import Path

import faiss
import numpy as np

from config import get_settings


def _index_paths(document_id: str) -> tuple[Path, Path]:
    settings = get_settings()
    base: Path = settings.faiss_index_dir
    base.mkdir(parents=True, exist_ok=True)
    return base / f"{document_id}.index", base / f"{document_id}.meta.json"


def save_document_index(
    document_id: str,
    vectors: np.ndarray,
    metadata: list[dict],
) -> None:
    """Persist a FAISS IndexFlatIP and its aligned metadata to disk.

    vectors must be L2-normalized float32 of shape (n, dim).
    metadata must be a list of n dicts aligned to the vector rows.
    """
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    index_path, meta_path = _index_paths(document_id)
    faiss.write_index(index, str(index_path))
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def load_document_index(document_id: str) -> tuple[faiss.IndexFlatIP, list[dict]]:
    """Load a persisted FAISS index and its metadata for a document."""
    index_path, meta_path = _index_paths(document_id)

    if not index_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"No index found for document '{document_id}'. Upload it first."
        )

    index = faiss.read_index(str(index_path))
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    return index, metadata


def search_index(
    index: faiss.IndexFlatIP,
    query_vector: np.ndarray,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Run a top-k inner-product search.

    query_vector: L2-normalized float32 of shape (1, dim).
    Returns (scores, indices) arrays of shape (top_k,).
    """
    scores, indices = index.search(query_vector, top_k)
    return scores[0], indices[0]

