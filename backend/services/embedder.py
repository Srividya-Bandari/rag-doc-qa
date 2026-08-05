import numpy as np
import openai

from config import get_settings


def _l2_normalize(vectors: list[list[float]]) -> np.ndarray:
    arr = np.array(vectors, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return arr / norms


def embed_chunks(chunks: list[dict]) -> np.ndarray:
    """Embed chunk texts in batches and return L2-normalized vectors.

    Input:  list of chunk dicts (must contain "chunk_text")
    Output: float32 numpy array of shape (len(chunks), embedding_dim),
            aligned to the input chunk order.
    """
    settings = get_settings()
    client = openai.OpenAI(api_key=settings.openai_api_key)
    batch_size = settings.embedding_batch_size
    texts = [c["chunk_text"] for c in chunks]

    raw_vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        response = client.embeddings.create(
            model=settings.embedding_model,
            input=batch,
        )
        # Responses are returned in the same order as input
        raw_vectors.extend([item.embedding for item in response.data])

    return _l2_normalize(raw_vectors)

