from config import Settings


def test_default_retrieval_settings() -> None:
    settings = Settings()
    assert settings.top_k == 5
    assert settings.similarity_threshold == 0.45
    assert settings.embedding_batch_size == 64
    assert settings.max_upload_size_mb == 10
    assert settings.max_chat_history_messages == 8
