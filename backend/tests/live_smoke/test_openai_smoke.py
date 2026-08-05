import os

import pytest


@pytest.mark.live_openai
@pytest.mark.skipif(
    os.getenv("RUN_LIVE_OPENAI_TESTS", "false").lower() != "true",
    reason="Set RUN_LIVE_OPENAI_TESTS=true to run live OpenAI smoke tests",
)
def test_live_openai_smoke_placeholder() -> None:
    assert True

