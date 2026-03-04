"""Lightweight LLM client for rocm3d-autorun.

Public API
----------
::

    from llms import LLMClient, LLMConfig, LLMError, get_client

    # From env vars (LLM_PROVIDER / LLM_API_KEY / LLM_BASE_URL)
    client = get_client()

    # Explicit config
    client = LLMClient(LLMConfig(provider="openai", api_key="sk-..."))

    text = client.chat(messages=[{"role": "user", "content": "Hello"}])

See .env.example at the repo root for all supported configuration options.
"""

from .client import (
    LLMClient,
    LLMConfig,
    LLMError,
    get_client,
    get_default_client,
)

__all__ = [
    "LLMClient",
    "LLMConfig",
    "LLMError",
    "get_client",
    "get_default_client",
]
