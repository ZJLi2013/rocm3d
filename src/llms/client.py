"""Lightweight, dependency-free LLM client supporting multiple providers.

Supported providers
-------------------
- ``openai``        : OpenAI API (api.openai.com, Bearer token)
- ``anthropic``     : Anthropic API (api.anthropic.com, x-api-key header)
- ``openai_compat`` : Any OpenAI-compatible endpoint (custom base_url, Bearer token)
                      Covers: vLLM, Ollama, Azure OpenAI, AMD LLM Gateway, etc.

Configuration via environment variables
-----------------------------------------
  LLM_PROVIDER        openai | anthropic | openai_compat  (default: openai_compat)
  LLM_API_KEY         API key / subscription key
  LLM_BASE_URL        Base URL  (required for openai_compat)
  LLM_MODEL           Model name
  LLM_TEMPERATURE     Sampling temperature  (default: 1.0)
  LLM_MAX_TOKENS      Max tokens to generate (default: 4096)
  LLM_TOP_P           Nucleus sampling      (default: 1.0)
  LLM_AUTH_HEADER     Custom auth header name (default: Authorization)
                      e.g. ``Ocp-Apim-Subscription-Key`` for AMD gateway

Quick-start examples (.env)
-----------------------------
  # OpenAI
  LLM_PROVIDER=openai
  LLM_API_KEY=sk-...

  # Anthropic
  LLM_PROVIDER=anthropic
  LLM_API_KEY=sk-ant-...
  LLM_MODEL=claude-3-5-haiku-20241022

  # AMD LLM Gateway
  LLM_PROVIDER=openai_compat
  LLM_BASE_URL=https://llm-api.amd.com/OpenAI
  LLM_API_KEY=<subscription-key>
  LLM_AUTH_HEADER=Ocp-Apim-Subscription-Key

  # Local vLLM / Ollama
  LLM_PROVIDER=openai_compat
  LLM_BASE_URL=http://localhost:8000
  LLM_API_KEY=none
  LLM_MODEL=llama3
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Optional


ProviderName = Literal["openai", "anthropic", "openai_compat"]

_PROVIDER_BASE_URLS: dict[str, str] = {
    "openai": "https://api.openai.com",
    "anthropic": "https://api.anthropic.com",
    "openai_compat": "",  # must be supplied by caller
}
_ANTHROPIC_API_VERSION = "2023-06-01"


@dataclass(frozen=True)
class LLMConfig:
    """Generic LLM client configuration.

    Parameters
    ----------
    provider:
        Which provider API to use. ``openai_compat`` works with any OpenAI-style
        endpoint (vLLM, Ollama, AMD gateway, Azure OpenAI, …).
    base_url:
        Base URL for the API. Inferred from ``provider`` when not set
        (required for ``openai_compat``).
    model:
        Model identifier passed in the request body.
    api_key:
        API key / subscription key. Loaded from ``LLM_API_KEY`` env var when empty.
    auth_header:
        HTTP header used for authentication.
        Defaults to ``Authorization`` (Bearer token format).
        Set to e.g. ``Ocp-Apim-Subscription-Key`` for AMD gateway.
    temperature / max_tokens / top_p:
        Sampling parameters; can be overridden per-call.
    """

    provider: ProviderName = "openai_compat"
    base_url: str = ""
    model: str = "gpt-4o-mini"
    api_key: str = ""
    auth_header: str = ""  # empty → "Authorization" (Bearer)
    temperature: float = 1.0
    max_tokens: int = 4096
    top_p: float = 1.0


class LLMError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# .env loader (no third-party deps)
# ---------------------------------------------------------------------------

_DOTENV_LOADED = False


def _load_env_file_if_present() -> None:
    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return
    _DOTENV_LOADED = True

    candidates: list[Path] = [Path.cwd() / ".env"]
    repo_env = Path(__file__).resolve().parents[2] / ".env"
    if repo_env != candidates[0]:
        candidates.append(repo_env)

    env_path = next((p for p in candidates if p.is_file()), None)
    if env_path is None:
        return

    for raw_line in env_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue
        os.environ[key] = value.strip()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_api_key(cfg: LLMConfig) -> str:
    key = cfg.api_key or os.environ.get("LLM_API_KEY", "")
    if not key:
        raise LLMError(
            "LLM API key not found. Provide via:\n"
            "  env var LLM_API_KEY, or\n"
            "  LLMConfig(api_key=...)"
        )
    return key


def _resolve_base_url(cfg: LLMConfig) -> str:
    url = (
        cfg.base_url
        or os.environ.get("LLM_BASE_URL", "")
        or _PROVIDER_BASE_URLS.get(cfg.provider, "")
    )
    if not url:
        raise LLMError(
            f"base_url is required for provider '{cfg.provider}'. "
            "Set via LLM_BASE_URL env var or LLMConfig(base_url=...)."
        )
    return url.rstrip("/")


def _build_auth_headers(cfg: LLMConfig, api_key: str) -> dict[str, str]:
    if cfg.provider == "anthropic":
        return {
            "x-api-key": api_key,
            "anthropic-version": _ANTHROPIC_API_VERSION,
        }
    header_name = cfg.auth_header or os.environ.get("LLM_AUTH_HEADER", "") or "Authorization"
    if header_name.lower() == "authorization":
        return {"Authorization": f"Bearer {api_key}"}
    # Custom header (e.g. Ocp-Apim-Subscription-Key) — value passed as-is
    return {header_name: api_key}


def _post_json(
    *,
    candidate_urls: list[str],
    body: dict[str, Any],
    headers: dict[str, str],
    response_extractor: Callable[[dict[str, Any]], str],
    timeout_sec: float,
) -> str:
    last_err: Exception | None = None
    last_body: str | None = None

    for url in candidate_urls:
        req = urllib.request.Request(
            url=url,
            method="POST",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json", **headers},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
            break
        except urllib.error.HTTPError as e:
            last_body = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else str(e)
            last_err = e
            if getattr(e, "code", None) != 404:
                raise LLMError(
                    f"HTTP {getattr(e, 'code', '?')} from {url}: {last_body}"
                ) from e
            continue
        except Exception as e:
            raise LLMError(f"Request error: {e}") from e
    else:
        raise LLMError(
            f"All URL candidates returned 404. Last response: {last_body}"
        ) from last_err

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise LLMError(f"Non-JSON response: {raw[:2000]}") from e

    try:
        return response_extractor(data)
    except Exception as e:
        raise LLMError(f"Unexpected response schema: {data}") from e


# ---------------------------------------------------------------------------
# LLMClient
# ---------------------------------------------------------------------------


class LLMClient:
    """Minimal, dependency-free LLM client.

    Usage::

        # From environment variables
        client = get_client()

        # Explicit config
        client = LLMClient(LLMConfig(provider="openai", api_key="sk-..."))

        text = client.chat(messages=[{"role": "user", "content": "Hello"}])
    """

    def __init__(self, cfg: LLMConfig) -> None:
        self._cfg = cfg

    @property
    def config(self) -> LLMConfig:
        return self._cfg

    def chat(
        self,
        *,
        messages: list[dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        timeout_sec: float = 120.0,
    ) -> str:
        """Send a chat request and return the assistant's response text."""
        if self._cfg.provider == "anthropic":
            return self._chat_anthropic(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout_sec=timeout_sec,
            )
        return self._chat_openai_compat(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            timeout_sec=timeout_sec,
        )

    # ------------------------------------------------------------------
    # Provider implementations
    # ------------------------------------------------------------------

    def _chat_openai_compat(
        self,
        *,
        messages: list[dict[str, str]],
        model: Optional[str],
        temperature: Optional[float],
        max_tokens: Optional[int],
        top_p: Optional[float],
        timeout_sec: float,
    ) -> str:
        api_key = _resolve_api_key(self._cfg)
        base_url = _resolve_base_url(self._cfg)
        auth_headers = _build_auth_headers(self._cfg, api_key)

        body: dict[str, Any] = {
            "model": model or self._cfg.model,
            "messages": messages,
            "temperature": self._cfg.temperature if temperature is None else temperature,
            "max_tokens": self._cfg.max_tokens if max_tokens is None else max_tokens,
            "top_p": self._cfg.top_p if top_p is None else top_p,
        }

        return _post_json(
            candidate_urls=[
                f"{base_url}/v1/chat/completions",
                f"{base_url}/chat/completions",
            ],
            body=body,
            headers=auth_headers,
            response_extractor=lambda d: d["choices"][0]["message"]["content"],
            timeout_sec=timeout_sec,
        )

    def _chat_anthropic(
        self,
        *,
        messages: list[dict[str, str]],
        model: Optional[str],
        temperature: Optional[float],
        max_tokens: Optional[int],
        timeout_sec: float,
    ) -> str:
        api_key = _resolve_api_key(self._cfg)
        base_url = _resolve_base_url(self._cfg)
        auth_headers = _build_auth_headers(self._cfg, api_key)

        # Anthropic separates system message from the messages array
        system = next((m["content"] for m in messages if m["role"] == "system"), "")
        user_messages = [m for m in messages if m["role"] != "system"]

        body: dict[str, Any] = {
            "model": model or self._cfg.model,
            "messages": user_messages,
            "max_tokens": self._cfg.max_tokens if max_tokens is None else max_tokens,
            "temperature": self._cfg.temperature if temperature is None else temperature,
        }
        if system:
            body["system"] = system

        return _post_json(
            candidate_urls=[f"{base_url}/v1/messages"],
            body=body,
            headers=auth_headers,
            response_extractor=lambda d: d["content"][0]["text"],
            timeout_sec=timeout_sec,
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_client(cfg: Optional[LLMConfig] = None) -> LLMClient:
    """Build an :class:`LLMClient` from an explicit config or environment variables.

    Environment variable precedence::

        LLM_PROVIDER        openai | anthropic | openai_compat  (default: openai_compat)
        LLM_API_KEY         API key
        LLM_BASE_URL        Base URL for openai_compat
        LLM_MODEL           Model name
        LLM_TEMPERATURE     float  (default 1.0)
        LLM_MAX_TOKENS      int    (default 4096)
        LLM_TOP_P           float  (default 1.0)
        LLM_AUTH_HEADER     Custom auth header (default: Authorization)
    """
    _load_env_file_if_present()

    if cfg is not None:
        return LLMClient(cfg)

    def _float(name: str, default: float) -> float:
        v = os.environ.get(name)
        return default if not v else float(v)

    def _int(name: str, default: int) -> int:
        v = os.environ.get(name)
        return default if not v else int(v)

    raw_provider = os.environ.get("LLM_PROVIDER", "openai_compat")
    if raw_provider not in ("openai", "anthropic", "openai_compat"):
        raise LLMError(
            f"LLM_PROVIDER must be one of: openai, anthropic, openai_compat. Got: {raw_provider!r}"
        )

    built_cfg = LLMConfig(
        provider=raw_provider,  # type: ignore[arg-type]
        base_url=os.environ.get("LLM_BASE_URL", ""),
        model=os.environ.get("LLM_MODEL", "gpt-4o-mini"),
        api_key="",  # resolved lazily from env in _resolve_api_key
        auth_header=os.environ.get("LLM_AUTH_HEADER", ""),
        temperature=_float("LLM_TEMPERATURE", 1.0),
        max_tokens=_int("LLM_MAX_TOKENS", 4096),
        top_p=_float("LLM_TOP_P", 1.0),
    )
    return LLMClient(built_cfg)


def get_default_client() -> LLMClient:
    """Alias for :func:`get_client` — kept for API compatibility."""
    return get_client()
