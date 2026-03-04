"""Tests for src/llms/client.py.

Unit tests run without network access (mock HTTP).
The live integration test at the bottom requires LLM_API_KEY / LLM_GATEWAY_KEY in env.
"""

from __future__ import annotations

import json
import os
import sys
import unittest
import urllib.error
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llms.client import (  # noqa: E402
    LLMClient,
    LLMConfig,
    LLMError,
    get_client,
    get_default_client,
)
from llms import (  # noqa: E402
    AMDGatewayClient,
    AMDGatewayClientConfig,
    AMDGatewayClientError,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_openai_response(content: str) -> bytes:
    return json.dumps({
        "choices": [{"message": {"role": "assistant", "content": content}}]
    }).encode()


def _make_anthropic_response(content: str) -> bytes:
    return json.dumps({
        "content": [{"type": "text", "text": content}]
    }).encode()


def _mock_urlopen(response_bytes: bytes):
    """Return a context-manager mock that yields a readable fake HTTP response."""
    resp = MagicMock()
    resp.read.return_value = response_bytes
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


# ---------------------------------------------------------------------------
# Unit tests — no network
# ---------------------------------------------------------------------------

class TestLLMConfig(unittest.TestCase):
    def test_defaults(self) -> None:
        cfg = LLMConfig()
        self.assertEqual(cfg.provider, "openai_compat")
        self.assertEqual(cfg.model, "gpt-4o-mini")
        self.assertEqual(cfg.temperature, 1.0)
        self.assertEqual(cfg.max_tokens, 4096)

    def test_backward_compat_aliases(self) -> None:
        self.assertIs(AMDGatewayClient, LLMClient)
        self.assertIs(AMDGatewayClientConfig, LLMConfig)
        self.assertIs(AMDGatewayClientError, LLMError)


class TestLLMClientOpenAICompat(unittest.TestCase):
    def _client(self, **kwargs) -> LLMClient:
        return LLMClient(LLMConfig(
            provider="openai_compat",
            base_url="https://fake.example.com",
            api_key="test-key",
            **kwargs,
        ))

    @patch("urllib.request.urlopen")
    def test_chat_returns_content(self, mock_urlopen) -> None:
        mock_urlopen.return_value = _mock_urlopen(_make_openai_response("Hello!"))
        out = self._client().chat(messages=[{"role": "user", "content": "Hi"}])
        self.assertEqual(out, "Hello!")

    @patch("urllib.request.urlopen")
    def test_bearer_auth_header(self, mock_urlopen) -> None:
        mock_urlopen.return_value = _mock_urlopen(_make_openai_response("ok"))
        self._client().chat(messages=[{"role": "user", "content": "Hi"}])
        call_args = mock_urlopen.call_args
        req = call_args[0][0]
        self.assertIn("Authorization", req.headers)
        self.assertTrue(req.headers["Authorization"].startswith("Bearer "))

    @patch("urllib.request.urlopen")
    def test_custom_auth_header(self, mock_urlopen) -> None:
        mock_urlopen.return_value = _mock_urlopen(_make_openai_response("ok"))
        client = self._client(auth_header="Ocp-Apim-Subscription-Key")
        client.chat(messages=[{"role": "user", "content": "Hi"}])
        req = mock_urlopen.call_args[0][0]
        # Header names are title-cased by urllib
        headers_lower = {k.lower(): v for k, v in req.headers.items()}
        self.assertIn("ocp-apim-subscription-key", headers_lower)

    @patch("urllib.request.urlopen")
    def test_model_override(self, mock_urlopen) -> None:
        mock_urlopen.return_value = _mock_urlopen(_make_openai_response("ok"))
        self._client().chat(messages=[{"role": "user", "content": "Hi"}], model="gpt-4o")
        req = mock_urlopen.call_args[0][0]
        body = json.loads(req.data)
        self.assertEqual(body["model"], "gpt-4o")

    @patch("urllib.request.urlopen")
    def test_http_error_raises_llmerror(self, mock_urlopen) -> None:
        err = urllib.error.HTTPError(url="x", code=401, msg="Unauthorized", hdrs=None, fp=BytesIO(b"unauthorized"))
        mock_urlopen.side_effect = err
        with self.assertRaises(LLMError):
            self._client().chat(messages=[{"role": "user", "content": "Hi"}])

    @patch("urllib.request.urlopen")
    def test_non_json_response_raises_llmerror(self, mock_urlopen) -> None:
        resp = _mock_urlopen(b"not json at all")
        mock_urlopen.return_value = resp
        with self.assertRaises(LLMError):
            self._client().chat(messages=[{"role": "user", "content": "Hi"}])


class TestLLMClientAnthropic(unittest.TestCase):
    def _client(self) -> LLMClient:
        return LLMClient(LLMConfig(
            provider="anthropic",
            api_key="sk-ant-test",
            model="claude-3-5-haiku-20241022",
        ))

    @patch("urllib.request.urlopen")
    def test_chat_returns_content(self, mock_urlopen) -> None:
        mock_urlopen.return_value = _mock_urlopen(_make_anthropic_response("Howdy!"))
        out = self._client().chat(messages=[{"role": "user", "content": "Hello"}])
        self.assertEqual(out, "Howdy!")

    @patch("urllib.request.urlopen")
    def test_x_api_key_header(self, mock_urlopen) -> None:
        mock_urlopen.return_value = _mock_urlopen(_make_anthropic_response("ok"))
        self._client().chat(messages=[{"role": "user", "content": "Hi"}])
        req = mock_urlopen.call_args[0][0]
        headers_lower = {k.lower(): v for k, v in req.headers.items()}
        self.assertIn("x-api-key", headers_lower)
        self.assertNotIn("authorization", headers_lower)

    @patch("urllib.request.urlopen")
    def test_system_message_extracted(self, mock_urlopen) -> None:
        mock_urlopen.return_value = _mock_urlopen(_make_anthropic_response("ok"))
        self._client().chat(messages=[
            {"role": "system", "content": "Be helpful"},
            {"role": "user", "content": "Hi"},
        ])
        req = mock_urlopen.call_args[0][0]
        body = json.loads(req.data)
        self.assertEqual(body.get("system"), "Be helpful")
        # system message should not be in messages array
        for m in body["messages"]:
            self.assertNotEqual(m.get("role"), "system")

    @patch("urllib.request.urlopen")
    def test_url_uses_messages_endpoint(self, mock_urlopen) -> None:
        mock_urlopen.return_value = _mock_urlopen(_make_anthropic_response("ok"))
        self._client().chat(messages=[{"role": "user", "content": "Hi"}])
        req = mock_urlopen.call_args[0][0]
        self.assertIn("/v1/messages", req.full_url)


class TestGetClient(unittest.TestCase):
    def test_missing_api_key_raises(self) -> None:
        env_backup = {k: os.environ.pop(k) for k in
                      ["LLM_API_KEY", "LLM_GATEWAY_KEY", "LLM_PROVIDER"]
                      if k in os.environ}
        try:
            client = get_client(LLMConfig(
                provider="openai_compat",
                base_url="https://fake.example.com",
                api_key="",
            ))
            with self.assertRaises(LLMError):
                client.chat(messages=[{"role": "user", "content": "Hi"}])
        finally:
            os.environ.update(env_backup)

    def test_get_default_client_is_alias(self) -> None:
        # Both should produce an LLMClient without raising when env has a key
        with patch.dict(os.environ, {"LLM_GATEWAY_KEY": "fake", "LLM_PROVIDER": ""}):
            os.environ.pop("LLM_PROVIDER", None)
            client = get_default_client()
            self.assertIsInstance(client, LLMClient)

    def test_legacy_gateway_key_auto_configures(self) -> None:
        env = {
            "LLM_GATEWAY_KEY": "legacy-key",
            "LLM_GATEWAY_BASE_URL": "https://llm-api.amd.com",
        }
        with patch.dict(os.environ, env, clear=False):
            os.environ.pop("LLM_PROVIDER", None)
            os.environ.pop("LLM_API_KEY", None)
            client = get_client()
        self.assertIsInstance(client, LLMClient)
        self.assertEqual(client.config.auth_header, "Ocp-Apim-Subscription-Key")
        self.assertIn("/OpenAI", client.config.base_url)

    def test_invalid_provider_raises(self) -> None:
        with patch.dict(os.environ, {"LLM_PROVIDER": "unsupported", "LLM_API_KEY": "x"}):
            with self.assertRaises(LLMError):
                get_client()


# ---------------------------------------------------------------------------
# Live integration test (skipped when no API key is available)
# ---------------------------------------------------------------------------

class TestLiveConnection(unittest.TestCase):
    def setUp(self) -> None:
        # Load .env if present
        from llms.client import _load_env_file_if_present
        _load_env_file_if_present()
        has_key = bool(
            os.environ.get("LLM_API_KEY")
            or os.environ.get("LLM_GATEWAY_KEY")
        )
        if not has_key:
            self.skipTest("No LLM_API_KEY / LLM_GATEWAY_KEY in environment — skipping live test")

    def test_live_chat(self) -> None:
        client = get_client()
        out = client.chat(
            messages=[{"role": "user", "content": "Reply with exactly: OK"}],
            timeout_sec=30,
            max_tokens=16,
        )
        self.assertIsInstance(out, str)
        self.assertTrue(out.strip())


if __name__ == "__main__":
    unittest.main()
