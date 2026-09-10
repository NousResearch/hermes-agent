"""Behavior tests for ChatGPT/Codex OAuth speech-to-text."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest


class _Response:
    def __init__(self, status_code=200, payload=None, text="", headers=None):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text
        self.headers = headers or {"content-type": "application/json"}

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests

            error = requests.HTTPError(f"{self.status_code} response")
            error.response = self
            raise error


def test_explicit_openai_codex_provider_uses_existing_oauth_login():
    from tools.transcription_tools import _get_provider

    resolver = Mock()
    with (
        patch(
            "hermes_cli.auth.has_codex_runtime_credentials",
            return_value=True,
        ),
        patch(
            "tools.transcription_tools._resolve_codex_stt_credentials",
            resolver,
        ),
    ):
        assert _get_provider({"provider": "openai-codex"}) == "openai-codex"

    resolver.assert_not_called()


def test_codex_oauth_is_never_selected_by_stt_autodetection():
    from tools.transcription_tools import _get_provider

    with (
        patch("tools.transcription_tools._detect_local_backend", return_value=None),
        patch("tools.transcription_tools._has_codex_stt_backend", return_value=True),
        patch("tools.transcription_tools._CLOUD_PROVIDER_SPECS", {}),
    ):
        assert _get_provider({"enabled": True}) == "none"


def test_codex_credentials_include_optional_chatgpt_account_id():
    from tools.transcription_tools import _resolve_codex_stt_credentials

    entry = SimpleNamespace(runtime_api_key="oauth-token", id="cred-1")
    pool = Mock()
    pool.select.return_value = entry
    with (
        patch("agent.credential_pool.load_pool", return_value=pool),
        patch(
            "hermes_cli.auth.codex_account_id_from_access_token",
            return_value="account-123",
        ) as account_id,
    ):
        credentials = _resolve_codex_stt_credentials()

    assert credentials["account_id"] == "account-123"
    assert credentials["credential_id"] == "cred-1"
    account_id.assert_called_once_with("oauth-token")


def test_openai_codex_transcription_uses_subscription_endpoint(tmp_path):
    from tools.transcription_tools import _transcribe_openai_codex

    audio = tmp_path / "voice.webm"
    audio.write_bytes(b"audio")
    response = _Response(payload={"text": "Merhaba Codex"})

    with (
        patch(
            "tools.transcription_tools._resolve_codex_stt_credentials",
            return_value={"api_key": "oauth-token", "account_id": "account-123"},
        ),
        patch("requests.post", return_value=response) as post,
    ):
        result = _transcribe_openai_codex(str(audio), language="tr")

    assert result == {
        "success": True,
        "transcript": "Merhaba Codex",
        "provider": "openai-codex",
    }
    kwargs = post.call_args.kwargs
    assert post.call_args.args[0] == "https://chatgpt.com/backend-api/transcribe"
    assert kwargs["headers"]["Authorization"] == "Bearer oauth-token"
    assert kwargs["headers"]["ChatGPT-Account-ID"] == "account-123"
    assert kwargs["headers"]["User-Agent"].startswith("codex_cli_rs/")
    assert kwargs["headers"]["originator"] == "codex_cli_rs"
    assert kwargs["data"] == {"language": "tr"}
    assert kwargs["allow_redirects"] is False
    filename, handle, mime = kwargs["files"]["file"]
    assert filename == "voice.webm"
    assert mime == "audio/webm"
    assert handle.closed


def test_openai_codex_transcription_refreshes_once_after_unauthorized(tmp_path):
    from tools.transcription_tools import _transcribe_openai_codex

    audio = tmp_path / "voice.ogg"
    audio.write_bytes(b"audio")
    resolver = Mock(return_value={"api_key": "old-token", "credential_id": "old"})
    retry = Mock(return_value={"api_key": "new-token", "credential_id": "new"})

    with (
        patch(
            "tools.transcription_tools._resolve_codex_stt_credentials", resolver
        ),
        patch("tools.transcription_tools._retry_codex_stt_credentials", retry),
        patch(
            "requests.post",
            side_effect=[
                _Response(status_code=401, text="expired"),
                _Response(payload={"text": "yenilendi"}),
            ],
        ) as post,
    ):
        assert _transcribe_openai_codex(str(audio)) == {
            "success": True,
            "transcript": "yenilendi",
            "provider": "openai-codex",
        }

    assert resolver.call_args_list[0].kwargs == {}
    retry.assert_called_once_with(
        {"api_key": "old-token", "credential_id": "old"}, 401
    )
    assert post.call_args_list[1].kwargs["headers"]["Authorization"] == "Bearer new-token"


def test_codex_retry_refreshes_matching_pool_credential():
    from tools.transcription_tools import _retry_codex_stt_credentials

    refreshed = SimpleNamespace(runtime_api_key="new-token", id="cred-1")
    pool = Mock()
    pool.try_refresh_matching.return_value = refreshed
    with (
        patch("agent.credential_pool.load_pool", return_value=pool),
        patch(
            "hermes_cli.auth.codex_account_id_from_access_token",
            return_value=None,
        ),
    ):
        credentials = _retry_codex_stt_credentials(
            {"api_key": "old-token", "credential_id": "cred-1"}, 401
        )

    assert credentials is not None
    assert credentials["api_key"] == "new-token"
    pool.try_refresh_matching.assert_called_once_with(
        api_key_hint="old-token", credential_id="cred-1"
    )
    pool.mark_exhausted_and_rotate.assert_not_called()


def test_codex_retry_rotates_pool_credential_after_rate_limit():
    from tools.transcription_tools import _retry_codex_stt_credentials

    next_entry = SimpleNamespace(runtime_api_key="next-token", id="cred-2")
    pool = Mock()
    pool.select_excluding.return_value = next_entry
    with (
        patch("agent.credential_pool.load_pool", return_value=pool),
        patch(
            "hermes_cli.auth.codex_account_id_from_access_token",
            return_value=None,
        ),
    ):
        credentials = _retry_codex_stt_credentials(
            {"api_key": "limited-token", "credential_id": "cred-1"}, 429
        )

    assert credentials is not None
    assert credentials["api_key"] == "next-token"
    pool.select_excluding.assert_called_once_with(
        credential_id="cred-1", api_key_hint="limited-token"
    )
    pool.mark_exhausted_and_rotate.assert_not_called()


def test_codex_rate_limit_uses_real_fill_first_pool_alternative():
    from agent.credential_pool import CredentialPool, PooledCredential
    from tools.transcription_tools import _retry_codex_stt_credentials

    pool = CredentialPool(
        "openai-codex",
        [
            PooledCredential(
                provider="openai-codex",
                id="a",
                label="first",
                auth_type="api_key",
                priority=0,
                source="manual:first",
                access_token="token-a",
            ),
            PooledCredential(
                provider="openai-codex",
                id="b",
                label="second",
                auth_type="api_key",
                priority=1,
                source="manual:second",
                access_token="token-b",
            ),
        ],
    )
    first = pool.select()
    assert first is not None
    assert first.id == "a"

    with (
        patch("agent.credential_pool.load_pool", return_value=pool),
        patch(
            "hermes_cli.auth.codex_account_id_from_access_token",
            return_value=None,
        ),
    ):
        retry = _retry_codex_stt_credentials(
            {"api_key": "token-a", "credential_id": "a"}, 429
        )

    assert retry is not None
    assert retry["credential_id"] == "b"
    assert retry["api_key"] == "token-b"
    assert [(entry.id, entry.last_status) for entry in pool.entries()] == [
        ("a", None),
        ("b", None),
    ]


def test_openai_codex_transcription_rotates_once_after_rate_limit(tmp_path):
    from tools.transcription_tools import _transcribe_openai_codex

    audio = tmp_path / "voice.ogg"
    audio.write_bytes(b"audio")
    retry = Mock(return_value={"api_key": "next-token", "credential_id": "next"})

    with (
        patch(
            "tools.transcription_tools._resolve_codex_stt_credentials",
            return_value={"api_key": "limited-token", "credential_id": "limited"},
        ),
        patch("tools.transcription_tools._retry_codex_stt_credentials", retry),
        patch(
            "requests.post",
            side_effect=[
                _Response(status_code=429, text="limited"),
                _Response(payload={"text": "diğer hesap"}),
            ],
        ) as post,
    ):
        result = _transcribe_openai_codex(str(audio))

    assert result["transcript"] == "diğer hesap"
    retry.assert_called_once()
    assert len(post.call_args_list) == 2


def test_openai_codex_second_unauthorized_is_terminal(tmp_path):
    from tools.transcription_tools import _transcribe_openai_codex

    audio = tmp_path / "voice.ogg"
    audio.write_bytes(b"audio")
    with (
        patch(
            "tools.transcription_tools._resolve_codex_stt_credentials",
            return_value={"api_key": "old-token", "credential_id": "old"},
        ),
        patch(
            "tools.transcription_tools._retry_codex_stt_credentials",
            return_value={"api_key": "new-token", "credential_id": "new"},
        ),
        patch(
            "tools.transcription_tools._mark_codex_stt_credentials_failed"
        ) as mark_failed,
        patch(
            "requests.post",
            side_effect=[_Response(status_code=401), _Response(status_code=401)],
        ) as post,
    ):
        result = _transcribe_openai_codex(str(audio))

    assert result["error"].endswith("HTTP 401.")
    assert len(post.call_args_list) == 2
    mark_failed.assert_called_once_with(
        {"api_key": "new-token", "credential_id": "new"}, 401
    )


def test_openai_codex_second_rate_limit_does_not_exhaust_shared_pool(tmp_path):
    from tools.transcription_tools import _transcribe_openai_codex

    audio = tmp_path / "voice.ogg"
    audio.write_bytes(b"audio")
    with (
        patch(
            "tools.transcription_tools._resolve_codex_stt_credentials",
            return_value={"api_key": "old-token", "credential_id": "old"},
        ),
        patch(
            "tools.transcription_tools._retry_codex_stt_credentials",
            return_value={"api_key": "new-token", "credential_id": "new"},
        ),
        patch(
            "tools.transcription_tools._mark_codex_stt_credentials_failed"
        ) as mark_failed,
        patch(
            "requests.post",
            side_effect=[_Response(status_code=429), _Response(status_code=429)],
        ) as post,
    ):
        result = _transcribe_openai_codex(str(audio))

    assert result["error"].endswith("HTTP 429.")
    assert len(post.call_args_list) == 2
    mark_failed.assert_not_called()


def test_openai_codex_missing_credentials_returns_safe_error(tmp_path):
    from tools.transcription_tools import _transcribe_openai_codex

    audio = tmp_path / "voice.ogg"
    audio.write_bytes(b"audio")
    with patch(
        "tools.transcription_tools._resolve_codex_stt_credentials",
        side_effect=ValueError("OpenAI Codex OAuth credentials are unavailable."),
    ):
        result = _transcribe_openai_codex(str(audio))

    assert result["error"] == "OpenAI Codex OAuth credentials are unavailable."


def test_openai_codex_refuses_redirects_without_replaying_audio(tmp_path):
    from tools.transcription_tools import _transcribe_openai_codex

    audio = tmp_path / "voice.ogg"
    audio.write_bytes(b"audio")
    response = _Response(
        status_code=307,
        payload={"text": "must not be accepted"},
        headers={"location": "https://attacker.invalid/upload"},
    )
    with (
        patch(
            "tools.transcription_tools._resolve_codex_stt_credentials",
            return_value={"api_key": "secret-token", "account_id": "account-1"},
        ),
        patch("requests.post", return_value=response) as post,
    ):
        result = _transcribe_openai_codex(str(audio))

    assert result["error"].endswith("refused an unexpected redirect.")
    assert post.call_args.kwargs["allow_redirects"] is False
    assert len(post.call_args_list) == 1


def test_openai_codex_cloudflare_challenge_has_specific_error(tmp_path):
    from tools.transcription_tools import _transcribe_openai_codex

    audio = tmp_path / "voice.ogg"
    audio.write_bytes(b"audio")
    response = _Response(
        status_code=403,
        text="secret response body",
        headers={"cf-mitigated": "challenge"},
    )
    with (
        patch(
            "tools.transcription_tools._resolve_codex_stt_credentials",
            return_value={"api_key": "secret-token"},
        ),
        patch("requests.post", return_value=response),
    ):
        result = _transcribe_openai_codex(str(audio))

    assert "blocked" in result["error"]
    assert "secret" not in result["error"]


def test_openai_codex_ordinary_forbidden_returns_safe_http_error(tmp_path):
    from tools.transcription_tools import _transcribe_openai_codex

    audio = tmp_path / "voice.ogg"
    audio.write_bytes(b"audio")
    with (
        patch(
            "tools.transcription_tools._resolve_codex_stt_credentials",
            return_value={"api_key": "secret-token"},
        ),
        patch(
            "requests.post",
            return_value=_Response(status_code=403, text="secret response body"),
        ),
    ):
        result = _transcribe_openai_codex(str(audio))

    assert result["error"].endswith("HTTP 403.")
    assert "secret" not in result["error"]


def test_openai_codex_timeout_returns_specific_error(tmp_path):
    import requests

    from tools.transcription_tools import _transcribe_openai_codex

    audio = tmp_path / "voice.ogg"
    audio.write_bytes(b"audio")
    with (
        patch(
            "tools.transcription_tools._resolve_codex_stt_credentials",
            return_value={"api_key": "secret-token"},
        ),
        patch("requests.post", side_effect=requests.Timeout),
    ):
        result = _transcribe_openai_codex(str(audio))

    assert result["error"].endswith("request timed out.")
    assert "secret-token" not in result["error"]


@pytest.mark.parametrize(
    ("payload", "error"),
    [
        (["not", "an", "object"], "invalid response"),
        ({}, "returned no text"),
        ({"text": ["not", "a", "string"]}, "returned no text"),
    ],
)
def test_openai_codex_rejects_malformed_success_payload(tmp_path, payload, error):
    from tools.transcription_tools import _transcribe_openai_codex

    audio = tmp_path / "voice.ogg"
    audio.write_bytes(b"audio")
    with (
        patch(
            "tools.transcription_tools._resolve_codex_stt_credentials",
            return_value={"api_key": "token"},
        ),
        patch("requests.post", return_value=_Response(payload=payload)),
    ):
        result = _transcribe_openai_codex(str(audio))

    assert error in result["error"]


def test_openai_codex_rejects_non_json_success(tmp_path):
    from tools.transcription_tools import _transcribe_openai_codex

    audio = tmp_path / "voice.ogg"
    audio.write_bytes(b"audio")
    response = _Response()
    response.json = Mock(side_effect=ValueError("not json"))
    with (
        patch(
            "tools.transcription_tools._resolve_codex_stt_credentials",
            return_value={"api_key": "token"},
        ),
        patch("requests.post", return_value=response),
    ):
        result = _transcribe_openai_codex(str(audio))

    assert result["error"].endswith("invalid JSON.")


def test_dispatch_routes_openai_codex_provider(tmp_path):
    from tools.transcription_tools import _transcribe_prepared_audio
    from hermes_cli.config import DEFAULT_CONFIG

    audio = tmp_path / "voice.webm"
    audio.write_bytes(b"audio")
    config = deepcopy(DEFAULT_CONFIG["stt"])
    config.update(provider="openai-codex", language="tr")
    config["openai_codex"]["timeout"] = 45

    with (
        patch("tools.transcription_tools._load_stt_config", return_value=config),
        patch("tools.transcription_tools._get_provider", return_value="openai-codex"),
        patch(
            "tools.transcription_tools._transcribe_openai_codex",
            return_value={
                "success": True,
                "transcript": "uçtan uca",
                "provider": "openai-codex",
            },
        ) as transcribe,
    ):
        assert _transcribe_prepared_audio(str(audio)) == {
            "success": True,
            "transcript": "uçtan uca",
            "provider": "openai-codex",
        }

    transcribe.assert_called_once_with(str(audio), language="tr", timeout=45)


def test_dispatch_prefers_nonempty_codex_language_override(tmp_path):
    from tools.transcription_tools import _transcribe_prepared_audio

    audio = tmp_path / "voice.webm"
    audio.write_bytes(b"audio")
    config = {
        "enabled": True,
        "provider": "openai-codex",
        "language": "tr",
        "openai_codex": {"language": "de", "timeout": 120},
    }
    with (
        patch("tools.transcription_tools._load_stt_config", return_value=config),
        patch("tools.transcription_tools._get_provider", return_value="openai-codex"),
        patch(
            "tools.transcription_tools._transcribe_openai_codex",
            return_value={
                "success": True,
                "transcript": "Deutsch",
                "provider": "openai-codex",
            },
        ) as transcribe,
    ):
        _transcribe_prepared_audio(str(audio))

    transcribe.assert_called_once_with(str(audio), language="de", timeout=120)


def test_dispatch_preserves_explicit_empty_codex_language_for_auto_detect(tmp_path):
    from tools.transcription_tools import _transcribe_prepared_audio

    audio = tmp_path / "voice.ogg"
    audio.write_bytes(b"audio")
    config = {
        "enabled": True,
        "provider": "openai-codex",
        "language": "en",
        "openai_codex": {"language": "", "timeout": 120},
    }

    with (
        patch("tools.transcription_tools._load_stt_config", return_value=config),
        patch("tools.transcription_tools._get_provider", return_value="openai-codex"),
        patch(
            "tools.transcription_tools._transcribe_openai_codex",
            return_value={
                "success": True,
                "transcript": "Türkçe",
                "provider": "openai-codex",
            },
        ) as transcribe,
    ):
        result = _transcribe_prepared_audio(str(audio))

    assert result["success"] is True
    transcribe.assert_called_once_with(str(audio), language="", timeout=120)
