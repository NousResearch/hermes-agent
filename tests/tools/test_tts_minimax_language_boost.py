"""MiniMax TTS ``language_boost`` passthrough onto the t2a_v2 payload.

``tts.minimax.language_boost`` is a pronunciation hint MiniMax accepts on the
t2a_v2 payload; without it the API picks its own default for the model. The
invariant here is the contract between config and wire payload: a configured
value reaches the request body, and an unset value leaves the body untouched.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from tools.tts_tool import _generate_minimax_tts

CREDENTIAL_SENTINEL = "FAKE_GLOBAL_CREDENTIAL"


@pytest.fixture(autouse=True)
def _fake_minimax_credentials(monkeypatch):
    """Mirrors tests/tools/test_tts_minimax_region.py: credentials resolve without a network call."""
    values = {"MINIMAX_API_KEY": CREDENTIAL_SENTINEL}
    monkeypatch.setattr(
        "hermes_cli.config.get_env_value",
        lambda name, default=None: values.get(name, default),
    )
    return values


@pytest.fixture
def posted_requests(monkeypatch):
    """Capture the payloads the provider would send, without touching the network."""
    requests: list[dict] = []

    def _fake_post_json(url, payload, headers):
        requests.append({"url": url, "payload": payload, "headers": headers})
        return SimpleNamespace(
            content=json.dumps(
                {"base_resp": {"status_code": 0}, "data": {"audio": "4f67" * 8}}
            ).encode("utf-8"),
            raise_for_status=lambda: None,
        )

    monkeypatch.setattr("tools.tts_tool_providers._post_json", _fake_post_json)
    return requests


def test_language_boost_reaches_t2a_v2_payload(posted_requests, tmp_path):
    config = {"minimax": {"language_boost": "Polish"}}

    _generate_minimax_tts("Dzien dobry.", str(tmp_path / "out.mp3"), config)

    assert len(posted_requests) == 1
    assert "t2a_v2" in posted_requests[0]["url"]
    assert posted_requests[0]["payload"]["language_boost"] == "Polish"


def test_absent_language_boost_leaves_payload_untouched(posted_requests, tmp_path):
    _generate_minimax_tts("Hello.", str(tmp_path / "out.mp3"), {"minimax": {}})

    assert len(posted_requests) == 1
    assert "language_boost" not in posted_requests[0]["payload"]
