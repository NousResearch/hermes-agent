"""One browser identity owns requirements and every proof attempt."""

import base64
import json
import time

import pytest

from tools import codex_web_audio as audio


def _decode(token):
    return json.loads(base64.b64decode(token[7:].removesuffix("~S")))


@pytest.mark.parametrize("proof_required", [False, True])
def test_prepare_and_every_proof_keep_the_same_fingerprint(monkeypatch, proof_required):
    requests = []
    attempts = []
    fingerprints = []
    build = audio._fingerprint_array

    def fingerprint(*args, **kwargs):
        config = build(*args, **kwargs)
        fingerprints.append(config.copy())
        return config

    def request(_session, _method, path, **kwargs):
        requests.append(kwargs["json_body"])
        if path == audio._SENTINEL_PREPARE:
            body = {"prepare_token": "prepare", "proofofwork": {
                "required": proof_required, "seed": "seed", "difficulty": "0",
            }}
        else:
            body = {"token": "requirements"}
        return None, json.dumps(body).encode()

    def proof_hash(value):
        attempts.append(json.loads(base64.b64decode(value.removeprefix("seed"))))
        return "f0000000" if len(attempts) < 3 else "00000000"

    monkeypatch.setattr(audio, "_fingerprint_array", fingerprint)
    monkeypatch.setattr(audio, "_request", request)
    monkeypatch.setattr(audio, "_pow_hash_hex", proof_hash)
    requirement, proof = audio._sentinel(
        object(), "token", "device", audio._OperationBudget(time.monotonic() + 30)
    )
    prepared = _decode(requests[0]["p"])
    assert requirement == "requirements"
    assert len(fingerprints) == 1
    assert prepared == fingerprints[0]
    if proof_required:
        assert [attempt[3] for attempt in attempts] == [0, 1, 2]
        assert _decode(proof) == attempts[-1]
        assert requests[1]["proofofwork"] == proof
        for attempt in attempts:
            assert len(attempt) == len(prepared) == 25
            assert all(attempt[i] == prepared[i] for i in range(25) if i not in (3, 9))
    else:
        assert not proof and not attempts
        assert "proofofwork" not in requests[1]


@pytest.mark.parametrize("timestamp, expected", [
    (0, "Thu Jan 01 1970 00:00:00 GMT+0000 (Coordinated Universal Time)"),
    (1709251199, "Thu Feb 29 2024 23:59:59 GMT+0000 (Coordinated Universal Time)"),
])
def test_browser_date_and_conversation_share_locale_independent_timezone(monkeypatch, timestamp, expected):
    # A gateway's local timezone/locale must not change the browser identity.
    monkeypatch.setattr(time, "timezone", -19800)
    monkeypatch.setattr(time, "tzname", ("IST", "IST"))
    monkeypatch.setattr(time, "strftime", lambda *_args: "localized host date")
    assert audio._browser_date(timestamp) == expected
    body = audio._conversation_body("speech")
    assert body["timezone"] == "UTC"
    assert body["timezone_offset_min"] == 0
