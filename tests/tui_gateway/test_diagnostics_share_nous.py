"""diagnostics.share_nous RPC — Desktop "Send Diagnostics" upload path.

Contract pinned:
* Reuses the CLI ``--nous`` pipeline (collect_share_bundle → build_nous_bundle
  → share_to_nous) with redaction FORCED on — the client cannot disable it.
* ``error_context`` and ``extra_files`` are redacted server-side, labels
  sanitized, sizes capped.
* Upload failures return a structured ``{ok: False, error}`` envelope, never a
  JSON-RPC error (the desktop renders them inline in the modal).
"""

from __future__ import annotations

import gzip
import json

import pytest

from tui_gateway import server


def _handler():
    fn = server._methods.get("diagnostics.share_nous")
    assert fn is not None, "diagnostics.share_nous not registered"
    return fn


@pytest.fixture()
def captured_upload(monkeypatch, tmp_path):
    """Mock ONLY the network leg; the bundle pipeline runs for real."""
    captured: dict = {}

    def _fake_share(blob: bytes) -> dict:
        captured["blob"] = blob
        return {
            "viewUrl": "https://nas.example/view/abc123",
            "id": "abc123",
            "expiresAt": "2026-09-05T00:00:00Z",
        }

    import hermes_cli.diagnostics_upload as du

    monkeypatch.setattr(du, "share_to_nous", _fake_share)
    return captured


def _envelope(blob: bytes) -> dict:
    return json.loads(gzip.decompress(blob).decode("utf-8"))


def test_share_nous_uploads_redacted_bundle(captured_upload):
    result = _handler()("rid-1", {})
    payload = result["result"]

    assert payload["ok"] is True
    assert payload["view_url"] == "https://nas.example/view/abc123"
    assert payload["upload_id"] == "abc123"

    envelope = _envelope(captured_upload["blob"])
    assert envelope["format"].startswith("hermes-debug-share/")
    assert envelope["redacted"] is True
    assert "report" in envelope["files"]


def test_share_nous_attaches_redacted_error_context(captured_upload):
    secret = "sk-abc123def456ghi789jkl012mno345pqr678"
    result = _handler()(
        "rid-2",
        {"error_context": f"layer: provider\ncode: rate_limit\nkey was {secret}"},
    )
    assert result["result"]["ok"] is True

    files = _envelope(captured_upload["blob"])["files"]
    context = files.get("error-context.txt", "")
    assert "layer: provider" in context
    assert secret not in context, "secret leaked through error_context redaction"


def test_share_nous_client_text_gets_upload_safe_log_redaction(captured_upload):
    """Client artifacts must ride the SAME redactor as backend logs
    (redact_debug_support_text): secrets AND email addresses — not just the bare
    secret pass, which leaves emails through (review finding on #92020)."""
    secret = "sk-abc123def456ghi789jkl012mno345pqr678"
    result = _handler()(
        "rid-2b",
        {
            "error_context": "user reported by alice@example.com",
            "extra_files": {"desktop.log": f"login bob@example.com token={secret}"},
        },
    )
    assert result["result"]["ok"] is True

    files = _envelope(captured_upload["blob"])["files"]
    assert "alice@example.com" not in files["error-context.txt"]
    assert "[REDACTED_EMAIL]" in files["error-context.txt"]
    assert "bob@example.com" not in files["client/desktop.log"]
    assert secret not in files["client/desktop.log"]


def test_redacted_support_egress_scrubs_structured_values_and_errors(
    captured_upload, monkeypatch
):
    from hermes_constants import get_hermes_home

    canaries = {
        "url": "r36_REDACTED_url_canary_0123456789",
        "header": "r36_REDACTED_header_canary_0123456789",
        "env": "r36_REDACTED_env_canary_0123456789",
        "argv": "r36_REDACTED_argv_canary_0123456789",
        "bearer": "r36_REDACTED_bearer_canary_0123456789",
        "error": "r36_REDACTED_error_canary_0123456789",
    }
    home = get_hermes_home()
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(
        "fallback_providers:\n"
        "  - provider: custom\n"
        "    model: support-fixture\n"
        "    base_url: 'https://outer.invalid/?next=https://inner.invalid/"
        f"?accessToken={canaries['url']}'\n"
        "    extra_headers:\n"
        f"      X-API-Key: {canaries['header']}\n"
        "    env:\n"
        f"      AWS_SECRET_ACCESS_KEY: {canaries['env']}\n"
        f"    args: ['--api-key', '{canaries['argv']}']\n",
        encoding="utf-8",
    )
    structured = (
        f"headers={{'Proxy-Authorization': 'Basic {canaries['header']}'}} "
        f"Command ['provider', '--api-key', '{canaries['argv']}'] "
        f"Bearer {canaries['bearer']} "
        f"https://a.invalid/?next=%2Fx%3Ffoo%26token%3D{canaries['url']}"
    )

    result = _handler()(
        "rid-structured",
        {"error_context": structured, "extra_files": {"desktop.log": structured}},
    )

    assert result["result"]["ok"] is True
    envelope = _envelope(captured_upload["blob"])
    assert envelope["redacted"] is True
    rendered = json.dumps(envelope, sort_keys=True)
    assert "support-fixture" in rendered
    assert not [canary for canary in canaries.values() if canary in rendered]

    # An oversized client value is scrubbed through a bounded window (line-aligned
    # when a line ends in [2x, 3x) cap), with the same capped output as a full scrub.
    import hermes_cli.debug_redaction as dr

    real_redact, scrubbed_sizes = dr.redact_debug_support_text, []

    def _spy(value, **kwargs):
        scrubbed_sizes.append(len(value))
        return real_redact(value, **kwargs)

    line = f"cookie={canaries['header']} " + "x" * 40 + "\n"
    monkeypatch.setattr(dr, "redact_debug_support_text", _spy)
    assert _handler()("rid-big", {"error_context": line * 2_000})["result"]["ok"] is True
    monkeypatch.setattr(dr, "redact_debug_support_text", real_redact)
    big_context = _envelope(captured_upload["blob"])["files"]["error-context.txt"]
    assert max(scrubbed_sizes) <= 2 * 8_000 + len(line)
    assert big_context == real_redact((line * 2_000).strip(), max_chars=8_000)
    scrubbed_sizes.clear()
    monkeypatch.setattr(dr, "redact_debug_support_text", _spy)
    one_line = f"cookie={canaries['header']} " + "x" * 2_000_000
    assert _handler()("rid-one-line", {"error_context": one_line})["result"]["ok"] is True
    monkeypatch.setattr(dr, "redact_debug_support_text", real_redact)
    assert max(scrubbed_sizes) <= 3 * 8_000

    import hermes_cli.diagnostics_upload as du

    def _fail(_blob: bytes) -> dict:
        # The tail is already scrubbed (a wrapped inner error); re-scrubbing it
        # must be idempotent, not turn ``[REDACTED]`` into ``[REDACTED]]``.
        raise RuntimeError(
            "PUT https://upload.invalid/?X-Amz-Security-Token="
            f"{canaries['error']} headers={{'X-API-Key': '{canaries['header']}'}} "
            f"x-api-key: Basic {canaries['header']} "
            + real_redact(f"cookie={canaries['error']}; --api-key {canaries['error']} "
                          f"--header X-API-Key {canaries['error']}")
            + " --header X-API-Key Basic [REDACTED]"
        )

    monkeypatch.setattr(du, "share_to_nous", _fail)
    failure = _handler()("rid-structured-error", {})["result"]
    assert failure["ok"] is False
    assert canaries["error"] not in failure["error"]
    assert canaries["header"] not in failure["error"]
    assert "[REDACTED]]" not in failure["error"]
    assert failure["error"].endswith(" --header X-API-Key Basic [REDACTED]")
    assert real_redact(failure["error"]) == failure["error"]
    # Prose after an already-masked value is not a key: kept, and a fixed point.
    benign = real_redact("x-api-key: *** see docs")
    assert benign.endswith(" see docs") and real_redact(benign) == benign
    for prose in ("see: documentation", "time: 12:30:45"):
        kept = real_redact(f"x-api-key: *** {prose}")
        assert kept.endswith(f" {prose}") and real_redact(kept) == kept
    # An RFC 6750 challenge is auth-params, not a token.
    assert real_redact('Bearer realm="api"') == 'Bearer realm="api"'
    # Short, colon-bearing and double-encoded keys are keys, not prose.
    for leaky, key in (
        ("x-api-key: Bearer abc1234", "abc1234"),
        ("-H 'x-auth-token: *** s3cr3t'", "s3cr3t"),
        ("x-api-key: *** qwertyuiopas", "qwertyuiopas"),
        ("--header x-api-key *** abc1234", "abc1234"),
        ("x-api-key: *** " + "q" * 32, "q" * 32),
        ("--header x-api-key *** " + "Q" * 32, "Q" * 32),
        ("x-api-key: *** sk-abc:xyz123456", "xyz123456"),
        ("Basic user:passw0rd", "passw0rd"),
        ("Bearer abcd1234=efgh5678 x", "efgh5678"),
        ("?r=x%2526access_token%253Dtok123456789", "tok123456789"),
    ):
        assert key not in real_redact(leaky)


def test_share_nous_linkless_success_is_a_failure(monkeypatch):
    """ok:true with neither view_url nor id would strand the user with an
    unreferencable upload — surface it as a structured failure instead."""
    import hermes_cli.diagnostics_upload as du

    monkeypatch.setattr(du, "share_to_nous", lambda blob: {})

    result = _handler()("rid-2c", {})
    payload = result["result"]
    assert payload["ok"] is False
    assert payload["error"]


def test_share_nous_extra_files_sanitized_and_redacted(captured_upload):
    secret = "sk-abc123def456ghi789jkl012mno345pqr678"
    result = _handler()(
        "rid-3",
        {
            "extra_files": {
                "desktop.log": f"boot ok\ntoken={secret}\n",
                "../../etc/passwd": "nope",
                "ok name (1).txt": "fine",
                7: "not-a-str-label",
                "empty": "   ",
            }
        },
    )
    assert result["result"]["ok"] is True

    files = _envelope(captured_upload["blob"])["files"]
    assert "client/desktop.log" in files
    assert secret not in files["client/desktop.log"]
    # Path separators are stripped from labels; traversal shapes can't survive.
    assert not any("/etc/passwd" in k or ".." in k for k in files)
    assert "client/ok name (1).txt" in files
    # Non-string labels and blank bodies are dropped.
    assert not any(k.startswith("client/7") for k in files)
    assert "client/empty" not in files


def test_share_nous_upload_failure_is_structured(monkeypatch):
    import hermes_cli.diagnostics_upload as du

    def _boom(blob: bytes) -> dict:
        raise RuntimeError("NAS unavailable")

    monkeypatch.setattr(du, "share_to_nous", _boom)

    result = _handler()("rid-4", {})
    payload = result["result"]
    assert payload["ok"] is False
    assert "NAS unavailable" in payload["error"]
