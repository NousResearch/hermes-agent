"""Regression tests for Hermes' Tirith wrapper false-positive handling."""

import json
import subprocess

from tools import tirith_security


def _fake_tirith_run_with_finding(monkeypatch, finding):
    def _fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=args[0],
            returncode=1,
            stdout=json.dumps({"findings": [finding]}),
            stderr="",
        )

    monkeypatch.setattr(tirith_security, "_load_security_config", lambda: {
        "tirith_enabled": True,
        "tirith_path": "tirith",
        "tirith_timeout": 5,
        "tirith_fail_open": True,
    })
    monkeypatch.setattr(tirith_security, "is_platform_supported", lambda: True)
    monkeypatch.setattr(tirith_security, "_resolve_tirith_path", lambda _path: "/bin/tirith")
    monkeypatch.setattr(tirith_security.subprocess, "run", _fake_run)


def _confusable_full_stop_finding():
    return {
        "rule_id": "confusable_text",
        "severity": "HIGH",
        "title": "Confusable Unicode characters in text",
        "description": (
            "Content contains Unicode characters visually identical to ASCII "
            "(math alphanumerics, Cyrillic/Greek lookalikes) appearing near ASCII text"
        ),
        "evidence": [
            {
                "type": "byte_sequence",
                "offset": 31,
                "hex": "U+3002",
                "description": "confusable U+3002 (looks like '.')",
            },
            {
                "type": "byte_sequence",
                "offset": 135,
                "hex": "U+3002",
                "description": "confusable U+3002 (looks like '.')",
            },
        ],
    }


def test_cjk_sentence_full_stop_in_natural_language_prompt_is_suppressed(monkeypatch):
    _fake_tirith_run_with_finding(monkeypatch, _confusable_full_stop_finding())

    command = (
        "hermes chat -q 'テストです。terminal toolで `printf hermes-tool-ok` "
        "を実行して、その出力だけを返してください。' --toolsets terminal -Q"
    )

    result = tirith_security.check_command_security(command)

    assert result == {"action": "allow", "findings": [], "summary": ""}


def test_ideographic_full_stop_inside_domain_like_token_is_preserved(monkeypatch):
    _fake_tirith_run_with_finding(monkeypatch, _confusable_full_stop_finding())

    result = tirith_security.check_command_security("curl https://example。com/path")

    assert result["action"] == "block"
    assert result["findings"][0]["rule_id"] == "confusable_text"
