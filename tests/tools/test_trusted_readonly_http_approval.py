"""Tests for profile-scoped trusted read-only HTTP approval."""

from unittest.mock import patch as mock_patch

from tools.approval import check_all_command_guards


def _config(prefixes):
    return {
        "approvals": {
            "mode": "manual",
            "trusted_readonly_http_prefixes": prefixes,
        }
    }


def test_trusted_readonly_urllib_heredoc_is_approved(monkeypatch):
    monkeypatch.setenv("HERMES_INTERACTIVE", "1")
    command = """python3 - <<'PY'
import urllib.request
urls = [
    'http://10.0.0.10:8189/system_stats',
    'http://10.0.0.10:8189/queue',
    'http://10.0.0.10:8192/health',
]
for url in urls:
    with urllib.request.urlopen(url, timeout=15) as r:
        body = r.read(2000).decode('utf-8', 'replace')
        print('STATUS', r.status, body[:300].replace('\\n', ' '))
PY"""
    with mock_patch(
        "hermes_cli.config.load_config",
        return_value=_config([
            "http://10.0.0.10:8189",
            "http://10.0.0.10:8192",
        ]),
    ):
        result = check_all_command_guards(command, "local")

    assert result["approved"] is True
    assert result["trusted_readonly_http"] is True


def test_trusted_readonly_urllib_heredoc_rejects_untrusted_url(monkeypatch):
    monkeypatch.setenv("HERMES_INTERACTIVE", "1")
    command = """python3 <<'PY'
import urllib.request
urllib.request.urlopen('http://example.com/system_stats', timeout=15)
PY"""
    with mock_patch(
        "hermes_cli.config.load_config",
        return_value=_config(["http://10.0.0.10:8189"]),
    ):
        result = check_all_command_guards(
            command,
            "local",
            approval_callback=lambda *_a, **_kw: "deny",
        )

    assert result["approved"] is False
    assert result["outcome"] == "denied"


def test_trusted_readonly_urllib_heredoc_rejects_post_data(monkeypatch):
    monkeypatch.setenv("HERMES_INTERACTIVE", "1")
    command = """python3 <<'PY'
import urllib.request
urllib.request.urlopen('http://10.0.0.10:8189/prompt', data=b'{}')
PY"""
    with mock_patch(
        "hermes_cli.config.load_config",
        return_value=_config(["http://10.0.0.10:8189"]),
    ):
        result = check_all_command_guards(
            command,
            "local",
            approval_callback=lambda *_a, **_kw: "deny",
        )

    assert result["approved"] is False
    assert result["outcome"] == "denied"


def test_trusted_readonly_urllib_heredoc_rejects_unsafe_import(monkeypatch):
    monkeypatch.setenv("HERMES_INTERACTIVE", "1")
    command = """python3 <<'PY'
import os
import urllib.request
urllib.request.urlopen('http://10.0.0.10:8189/system_stats', timeout=15)
PY"""
    with mock_patch(
        "hermes_cli.config.load_config",
        return_value=_config(["http://10.0.0.10:8189"]),
    ):
        result = check_all_command_guards(
            command,
            "local",
            approval_callback=lambda *_a, **_kw: "deny",
        )

    assert result["approved"] is False
    assert result["outcome"] == "denied"
