"""Regression test for issue #53428 — subprocess.run(text=True) without
explicit ``encoding=`` triggers ``UnicodeDecodeError`` on Chinese Windows
(cp936/GBK default encoding).

PR #55339 covers 21 call sites in ``agent/``, ``gateway/``, ``cli.py``,
``cron/``, plus 5 more in ``tools/`` and ``hermes_cli/`` (main.py,
setup.py, tts_tool.py, transcription_tools.py). The two call sites it
misses — ``hermes_cli/onepassword_secrets_cli.py::_op_whoami`` and
``_op_version`` — are guarded here.

Without ``encoding=``, ``text=True`` decodes child output with
``locale.getpreferredencoding(False)`` — cp936 on Chinese Windows —
which crashes on non-GBK bytes (issues #47939, #53428, #57238).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch, MagicMock


def _assert_utf8_kwargs(mock_run):
    """Lift the encoding/errors kwargs out of the mock and assert them."""
    assert mock_run.called, "subprocess.run was not called"
    kwargs = mock_run.call_args.kwargs
    assert kwargs.get("encoding") == "utf-8", (
        f"subprocess.run called without encoding='utf-8' "
        f"(got encoding={kwargs.get('encoding')!r}). "
        f"On Chinese Windows (cp936), text=True without explicit encoding "
        f"crashes with UnicodeDecodeError on non-GBK bytes. See #53428."
    )
    assert kwargs.get("errors") == "replace", (
        f"subprocess.run called without errors='replace' "
        f"(got errors={kwargs.get('errors')!r}). encoding='utf-8' alone "
        f"still raises UnicodeDecodeError on non-UTF-8 bytes emitted by "
        f"Windows-native CLIs. See #53428."
    )


def test_op_whoami_passes_utf8_encoding(tmp_path):
    """_op_whoami must pass encoding='utf-8', errors='replace' so op CLI
    output containing non-ASCII account names doesn't crash on cp936."""
    from hermes_cli import onepassword_secrets_cli as op_cli

    fake_binary = tmp_path / "op"
    fake_binary.write_bytes(b"")
    with patch.object(op_cli.subprocess, "run") as mock_run:
        mock_run.return_value = MagicMock(
            returncode=0, stdout="user@example.com", stderr=""
        )
        op_cli._op_whoami(fake_binary, account="")
    _assert_utf8_kwargs(mock_run)


def test_op_version_passes_utf8_encoding(tmp_path):
    """_op_version must pass encoding='utf-8', errors='replace' so op CLI
    output containing non-ASCII bytes doesn't crash on cp936. Pairs with
    _op_whoami — both run in the same setup/status CLI flow."""
    from hermes_cli import onepassword_secrets_cli as op_cli

    fake_binary = tmp_path / "op"
    fake_binary.write_bytes(b"")
    with patch.object(op_cli.subprocess, "run") as mock_run:
        mock_run.return_value = MagicMock(
            returncode=0, stdout="2.24.0", stderr=""
        )
        op_cli._op_version(fake_binary)
    _assert_utf8_kwargs(mock_run)
