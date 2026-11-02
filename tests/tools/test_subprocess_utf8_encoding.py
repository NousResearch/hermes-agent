"""Regression test for #53428 (fix PR #55339): ``subprocess.run(text=True)``
decoding of secret-manager CLI output must never crash on undecodable bytes.

On Chinese Windows (cp936/GBK) ``text=True`` without ``encoding=`` decodes with
the locale codepage and crashes on non-GBK bytes (#47939, #53428, #57238);
``encoding='utf-8'`` alone still raises on non-UTF-8 bytes emitted by
Windows-native CLIs. ``scripts/check-windows-footguns.py`` enforces the
``encoding=`` half repo-wide; this drives the real ``_op_whoami`` against a
child that emits invalid UTF-8 so the ``errors='replace'`` half is exercised
on every host.
"""

from __future__ import annotations

import sys
from pathlib import Path


def test_op_whoami_survives_non_utf8_cli_output(tmp_path, monkeypatch):
    from hermes_cli import onepassword_secrets_cli as op_cli

    # _op_whoami runs ``[binary, "whoami"]``. Using the interpreter as the
    # binary makes that ``python whoami``, i.e. run ./whoami from the cwd:
    # a portable stand-in for an ``op`` that prints a non-UTF-8 account name.
    (tmp_path / "whoami").write_text(
        "import sys\n"
        "sys.stdout.buffer.write(b'\\xff\\xfe caf\\xc3\\xa9@example.com\\n')\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    identity = op_cli._op_whoami(Path(sys.executable), account="")

    assert identity is not None
    assert "café@example.com" in identity
