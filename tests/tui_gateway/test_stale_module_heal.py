"""Regression: a desktop ``hermes serve`` (TUI gateway) must survive an in-place checkout update.

Field failure (Hermes gateway error report, 2026-09-14T02:52:31Z):

    ImportError: cannot import name 'profile_from_session_key_namespace' from
    'gateway.session' (/home/agent/.hermes/hermes-agent/gateway/session.py)
    ... tui_gateway/session_compression.py:15, in _tui_compression_config_signature
    ... gateway/run.py:2091

``hermes update`` pulled the checkout at 22:33 (adding ``profile_from_session_key_namespace``
to ``gateway.session`` and ``gateway.run``'s import of it) while the desktop's ``hermes serve``
process — started hours earlier — kept the PRE-update ``gateway.session`` in ``sys.modules``.
Its lazy ``from gateway.run import GatewayRunner`` then failed on every turn until restart.

The import site now goes through ``hermes_module_staleness.import_symbol``, which evicts the
stale ``gateway`` subtree and retries. This runs in a subprocess because the heal deliberately
purges real Hermes modules, and the snapshot/restore dance would still leave module-level
references in other already-imported test modules pointing at evicted objects.
"""

from __future__ import annotations

import os
import subprocess
import sys

REPO_ROOT = "."


def _spawn_stale_session_signature_probe() -> tuple:
    """Cache a pre-update ``gateway.session``, then read the compression signature."""
    code = (
        "import sys, types\n"
        "import gateway  # the long-lived process already had the package cached\n"
        "real = sys.modules['gateway.session']\n"
        "stale = types.ModuleType('gateway.session')\n"
        "stale.__dict__.update({k: v for k, v in real.__dict__.items()\n"
        "                       if k != 'profile_from_session_key_namespace'})\n"
        "sys.modules['gateway.session'] = stale\n"
        "sys.modules.pop('gateway.run', None)\n"
        "# Precondition: the reported crash shape (old inline import) must still reproduce.\n"
        "try:\n"
        "    from gateway.run import GatewayRunner  # noqa: F401\n"
        "except ImportError as exc:\n"
        "    assert 'profile_from_session_key_namespace' in str(exc), exc\n"
        "else:\n"
        "    sys.stdout.write('NO_REPRO: stale cache did not block gateway.run\\n')\n"
        "    sys.exit(3)\n"
        "sys.modules.pop('gateway.run', None)\n"
        "from tui_gateway.session_compression import _tui_compression_config_signature\n"
        "sig = _tui_compression_config_signature({'compression': {'tail_mode': 'x'}})\n"
        "assert ('compression.tail_mode', 'x') in sig, sig\n"
        "sys.stdout.write('OK healed=%s\\n' % ('gateway.run' in sys.modules))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ},
    )
    return proc.returncode, proc.stdout, proc.stderr


def test_compression_signature_heals_stale_gateway_session():
    rc, out, err = _spawn_stale_session_signature_probe()
    combined = out + err
    assert rc == 0, f"stale-module heal failed (rc={rc}): {err!r} / {out!r}"
    assert "OK healed=True" in combined, f"unexpected output: {out!r} / {err!r}"
