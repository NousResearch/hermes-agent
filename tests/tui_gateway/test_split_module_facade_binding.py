"""The tui_gateway siblings reach facade state through ``srv`` bound at the module tail.

Three ways that binding silently breaks, each seen once:

* a sibling imported FIRST (tests, the gateway process) runs server.py from its own tail import, and
  server.py's tail calls the sibling's ``register()`` while the sibling's ``srv`` is still unbound;
* a function rebinds ``srv`` locally (``for srv in ...``), so the earlier ``srv.<helper>`` in the
  same body is an ``UnboundLocalError`` — pyflakes cannot see it because the module-level ``srv``
  is bound after the function;
* a name looked up by string (``globals()[name]``) assumed the old rebound namespace, where the
  server's helpers were copied into every sibling; nothing is copied any more.
"""

from __future__ import annotations

import concurrent.futures
import pathlib
import subprocess
import sys

import yaml

from tui_gateway import server

REPO = pathlib.Path(__file__).resolve().parents[2]
SIBLINGS = sorted(p.stem for p in (REPO / "tui_gateway").glob("*.py") if not p.stem.startswith("__"))


def _import_first(module: str) -> tuple[str, int, str]:
    code = f"import sys, tui_gateway.{module}; from tui_gateway import server; print(len(server._methods), file=sys.stderr)"
    proc = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True, timeout=180)
    tail = proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else ""
    return module, proc.returncode, tail


def test_every_split_module_imports_first_in_a_clean_process():
    """Either import order completes: server-first (production) and sibling-first (tests, the
    gateway process importing methods_groups), and both end with the full method table."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        outcomes = list(pool.map(_import_first, SIBLINGS))
    failed = [(module, tail) for module, rc, tail in outcomes if rc != 0]
    assert not failed, failed
    counts = {tail for _module, _rc, tail in outcomes}
    assert counts == {str(len(server._methods))}, counts


def test_usage_bars_reaches_its_serializer_through_the_facade(monkeypatch):
    """``usage.bars`` serializes a real builder result instead of answering the ``available: False``
    fallback for every user (the serializer lives in billing_view, not in methods_session's globals)."""
    from agent.billing_usage import UsageBar, UsageModel

    model = UsageModel(available=True, status="healthy", plan_name="Pro", renews_at="2026-10-01T00:00:00Z",
                       subscription_remaining_usd=12.5, topup_remaining_usd=None, total_spendable_usd=12.5,
                       plan_bar=UsageBar(kind="plan", remaining_usd=12.5, total_usd=20.0, spent_usd=7.5))
    monkeypatch.setattr("agent.billing_usage.build_usage_model", lambda: model)

    result = server.handle_request({"jsonrpc": "2.0", "id": 1, "method": "usage.bars", "params": {}})["result"]

    assert result["available"] is True
    assert result["plan_name"] == "Pro"
    assert result["plan_bar"]["pct_used"] == 38


def test_profiles_configure_saves_mcp_toggles(tmp_path, monkeypatch):
    """Saving the MCP toggles from the Bots editor applies (``for srv in wanted`` used to shadow the
    facade binding and every save raised ``UnboundLocalError`` behind ``applied.mcp_servers=false``)."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    (home / "config.yaml").write_text(yaml.safe_dump({"mcp_servers": {
        "keep": {"command": "keep-server"}, "drop": {"command": "drop-server"}}}), encoding="utf-8")

    result = server.handle_request({"jsonrpc": "2.0", "id": 1, "method": "profiles.configure",
                                    "params": {"name": "default", "enabled_mcp_servers": ["keep"]}})["result"]

    assert result["applied"]["mcp_servers"] is True
    saved = yaml.safe_load((home / "config.yaml").read_text(encoding="utf-8"))["mcp_servers"]
    assert "disabled" not in saved["keep"]
    assert saved["drop"]["disabled"] is True
