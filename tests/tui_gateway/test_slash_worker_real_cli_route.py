"""The real slash worker answers ``/fast ultrafast`` on the session's endpoint, not the profile default's.

#141953: a Codex session on a profile whose default is a ``process://`` provider ran ``/fast`` in the
worker, whose ``HermesCLI`` seeded ``base_url`` from ``model.base_url``, so the Ultrafast gate saw the
default's URL and refused. Runs the worker's ``main()`` with a real ``HermesCLI``.
"""

from __future__ import annotations

import io
import json
import sys

import hermes_agent.cli as cli_mod
import hermes_agent.tui_gateway.slash_worker as sw

CODEX_URL = "https://chatgpt.com/backend-api/codex"


def test_fast_ultrafast_checks_the_session_endpoint(monkeypatch):
    import hermes_agent.agent.codex_service_tiers as tiers

    monkeypatch.setitem(cli_mod.CLI_CONFIG, "model", {
        "default": "claude-sonnet-5.5", "provider": "claude-subscription", "base_url": "process://claude-subscription"})
    asked, seen = [], {}
    monkeypatch.setattr(tiers, "codex_account_offers_tier",
                        lambda model, tier, *, base_url=None, access_token=None: asked.append(base_url) or True)
    real_run = sw._run
    monkeypatch.setattr(sw, "_run", lambda cli, command: seen.setdefault("cli", cli) and real_run(cli, command))
    monkeypatch.setattr(sw, "_prepare_slash_worker_runtime", lambda: None)
    monkeypatch.setattr(sw.sys, "stdin", io.StringIO(json.dumps({"id": 1, "command": "/fast ultrafast"}) + "\n"))
    out = io.StringIO()
    monkeypatch.setattr(sw.sys, "stdout", out)
    monkeypatch.setattr(sys, "argv", ["slash_worker", "--session-key", "s", "--model", "gpt-6-astra",
                                      "--provider", "openai-codex", "--base-url", CODEX_URL])

    sw.main()

    reply = json.loads(out.getvalue().splitlines()[-1])
    assert reply["ok"], reply
    assert asked == [CODEX_URL]
    assert seen["cli"].service_tier == "ultrafast"
    assert seen["cli"]._explicit_base_url is None  # a seed only: resume and /model still re-resolve the route
