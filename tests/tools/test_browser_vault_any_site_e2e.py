"""End to end: a card the user saves for any site in `hermes vault add` is filled through the real tool
registry only on the page the user confirms, once per fill.

Real imports and a temp HERMES_HOME: the CLI wizard writes the real encrypted vault, the model-facing tool is
dispatched through the registry, and the card prompt goes through the real consent routing to the CLI's
approval callback. Only the browser is a stand-in (the supervisor's view of one tab; it records every
secret-bearing write).
"""

from __future__ import annotations

import json
import stat
from types import SimpleNamespace
from unittest.mock import patch

from agent.vault_store import normalize_origin

_PAN = "4111111111111111"
_CARD_ANSWERS = [_PAN, "", "07", "2029", "123", ""]  # hidden prompts, in PAYMENT_FIELDS order
_CARD_CONTROLS = [
    {"autocomplete": "cc-number", "index": 0, "type": "text"},
    {"autocomplete": "cc-exp", "index": 1, "type": "text"},
    {"autocomplete": "cc-csc", "index": 2, "type": "text"},
]


class _Page:
    def __init__(self, href: str):
        self.href, self.writes = href, []

    def focus_page(self, origin: str, accept=None) -> dict:
        return {"ok": bool(origin) and normalize_origin(self.href) == origin, "url": self.href}

    def evaluate_runtime(self, expression: str) -> dict:
        if expression.strip() == "window.location.href":
            return {"ok": True, "result": self.href}
        if "expectedOrigin" in expression:
            self.writes.append(expression)
            return {"ok": True, "result": json.dumps({"filled": 3})}
        return {"ok": True, "result": json.dumps(_CARD_CONTROLS)}


def _vault(args):
    from tools.registry import registry
    import tools.browser_vault_tool  # registers browser_vault

    return registry.dispatch("browser_vault", args, task_id="any-site-e2e")


def test_cli_saved_any_site_card_fills_once_per_confirmed_site(tmp_path, monkeypatch):
    from agent import redact
    from hermes_cli import vault as vault_cli
    from tools import browser_vault_tool
    from tools.terminal_tool import set_approval_callback

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr("agent.vault_backends.base.is_installed", lambda name: False)
    monkeypatch.setattr("agent.redact._REDACT_ENABLED", True)

    # (a) the human saves the card for any site in the CLI wizard
    answers, hidden = ["Visa", "any", "y"], list(_CARD_ANSWERS)
    monkeypatch.setattr("builtins.input", lambda prompt="": answers.pop(0))
    monkeypatch.setattr(vault_cli.getpass, "getpass", lambda prompt="": hidden.pop(0))
    vault_cli._cmd_add(SimpleNamespace(kind="payment"))
    vault_file = home / "vault" / "vault.json.enc"
    assert stat.S_IMODE(vault_file.stat().st_mode) == 0o600
    assert _PAN.encode() not in vault_file.read_bytes()

    # (b) the model sees it as available for any site, never its number
    listed = _vault({"action": "list"})
    [item] = json.loads(listed)["items"]
    assert item["available"] is True and item["origin_match"] == "any_site" and _PAN not in listed

    asked, answer = [], ["once"]

    def approval(command, description, **kwargs):
        asked.append(command)
        return answer[0]

    set_approval_callback(approval)
    try:
        # (c) shop A: the prompt names shop A; approved once; the write is pinned to shop A
        page_a = _Page("https://shop-a.example/checkout")
        with patch.object(browser_vault_tool, "_ensure_supervisor", return_value=page_a):
            out = _vault({"action": "fill", "handle": item["handle"]})
        assert json.loads(out)["success"] is True and _PAN not in out
        assert asked == ["Fill payment card 'Visa' on https://shop-a.example"]
        [write] = page_a.writes
        assert '"https://shop-a.example"' in write

        # (d) shop B: asked again (nothing is remembered per card or site); denied → nothing written
        answer[0] = "deny"
        page_b = _Page("https://shop-b.example/checkout")
        with patch.object(browser_vault_tool, "_ensure_supervisor", return_value=page_b):
            out = json.loads(_vault({"action": "fill", "handle": item["handle"]}))
        assert out["error_type"] == "payment_declined"
        assert asked[1:] == ["Fill payment card 'Visa' on https://shop-b.example"]
        assert page_b.writes == []

        # (e) the card number filled on shop A is masked if a later browser result echoes it
        assert _PAN not in redact.redact_registered_vault_values(f"page shows {_PAN}")
    finally:
        set_approval_callback(None)
        redact.clear_vault_redaction_values()
