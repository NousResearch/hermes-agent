"""llm.oneshot honors an explicit ``provider``/``model`` pin — regression.

The desktop prompt-enhancer plugin carries its own per-task model pin: its
composer picker sends ``provider``/``model`` on every ``llm.oneshot`` request.
Before this arm existed the params contract rejected the extra keys outright
(``extra="forbid"`` → 4000 "out of sync"), and ``run_oneshot`` could only reach
``auxiliary.<task>`` config or the session/main model — a caller-side pin had
no way in. These tests pin the three layers:

* contract: ``LlmOneshotParams`` accepts and round-trips provider/model;
* handler: ``llm.oneshot`` forwards them to ``run_oneshot`` (blank strings
  normalize to None so they don't shadow the fallback arms);
* run_oneshot: the pin passes through to ``call_llm`` as the explicit-pin arm.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

REPO = Path(__file__).resolve().parents[2]


# ------------------------------------------------------------------ contract
def test_llm_oneshot_contract_accepts_provider_and_model():
    from tui_gateway.contracts.sessions import LlmOneshotParams

    p = LlmOneshotParams(
        template="enhance", input="hello", task="prompt_enhancement",
        provider="z-ai", model="glm-5.3-flash",
    )
    assert p.provider == "z-ai"
    assert p.model == "glm-5.3-flash"
    # Older callers send neither — both default to None.
    bare = LlmOneshotParams(template="enhance", input="hi")
    assert bare.provider is None and bare.model is None


def test_openrpc_contract_lists_the_new_params():
    spec = json.loads((REPO / "apps/shared/src/gateway-contract.openrpc.json").read_text(encoding="utf-8"))
    schemas = spec["components"]["schemas"]["LlmOneshotParams"]
    props = schemas["properties"]
    assert "provider" in props and "model" in props


# ------------------------------------------------------------------- handler
def _dispatch_llm_oneshot(params: dict) -> dict:
    """Invoke the registered llm.oneshot handler directly (per test_goal_command
    precedent), capturing the kwargs that reach run_oneshot."""
    from tui_gateway import server

    captured: dict = {}

    def _fake_run(**kwargs):
        captured.update(kwargs)
        return "ok"

    with patch("agent.oneshot.run_oneshot", side_effect=_fake_run):
        out = server._methods["llm.oneshot"](7, params)
    assert out["result"]["text"] == "ok", out
    return captured


def test_llm_oneshot_forwards_pin_and_strips_blanks():
    cap = _dispatch_llm_oneshot(
        {"instructions": "i", "input": "x", "provider": "z-ai", "model": " glm-5.3-flash "}
    )
    assert cap["provider"] == "z-ai"
    assert cap["model"] == "glm-5.3-flash"

    # Whitespace-only must NOT become a pin (must fall through to the
    # session/task/main-model arms of the resolver).
    cap2 = _dispatch_llm_oneshot({"instructions": "i", "input": "x", "provider": "  ", "model": ""})
    assert cap2["provider"] is None and cap2["model"] is None


def test_llm_oneshot_without_pin_is_unchanged():
    cap = _dispatch_llm_oneshot({"instructions": "i", "input": "x"})
    assert cap["provider"] is None and cap["model"] is None


# ------------------------------------------------------------- run_oneshot
def test_run_oneshot_signature_gains_provider_model():
    from agent.oneshot import run_oneshot

    sig = inspect.signature(run_oneshot)
    for name in ("provider", "model"):
        assert name in sig.parameters
        assert sig.parameters[name].default is None


def test_run_oneshot_passes_explicit_pin_to_call_llm():
    import agent.oneshot as oneshot

    with patch("agent.oneshot.call_llm") as mock:
        mock.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="hi", reasoning_content=None))]
        )
        oneshot.run_oneshot(
            instructions="i", user_input="u",
            task="prompt_enhancement", provider="z-ai", model="glm-5.3-flash",
        )
    kwargs = mock.call_args.kwargs
    assert kwargs.get("provider") == "z-ai"
    assert kwargs.get("model") == "glm-5.3-flash"


def test_run_oneshot_default_keeps_pin_none():
    from unittest.mock import MagicMock

    import agent.oneshot as oneshot

    with patch("agent.oneshot.call_llm") as mock:
        mock.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="hi", reasoning_content=None))]
        )
        oneshot.run_oneshot(instructions="i", user_input="u", task="title_generation")
    kwargs = mock.call_args.kwargs
    assert kwargs.get("provider") is None
    assert kwargs.get("model") is None
