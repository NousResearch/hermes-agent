"""Two-sided contract: every notification name ``tui_gateway`` emits is listed in
``apps/shared/src/gateway-events.json`` and nothing in the JSON is orphaned.

The TypeScript half (``apps/shared/src/gateway-events.test.ts``) pins the typed
``GatewayEventMap`` to the same JSON, so a name added on either side alone goes red
somewhere. This file reads only Python sources and the JSON (never ``.ts`` text —
see ``tui_gateway/AGENTS.md``).

Names are collected from the emitter side: literal first arguments to the emit
helpers, plus the tables that derive names at runtime (``_EXPIRING_REQUESTS`` →
``*.expire``, the change-watcher table, child delta mirroring, the subagent relay
events from ``tools/delegate_tool*.py``, the ``desktop_ui`` tool emitters, and the
literal ``gateway.ready`` / ``setup.ready`` / browser-controller frames).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
CONTRACT = REPO / "apps" / "shared" / "src" / "gateway-events.json"
GATEWAY_DIR = REPO / "tui_gateway"

# Every helper whose first positional argument is the wire ``type``.
_EMIT_HELPERS = (
    "_emit", "_block", "_read_block", "_broadcast_global_event", "_voice_emit", "_pet_emit", "_emit_tool_lifecycle")
_LITERAL_EMIT = re.compile(r"\b(?:%s)\(\s*\"([a-z_][a-z0-9_.]*)\"" % "|".join(_EMIT_HELPERS))
# ``{"type": "gateway.ready", ...}`` literal frames (entry.py / ws.py) and other
# ``"type": "<name>"`` params written straight into an ``event`` frame.
_LITERAL_FRAME = re.compile(r"\"method\":\s*\"event\".{0,120}?\"type\":\s*\"([a-z_][a-z0-9_.]*)\"", re.S)
_SIDE_AGENT = re.compile(r"_spawn_side_agent\((?:[^()]|\([^()]*\))*?\"([a-z_][a-z0-9_.]*\.complete)\"", re.S)
_SUBAGENT_RELAY = re.compile(r"\"(subagent\.[a-z_]+)\"")
_DESKTOP_UI_EMIT = re.compile(r"desktop_ui\.(?:emit|emit_or_error)\(\s*\"([a-z_][a-z0-9_.]*)\"")
_BROKER_FRAME = re.compile(r"^FRAME_[A-Z_]+ = \"(browser\.controller\.[a-z_]+)\"", re.M)
_SETUP_READY = re.compile(r"^SETUP_READY_EVENT = \"([a-z_.]+)\"", re.M)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def emitted_event_names() -> set[str]:
    names: set[str] = set()
    for src in GATEWAY_DIR.glob("*.py"):
        text = _read(src)
        names.update(_LITERAL_EMIT.findall(text))
        names.update(_LITERAL_FRAME.findall(text))
        names.update(_SIDE_AGENT.findall(text))
    # ``.request`` bridges that time out fire ``f"{event.removesuffix('.request')}.expire"``.
    from tui_gateway.server import _EXPIRING_REQUESTS

    names.update(f"{event.removesuffix('.request')}.expire" for event in _EXPIRING_REQUESTS)
    from tui_gateway.change_watcher import _CHANGE_WATCHES

    names.update(_CHANGE_WATCHES)
    from tui_gateway.agent_callbacks import _CHILD_DELTA_EVENTS

    names.update(_CHILD_DELTA_EVENTS.values())
    # ``_progress_subagent`` relays every ``subagent.*`` event verbatim EXCEPT the child's per-token
    # ``subagent.text`` (mirrored into the watch window as ``message.delta`` instead); the names live in the relay.
    for src in (REPO / "tools").glob("delegate_tool*.py"):
        names.update(_SUBAGENT_RELAY.findall(_read(src)))
    names.discard("subagent.text")
    for src in (REPO / "tools").glob("*.py"):
        names.update(_DESKTOP_UI_EMIT.findall(_read(src)))
    names.update(_BROKER_FRAME.findall(_read(REPO / "gateway" / "browser_control_broker.py")))
    names.update(_SETUP_READY.findall(_read(REPO / "hermes_cli" / "free_tier_bootstrap.py")))
    # Dispatch-table keys that double as the emitted name (``_PROGRESS_HANDLERS`` re-emits
    # ``event_type``) are already literal ``_emit("...")`` calls inside their handlers.
    return names


@pytest.fixture(scope="module")
def contract() -> list[str]:
    return json.loads(_read(CONTRACT))


def test_contract_is_sorted_and_unique(contract):
    assert contract == sorted(set(contract)), "gateway-events.json must be a sorted, duplicate-free list"


def test_every_emitted_event_is_in_the_contract(contract):
    missing = emitted_event_names() - set(contract)
    assert not missing, (
        f"tui_gateway emits {sorted(missing)} but apps/shared/src/gateway-events.json does not list them; "
        "add the name(s) there AND to BACKEND_EVENT_NAMES / GatewayEventMap in apps/shared/src/gateway-events.ts")


def test_contract_has_no_orphan_names(contract):
    orphans = set(contract) - emitted_event_names()
    assert not orphans, (
        f"apps/shared/src/gateway-events.json lists {sorted(orphans)} but no tui_gateway emitter names them; "
        "drop the entry (and its GatewayEventMap key) or wire the emitter")
