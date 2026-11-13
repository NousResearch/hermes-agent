"""The committed TypeScript + OpenRPC contract files are exactly what ``tui_gateway/contracts``
renders, and the contract catalog covers the whole wire.

Regenerate with ``.venv/bin/python scripts/gen_gateway_contracts.py`` when a model changes. The
two files are listed in ``scripts/ci/classify_changes.py::_PY_RELEVANT_CONTRACT_FILES`` so a
TS-only PR that edits them still runs this test.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
GEN = REPO / "scripts" / "gen_gateway_contracts.py"


@pytest.fixture(scope="module")
def gen():
    spec = importlib.util.spec_from_file_location("gen_gateway_contracts", GEN)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_generated_files_are_current(gen):
    """Both committed artefacts equal an in-memory regeneration (byte-for-byte)."""
    stale = [path.relative_to(REPO) for path, text in gen.render_all().items()
             if (path.read_text(encoding="utf-8") if path.exists() else None) != text]
    assert not stale, f"stale generated contract files {stale}: run scripts/gen_gateway_contracts.py"


# The emitter inventory the old gateway-events.json scan used, kept as the completeness oracle:
# names must come from CODE the gateway runs, never from the contract tables themselves.
_EMIT_HELPERS = ("_emit", "_broadcast_global_event", "_voice_emit", "_pet_emit", "_emit_tool_lifecycle")
_LITERAL_EMIT = re.compile(r"\b(?:{})\(\s*\"([a-z_][a-z0-9_.]*)\"".format("|".join(_EMIT_HELPERS)))
_REQUEST_HELPERS = ("server_requests\\.send", "server_requests\\.send_async", "_ask", "_read_block")
_LITERAL_REQUEST = re.compile(r"\b(?:{})\(\s*\"([a-z_][a-z0-9_.]*)\"".format("|".join(_REQUEST_HELPERS)))
_LITERAL_FRAME = re.compile(r"\"method\":\s*\"event\".{0,120}?\"type\":\s*\"([a-z_][a-z0-9_.]*)\"", re.DOTALL)
_SIDE_AGENT = re.compile(r"_spawn_side_agent\((?:[^()]|\([^()]*\))*?\"([a-z_][a-z0-9_.]*\.complete)\"", re.DOTALL)
_SUBAGENT_RELAY = re.compile(r"\"(subagent\.[a-z_]+)\"")
_DESKTOP_UI_EMIT = re.compile(r"desktop_ui\.(?:emit|emit_or_error)\(\s*\"([a-z_][a-z0-9_.]*)\"")
_SETUP_READY = re.compile(r"^SETUP_READY_EVENT = \"([a-z_.]+)\"", re.MULTILINE)
_FREE_TIER_CHALLENGE = re.compile(r"^CHALLENGE_EVENT = \"([a-z_.]+)\"", re.MULTILINE)
# Session-authority local-viewer turn projections (gateway/local_viewer_events.py).
_LOCAL_VIEWER_PUBLISH = re.compile(r"\b(?:publish|_publish_execution)\(\s*\"([a-z_][a-z0-9_.]*)\"")


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def emitted_event_names() -> set[str]:
    names: set[str] = set()
    for src in (REPO / "tui_gateway").glob("*.py"):
        text = _read(src)
        names.update(_LITERAL_EMIT.findall(text))
        names.update(_LITERAL_FRAME.findall(text))
        names.update(_SIDE_AGENT.findall(text))
    from tui_gateway.agent_callbacks import _CHILD_DELTA_EVENTS
    from tui_gateway.change_watcher import _CHANGE_WATCHES

    names.update(_CHANGE_WATCHES)
    names.update(_CHILD_DELTA_EVENTS.values())
    for src in (REPO / "tools").glob("delegate_tool*.py"):
        names.update(_SUBAGENT_RELAY.findall(_read(src)))
    names.discard("subagent.text")  # mirrored into the watch window as message.delta, never emitted
    from tools.registry import _tool_module_candidates

    for src in _tool_module_candidates(REPO / "tools"):
        names.update(_DESKTOP_UI_EMIT.findall(_read(src)))
    names.update(_SETUP_READY.findall(_read(REPO / "hermes_cli" / "free_tier_bootstrap.py")))
    names.update(_FREE_TIER_CHALLENGE.findall(_read(REPO / "hermes_cli" / "anon_challenge.py")))
    names.update(_LOCAL_VIEWER_PUBLISH.findall(_read(REPO / "gateway" / "local_viewer_events.py")))
    return names


def sent_server_requests() -> set[str]:
    names: set[str] = set()
    for src in (REPO / "tui_gateway").glob("*.py"):
        names.update(_LITERAL_REQUEST.findall(_read(src)))
    return names


@pytest.mark.parametrize('preview', [None, '', 'First user preview'])
def test_canonical_session_preview_is_optional_nullable_and_strict(preview):
    from pydantic import ValidationError
    from tui_gateway.contracts.canonical_sessions import CanonicalSessionListRow
    from tui_gateway.contracts.sessions import SessionListRow
    row = dict(id='old', session_id='old', title='', source='tui', started_at=1,
               message_count=0, running=False)
    assert CanonicalSessionListRow.model_validate(row).preview is None
    assert CanonicalSessionListRow.model_validate(dict(row, preview=preview)).preview == preview
    with pytest.raises(ValidationError):
        CanonicalSessionListRow.model_validate(dict(row, preview=42))
    legacy = dict(id='legacy', source='tui', started_at=1, message_count=0)
    assert SessionListRow.model_validate(legacy).preview == ''
    if preview is None:
        with pytest.raises(ValidationError):
            SessionListRow.model_validate(dict(legacy, preview=None))
    else:
        assert SessionListRow.model_validate(dict(legacy, preview=preview)).preview == preview


@pytest.mark.parametrize('model_name', ['canonical', 'legacy'])
def test_session_page_contract_bounds_and_old_defaults(model_name):
    from pydantic import ValidationError
    from tui_gateway.contracts.canonical_sessions import CanonicalListParams, CanonicalListResult
    from tui_gateway.contracts.sessions import SessionListParams, SessionListResult
    params, result = ((CanonicalListParams, CanonicalListResult) if model_name == 'canonical'
                      else (SessionListParams, SessionListResult))
    assert params().offset == 0
    assert params(offset=200, limit=50).offset == 200
    old = result.model_validate({'sessions': [], **({'scope': 'live'} if model_name == 'canonical' else {})})
    assert (old.offset, old.limit, old.has_more) == (0, 200, False)
    for field, values in [('offset', [-1, True, False, 1.5, '1', None, 9007199254740992]),
                          ('limit', [0, 201, True, False, '50', 1.5])]:
        for value in values:
            with pytest.raises(ValidationError):
                params.model_validate({field: value})
    if model_name == 'canonical':
        with pytest.raises(ValidationError):
            params(limit=None)
    else:
        assert params(limit=None).limit is None


def test_catalog_covers_the_whole_wire():
    """Every registered method, every emitted event and every sent server request has a contract,
    and no contract is orphaned (a deleted handler must take its contract with it)."""
    from tui_gateway import server
    from tui_gateway.contracts import registry

    registry.assert_complete(server._methods, emitted_event_names(), sent_server_requests())
