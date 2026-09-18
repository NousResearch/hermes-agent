"""Runtime catalog and generated-contract invariants.

``server._event_frame`` proves a payload's SHAPE once it has found the event's contract; it cannot prove
that every emitted event NAME has one (an undeclared or mistyped literal is a ``KeyError`` on the first
execution of that path). The source inventory below keeps that closure mechanical.

The producer side is guarded at RUNTIME, never by a source scan (root ``AGENTS.md``: a test must not
read production source). ``test_every_registered_event_refuses_a_non_model_payload`` sweeps the whole
registry through the real frame builder, and the producer regressions
(``test_typed_event_producers_settle.py``, ``test_pet_generate_rpc.py``) drive the real handlers so a
dict-handed payload — which ``_pet_emit`` swallows at debug level — fails instead of passing silently.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
from jsonschema import Draft7Validator

REPO = Path(__file__).resolve().parents[3]
GEN = REPO / "scripts" / "gen_gateway_contracts.py"
META_SCHEMA = REPO / "tests" / "tui_gateway" / "contracts" / "fixtures" / "openrpc-meta-schema.json"


@pytest.fixture(scope="module")
def gen():
    spec = importlib.util.spec_from_file_location("gen_gateway_contracts_generated", GEN)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _generator(*args: str) -> subprocess.CompletedProcess[str]:
    environment = {**os.environ, "PYTHONPATH": str(REPO)}
    return subprocess.run(
        [sys.executable, str(GEN), *args],  # the runner's interpreter, whatever venv layout selected it
        cwd=REPO,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )


# The emitter inventory: names must come from CODE the gateway runs, never from the contract tables.
_EMIT_HELPERS = ("_emit", "_broadcast_global_event", "_voice_emit", "_pet_emit", "_emit_tool_lifecycle")
_LITERAL_EMIT = re.compile(r"\b(?:%s)\(\s*\"([a-z_][a-z0-9_.]*)\"" % "|".join(_EMIT_HELPERS))
_REQUEST_HELPERS = ("server_requests\\.send", "server_requests\\.send_async", "_ask", "_read_block")
_LITERAL_REQUEST = re.compile(r"\b(?:%s)\(\s*\"([a-z_][a-z0-9_.]*)\"" % "|".join(_REQUEST_HELPERS))
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
    names.update(_BROKER_FRAME.findall(_read(REPO / "gateway" / "browser_control_broker.py")))
    names.update(_SETUP_READY.findall(_read(REPO / "hermes_cli" / "free_tier_bootstrap.py")))
    return names


def sent_server_requests() -> set[str]:
    names: set[str] = set()
    for src in (REPO / "tui_gateway").glob("*.py"):
        names.update(_LITERAL_REQUEST.findall(_read(src)))
    return names


def test_generated_files_are_current(gen):
    """Both committed artefacts equal an in-memory regeneration (byte-for-byte): the TypeScript and
    OpenRPC the clients compile against never drift from the models with CI green."""
    stale = [path.relative_to(REPO) for path, text in gen.render_all().items()
             if (path.read_text(encoding="utf-8") if path.exists() else None) != text]
    assert not stale, f"stale generated contract files {stale}: run scripts/gen_gateway_contracts.py"


def test_catalog_covers_the_whole_wire():
    """Every registered method, every emitted event name and every sent server request has a contract,
    and no contract is orphaned (a deleted handler or emitter must take its contract with it)."""
    from tui_gateway import server
    from tui_gateway.contracts import registry

    registry.assert_complete(server._methods, emitted_event_names(), sent_server_requests())


def _meta_schema_registry():
    """Resolve the meta-schema's EXTERNAL refs from the vendored stub, never the network.

    The vendored meta-schema references ``https://meta.json-schema.tools``; ``jsonschema`` answers
    that with an automatic remote retrieval (deprecated, and dependent on the runner's egress —
    this check flaked in CI for exactly that reason). The stub carries the only shape the
    meta-schema dereferences, so the validation stays hermetic and unchanged for our document.
    """
    import referencing
    from referencing.jsonschema import DRAFT7

    stub = json.loads((META_SCHEMA.parent / "meta-json-schema-tools.json").read_text(encoding="utf-8"))

    return referencing.Registry().with_resource(
        "https://meta.json-schema.tools", referencing.Resource.from_contents(stub, default_specification=DRAFT7))


def test_openrpc_validates_against_vendored_meta_schema(gen):
    schema = json.loads(META_SCHEMA.read_text(encoding="utf-8"))
    document = json.loads(gen.render_openrpc())

    assert list(Draft7Validator(schema, registry=_meta_schema_registry()).iter_errors(document)) == []


def test_rendering_is_deterministic(gen):
    first = tuple(gen.render_all(Path("/tmp/pr3-determinism-one")).values())
    second = tuple(gen.render_all(Path("/tmp/pr3-determinism-two")).values())

    assert first == second


def test_check_detects_stale_output_in_an_explicit_output_directory(tmp_path):
    result = _generator("--out-dir", str(tmp_path))
    assert result.returncode == 0, result.stderr

    fresh = _generator("--check", "--out-dir", str(tmp_path))
    assert fresh.returncode == 0, fresh.stderr

    generated = tmp_path / "gateway-contract.generated.ts"
    generated.write_text(generated.read_text(encoding="utf-8") + " ", encoding="utf-8")
    stale = _generator("--check", "--out-dir", str(tmp_path))
    assert stale.returncode == 1
    assert "gateway-contract.generated.ts" in stale.stderr


def test_every_registered_event_refuses_a_non_model_payload():
    """Runtime seam over the WHOLE registry: a payload-carrying event accepts its registered model and
    refuses anything else (a dict, the crossing class that lands as TypeError after state changed).

    ``model_construct`` builds an instance without validation, so the sweep covers every event
    regardless of which fields are required.
    """
    from tui_gateway import server
    from tui_gateway.contracts import registry

    assert registry.EVENTS, "the event registry is empty"
    for name, contract in registry.EVENTS.items():
        with pytest.raises(TypeError):
            server._event_frame(name, "sid", {})  # a dict is never a payload, for any event
        if contract.payload is None:
            continue
        frame = server._event_frame(name, "sid", contract.payload.model_construct())
        assert frame["params"]["type"] == name
        assert isinstance(frame["params"]["payload"], dict)
