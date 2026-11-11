"""Delegated children size workspace context files to their *own* routed window (#108891).

Integration coverage for the review on #108891: the child's model can differ from the parent's when a
delegation backend (``delegation.backend``, e.g. the advanced-delegation ``model_map``) routes a task
to another model, so the child's context-file budget must follow the routed child's window -- not the
parent's, and not the flat 20,000-char floor.

What is real: the delegation provider contract and registry, ``resolve_task_routes`` /
``_build_children`` / ``_build_child_agent``, credential resolution, ``AIAgent`` and its
``ContextCompressor`` construction, ``agent.model_metadata`` window resolution, and
``build_context_files_prompt`` truncation. What is faked: only the network. Each test serves an
OpenAI-compatible ``/v1/models`` endpoint from a loopback HTTP server (the tests/AGENTS.md pattern);
it reports ``context_length`` per model id, so a named model resolves to 128K or 1M through the real
resolver. A model the endpoint does not list stays unresolved by metadata and lands on the resolver's
own fallback. No LLM request is made: children are built, never run.

Trees without the delegation-backend mechanism (no ``tools.delegate_tool_routes``) are driven through
the only per-child route they have, ``delegation.model``/``base_url`` credentials into
``_build_children``; backend-only cases skip there.
"""

from __future__ import annotations

import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import hermes_yaml as yaml
import pytest

API_KEY = "test-key"
SMALL, LARGE, UNLISTED = "atlas-128k", "zenith-1m", "mystery-ctx"
WINDOWS = {SMALL: 128_000, LARGE: 1_000_000}

FILE_CHARS = 300_000
MIDDLE_AT = 150_000  # kept only when the head slice (70% of the cap) reaches past it, i.e. cap > 214,285
HEAD, MIDDLE, TAIL = "HEAD_MARKER_7f3a", "MIDDLE_MARKER_7f3a", "TAIL_MARKER_7f3a"
_TRUNCATED_RE = re.compile(r"\[\.\.\.truncated (\S*AGENTS\.md): kept (\d+)\+(\d+) of (\d+) chars")

try:  # current core: per-task routes from the active delegation provider
    from agent.delegation_provider import DelegationProvider
    from tools.delegate_tool_routes import resolve_task_routes
except ImportError:  # pre-backend core: one credential bundle for every child
    DelegationProvider, resolve_task_routes = object, None

needs_backends = pytest.mark.skipif(resolve_task_routes is None, reason="tree predates delegation backends")


# ── network stand-in ─────────────────────────────────────────────────────────


class _ModelsEndpoint:
    """Loopback OpenAI-compatible endpoint: ``GET /v1/models`` lists ``windows``; everything else 404s
    (so the resolver's llama.cpp / Ollama / LM Studio probes miss quickly). ``require_key`` answers 401 to
    a ``/models`` request without ``Authorization: Bearer <key>``, like an authenticated gateway."""

    def __init__(self, windows: dict[str, int], require_key: str | None = None):
        self.requests: list[tuple[str, str]] = []
        endpoint = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _reply(self, status, body=None):
                data = json.dumps(body if body is not None else {"error": "not found"}).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                auth = self.headers.get("Authorization", "")
                endpoint.requests.append((self.path, auth))
                if self.path.rstrip("/") not in ("/v1/models", "/models"):
                    return self._reply(404)
                if require_key and auth != f"Bearer {require_key}":
                    return self._reply(401, {"error": "unauthorized"})
                data = [{"id": m, "object": "model", "owned_by": "test", "context_length": n}
                        for m, n in windows.items()]
                self._reply(200, {"object": "list", "data": data})

            def do_POST(self):  # metadata probes only (e.g. Ollama /api/show); children are never run
                endpoint.requests.append((self.path, "POST"))
                self._reply(404)

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.base_url = f"http://127.0.0.1:{self._server.server_address[1]}/v1"
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def close(self):
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)


@pytest.fixture
def endpoints():
    started: list[_ModelsEndpoint] = []

    def start(windows=None, require_key=None):
        started.append(_ModelsEndpoint(WINDOWS if windows is None else windows, require_key))
        return started[-1]

    yield start
    for ep in started:
        ep.close()


# ── test-local delegation backend (shape of advanced-delegation's model_map) ─


class _ModelMap(DelegationProvider):
    """``delegation.backend: model_map``: a task's ``option`` picks a route from
    ``delegation.backends.model_map.options``; no option = the parent's route."""

    name = "model_map"

    def task_parameters(self, config):
        return {"option": {"type": "string", "enum": list(self.settings(config).get("options") or {})}}

    def route_task(self, task, config):
        option = task.get("option")
        return {**self.settings(config)["options"][option], "label": option} if option else {}


@pytest.fixture
def model_map_backend():
    if resolve_task_routes is None:
        yield None
        return
    from agent.delegation_registry import _registry

    provider = _ModelMap()
    previous = _registry.get_provider(provider.name)
    _registry.register(provider)
    yield provider
    _registry.restore_registration(provider.name, provider, previous)


# ── fixtures: home, workspace, parent ────────────────────────────────────────


def _route(model, base_url):
    return {"model": model, "provider": "custom", "base_url": base_url, "api_key": API_KEY}


def _write_config(home, parent_model, parent_base_url, options, **extra):
    """config.yaml with NO ``context_file_max_chars`` and NO ``model.context_length``: every window comes
    from metadata resolution, every cap from the dynamic window-scaled budget."""
    home.mkdir(parents=True, exist_ok=True)
    config = {
        "model": {"default": parent_model, "provider": "custom", "base_url": parent_base_url},
        "delegation": {"backend": "model_map", "backends": {"model_map": {"options": options}}},
        **extra,
    }
    (home / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    return config


def _workspace(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / ".git").mkdir()
    line = "Workspace rule filler: keep changes small and tested.\n"
    middle = f"\n{MIDDLE}\n"
    head = HEAD + "\n" + line * ((MIDDLE_AT - len(HEAD) - 1) // len(line))
    head += "x" * (MIDDLE_AT - len(head))
    tail_len = FILE_CHARS - MIDDLE_AT - len(middle)
    tail = line * ((tail_len - len(TAIL) - 1) // len(line))
    tail += "y" * (tail_len - len(tail) - len(TAIL) - 1) + "\n" + TAIL
    contents = head + middle + tail
    assert len(contents) == FILE_CHARS and contents.index(MIDDLE) == MIDDLE_AT + 1
    (workspace / "AGENTS.md").write_text(contents, encoding="utf-8")
    return workspace


def _parent(workspace, model, base_url):
    from run_agent import AIAgent

    parent = AIAgent(model=model, provider="custom", api_key=API_KEY, base_url=base_url,
                     api_mode="chat_completions", platform="cli", quiet_mode=True, skip_context_files=True,
                     skip_memory=True, save_trajectories=False, enabled_toolsets=["file"])
    parent.cwd = str(workspace)
    return parent


def _build_child(parent, config, task):
    """One child through the delegate path's own builder: backend routes when the tree has them, else
    the ``delegation.*`` credential override (the route this pre-backend tree offers)."""
    from tools import delegate_tool as dt

    if resolve_task_routes is not None:
        routes, error = resolve_task_routes([task], config["delegation"], parent)
        assert error is None and routes[0].error is None, (error, routes and routes[0].error)
        children, error = dt._build_children([task], [None], routes, top_role="leaf", max_iterations=4,
                                             parent_agent=parent, live_deleg_id=None, live_writers=[])
    else:
        option = task.get("option")
        route_cfg = dict(config["delegation"]["backends"]["model_map"]["options"][option]) if option else {}
        creds = dt._resolve_delegation_credentials(route_cfg, parent)
        children, error = dt._build_children([task], [None], creds, top_role="leaf", max_iterations=4,
                                             parent_agent=parent, routing_cfg=route_cfg, live_deleg_id=None,
                                             live_writers=[])
    assert error is None, error
    return children[0][2]


def _context_budget(prompt: str) -> dict:
    """The context-file cap the child's prompt was built with, read back from the truncation marker
    (head = int(cap * 0.7), tail = int(cap * 0.2), see agent.prompt_builder._truncate_content)."""
    found = _TRUNCATED_RE.search(prompt)
    assert found, "AGENTS.md (300,000 chars) must be truncated at every budget up to the 500K ceiling"
    head, tail, total = int(found.group(2)), int(found.group(3)), int(found.group(4))
    assert FILE_CHARS <= total < FILE_CHARS + 64  # + the "## AGENTS.md" heading the loader prepends
    cap = round(head / 0.7)
    assert (int(cap * 0.7), int(cap * 0.2)) == (head, tail), (head, tail)
    return {"cap": cap, "kept": head + tail, "head_marker": HEAD in prompt, "middle_marker": MIDDLE in prompt,
            "tail_marker": TAIL in prompt, "prompt_chars": len(prompt)}


def _expected_cap(window):
    from agent.prompt_builder import _dynamic_context_file_max_chars

    return _dynamic_context_file_max_chars(window)


def _report(case, **observed):
    """Observed numbers, one ``H108891 {json}`` line per case (visible with ``pytest -s``)."""
    print("H108891 " + json.dumps({"case": case, **observed}, sort_keys=True))


def _run_case(tmp_path, monkeypatch, endpoints, *, case, parent_model, option, options_for, require_key=None,
              child_endpoint_windows=None):
    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    parent_ep = endpoints(require_key=require_key)
    child_ep = endpoints(child_endpoint_windows, require_key) if child_endpoint_windows else parent_ep
    config = _write_config(home, parent_model, parent_ep.base_url, options_for(child_ep.base_url))
    workspace = _workspace(tmp_path)
    monkeypatch.setenv("TERMINAL_CWD", str(workspace))  # the workspace hint's first source
    parent = _parent(workspace, parent_model, parent_ep.base_url)
    child = None
    try:
        child = _build_child(parent, config, {"goal": "Follow the workspace rules", "option": option})
        parent_window = parent.context_compressor.context_length
        child_window = child.context_compressor.context_length
        budget = _context_budget(child.ephemeral_system_prompt)
        _report(case, parent_model=parent_model, child_model=child.model, parent_window=parent_window,
                child_window=child_window, parent_cap=_expected_cap(parent_window), **budget,
                unauthenticated_models_requests=sum(1 for p, a in parent_ep.requests + (
                    child_ep.requests if child_ep is not parent_ep else []) if p.endswith("/models") and not a),
                post_paths=sorted({p for p, a in parent_ep.requests + child_ep.requests if a == "POST"}))
        assert not any("completions" in p or "responses" in p for p, _ in parent_ep.requests + child_ep.requests)
        return parent, child, parent_window, child_window, budget
    finally:
        # Every value the callers assert on is captured above, so both agents close here.
        if child is not None:
            child.close()
        parent.close()


# ── the three cases the review asks for (+ a same-model control) ─────────────


def _options(base_url):
    return {"small": _route(SMALL, base_url), "large": _route(LARGE, base_url),
            "unlisted": _route(UNLISTED, base_url), "same-id": _route(LARGE, base_url)}


@pytest.mark.parametrize(
    "case, parent_model, option, child_model, child_window, cap, middle_kept",
    [
        pytest.param(case, *rest, id=case) for case, *rest in [
            ("128k_parent_routes_to_1m_child", SMALL, "large", LARGE, 1_000_000, 240_000, True),
            ("1m_parent_routes_to_128k_child", LARGE, "small", SMALL, 128_000, 30_720, False),
            ("1m_parent_default_route_control", LARGE, None, LARGE, 1_000_000, 240_000, True),
        ]
    ],
)
def test_routed_child_context_files_use_child_window(tmp_path, monkeypatch, endpoints, model_map_backend, case,
                                                     parent_model, option, child_model, child_window, cap,
                                                     middle_kept):
    _, child, parent_window, observed_window, budget = _run_case(
        tmp_path, monkeypatch, endpoints, case=case, parent_model=parent_model, option=option,
        options_for=_options)
    assert child.model == child_model
    assert parent_window == WINDOWS[parent_model]
    assert observed_window == child_window  # the stub endpoint drove the child's real compressor
    assert cap == _expected_cap(child_window)
    assert budget["cap"] == cap, f"{case}: child context files capped at {budget['cap']:,}, want {cap:,} ({budget})"
    assert budget["middle_marker"] is middle_kept and budget["head_marker"] and budget["tail_marker"]


def test_unresolved_child_window_uses_the_childs_fallback_not_parent_or_floor(tmp_path, monkeypatch, endpoints,
                                                                             model_map_backend):
    """The endpoint does not list the routed model, so no metadata source knows its window. The child's
    compressor then runs on the resolver's DEFAULT_FALLBACK_CONTEXT (256K); its context files should be
    budgeted on that same window (61,440 chars) -- never the 1M parent's 240,000, and not the 20,000
    floor the pre-construction prompt build falls to."""
    from agent.model_metadata import DEFAULT_FALLBACK_CONTEXT

    _, child, parent_window, child_window, budget = _run_case(
        tmp_path, monkeypatch, endpoints, case="1m_parent_routes_to_unresolved_child", parent_model=LARGE,
        option="unlisted", options_for=_options)
    assert child.model == UNLISTED and parent_window == 1_000_000
    assert child_window == DEFAULT_FALLBACK_CONTEXT == 256_000
    assert budget["cap"] != _expected_cap(parent_window), "child inherited the 1M parent's budget"
    assert budget["cap"] == _expected_cap(child_window) == 61_440, budget
    assert not budget["middle_marker"]


# ── route shapes that separate "resolve before construction" from "read the built child" ──


@needs_backends
def test_auth_gated_models_endpoint(tmp_path, monkeypatch, endpoints, model_map_backend):
    """The routed endpoint answers /models only with the route's key (an authenticated gateway). The child's
    compressor resolves with that key; a pre-construction lookup without it sees 401 and must not win."""
    _, _child, parent_window, child_window, budget = _run_case(
        tmp_path, monkeypatch, endpoints, case="auth_gated_128k_parent_to_1m_child", parent_model=SMALL,
        option="large", options_for=_options, require_key=API_KEY)
    assert (parent_window, child_window) == (128_000, 1_000_000)
    assert budget["cap"] == 240_000, budget


@needs_backends
def test_same_model_id_on_another_endpoint(tmp_path, monkeypatch, endpoints, model_map_backend):
    """The route keeps the parent's model id but sends it to another endpoint that serves it with a 128K
    window (e.g. a local server running the same weights with a shorter context)."""
    _, child, parent_window, child_window, budget = _run_case(
        tmp_path, monkeypatch, endpoints, case="same_model_id_1m_parent_to_128k_endpoint", parent_model=LARGE,
        option="same-id", options_for=_options, child_endpoint_windows={LARGE: 128_000, SMALL: 128_000})
    assert child.model == LARGE
    assert (parent_window, child_window) == (1_000_000, 128_000)
    assert budget["cap"] == 30_720, budget


@needs_backends
def test_profile_route_window_comes_from_the_profiles_config(tmp_path, monkeypatch, endpoints, model_map_backend):
    """A profile route runs the child as that profile, with that profile's config. Here only the profile's
    config.yaml knows the routed model's window (a ``model_overrides`` entry); the shared endpoint does not
    list it. The child's compressor resolves 1M inside the profile scope; the budget must match it."""
    home = tmp_path / "os-home" / ".hermes"
    monkeypatch.setenv("HOME", str(home.parent))
    monkeypatch.setenv("HERMES_HOME", str(home))
    ep = endpoints()
    config = _write_config(home, LARGE, ep.base_url, {"worker": {"profile": "worker"}})
    profile_home = home / "profiles" / "worker"
    profile_home.mkdir(parents=True)
    (profile_home / "config.yaml").write_text(yaml.safe_dump({
        "model": {"default": UNLISTED, "provider": "custom", "base_url": ep.base_url, "api_key": API_KEY},
        "model_overrides": {"custom": {UNLISTED: {"context_window": 1_000_000}}},
    }), encoding="utf-8")
    workspace = _workspace(tmp_path)
    monkeypatch.setenv("TERMINAL_CWD", str(workspace))
    parent = _parent(workspace, SMALL, ep.base_url)
    child = None
    try:
        child = _build_child(parent, config, {"goal": "Follow the workspace rules", "option": "worker"})
        parent_window = parent.context_compressor.context_length
        child_window = child.context_compressor._resolved_context_length  # as resolved in the profile scope
        budget = _context_budget(child.ephemeral_system_prompt)
        _report("profile_route_128k_parent_to_1m_profile_model", parent_model=SMALL, child_model=child.model,
                parent_window=parent_window, child_window=child_window, parent_cap=_expected_cap(parent_window),
                unauthenticated_models_requests=sum(1 for p, a in ep.requests if p.endswith("/models") and not a),
                post_paths=sorted({p for p, a in ep.requests if a == "POST"}), **budget)
    finally:
        if child is not None:
            child.close()
        parent.close()
    assert child.model == UNLISTED and parent_window == 128_000
    assert child_window == 1_000_000
    assert budget["cap"] == 240_000, budget
