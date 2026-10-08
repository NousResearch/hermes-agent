"""Real-import E2E for delegation model profiles (T8).

AGENTS.md: anything touching resolution chains or config propagation must exercise the
real path with real imports against a temp HERMES_HOME — unit mocks hide integration bugs.

These tests write an ACTUAL config.yaml into the sandboxed HERMES_HOME and drive the real
runtime chain: config.yaml on disk → hermes_cli.config.load_config_readonly (via
tools.delegate_tool_config._load_config) → the profile branch in
_resolve_delegation_credentials → agent.delegation_model_routing.resolve_profile_route →
hermes_cli.runtime_provider.resolve_runtime_provider → per-task routes in
_resolve_task_credentials → _build_child_agent construction kwargs.

The ONLY mocked seam is the AIAgent construction boundary (run_agent.AIAgent), captured to
inspect the kwargs a child would be built with. Config loading, profile parsing, credential
resolution and the schema gate all run for real against the on-disk file.

Cache note: hermes_cli.config memoises merged config in _LOAD_CONFIG_CACHE keyed on the
config file's (mtime_ns, size). Tests that rewrite config.yaml clear that cache explicitly
(the public-enough module-level dict; same pattern as tests/tools/test_approval_config_readonly.py)
so a rewrite is always observed regardless of filesystem mtime granularity.
"""

import threading
from unittest.mock import MagicMock, patch

import pytest

import hermes_cli.config as hc


SMALL_MODEL = "prof/small-e2e"
BIG_MODEL = "prof/big-e2e"
BIG_FALLBACK = {"provider": "openrouter", "model": "prof/backup-e2e"}

CONFIG_GATE_ON = f"""
model:
  default: test-model
delegation:
  agent_routing: true
  default_profile: small
  profiles:
    small:
      provider: openrouter
      model: {SMALL_MODEL}
      fallback: []
    big:
      provider: openrouter
      model: {BIG_MODEL}
      fallback:
        - provider: {BIG_FALLBACK["provider"]}
          model: {BIG_FALLBACK["model"]}
"""

CONFIG_GATE_OFF = CONFIG_GATE_ON.replace("agent_routing: true", "agent_routing: false")

# 'temperature' is not a profile key (_PROFILE_KEYS is a closed set) — the config-check
# lane must report it and the spawn path must refuse it cleanly.
CONFIG_MALFORMED = """
model:
  default: test-model
delegation:
  agent_routing: true
  profiles:
    small:
      provider: openrouter
      model: prof/small-e2e
      temperature: 0.2
"""


def _write_config(home, text):
    """Write the REAL config.yaml and drop the merged-config memo so the next
    load_config_readonly() re-reads the file even on coarse-mtime filesystems."""
    (home / "config.yaml").write_text(text)
    hc._LOAD_CONFIG_CACHE.clear()


@pytest.fixture
def real_home(tmp_path, monkeypatch):
    """A temp HERMES_HOME holding a real config.yaml the production loaders read."""
    home = tmp_path / "hermes-e2e-home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    # _load_config()'s readonly branch is skipped entirely under this flag — the whole
    # point of the suite is exercising it, so make sure it is off.
    monkeypatch.delenv("HERMES_IGNORE_USER_CONFIG", raising=False)
    # Real credential rung: the openrouter provider resolves from this env var through
    # the genuine resolve_runtime_provider ladder (no network involved for api_key rungs).
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-e2e-test-key")
    _write_config(home, CONFIG_GATE_ON)
    yield home
    hc._LOAD_CONFIG_CACHE.clear()


def _make_parent(depth=0):
    parent = MagicMock()
    parent.base_url = "https://openrouter.ai/api/v1"
    parent.api_key = "parent-key"
    parent.provider = "openrouter"
    parent.api_mode = "chat_completions"
    parent.model = "anthropic/claude-sonnet-4"
    parent.platform = "cli"
    parent.session_id = "parent-e2e"
    parent.providers_allowed = None
    parent.providers_ignored = None
    parent.providers_order = None
    parent.provider_sort = None
    parent._session_db = None
    parent._delegate_depth = depth
    parent._active_children = []
    parent._active_children_lock = threading.Lock()
    parent._print_fn = None
    parent.tool_progress_callback = None
    parent.thinking_callback = None
    parent._fallback_chain = [{"provider": "openrouter", "model": "parent-fallback"}]
    return parent


def _make_child():
    child = MagicMock()
    child.run_conversation.return_value = {
        "final_response": "done", "completed": True, "api_calls": 1, "messages": [],
    }
    child._delegate_saved_tool_names = []
    child._credential_pool = None
    child.session_prompt_tokens = 0
    child.session_completion_tokens = 0
    child.model = "test"
    return child


class _CaptureAIAgent:
    """Patch ONLY run_agent.AIAgent — the construction boundary. Everything upstream
    (config load, profile parse, credential resolution, per-task routing) runs for real."""

    def __init__(self):
        self.built = []
        self._patch = patch("run_agent.AIAgent", side_effect=self._factory)

    def _factory(self, *args, **kwargs):
        self.built.append(kwargs)
        return _make_child()

    def __enter__(self):
        self._patch.start()
        return self

    def __exit__(self, *exc):
        self._patch.stop()
        return False


# ── E2E-1: per-task mixed profiles + fallback isolation ─────────────────────

def test_e2e_batch_mixes_profiles_from_real_config(real_home):
    """Two tasks, two profiles, ONE real config.yaml: the captured construction kwargs
    carry the two different profile models, real openrouter credentials from the env,
    and per-profile fallback isolation (small's [] blocks the parent chain; big's chain
    comes from the file, not the parent)."""
    from tools.delegate_tool import _handle_model_call

    parent = _make_parent()
    with _CaptureAIAgent() as cap:
        result = _handle_model_call(
            {"tasks": [
                {"goal": "Summarize the release notes for QA", "model_profile": "small"},
                {"goal": "Design the storage migration plan", "model_profile": "big"},
            ]},
            parent_agent=parent, background=False,
        )
    assert "error" not in result.lower() or '"success": true' in result.lower(), result
    assert [b["model"] for b in cap.built] == [SMALL_MODEL, BIG_MODEL]
    assert all(b["api_key"] == "sk-or-e2e-test-key" for b in cap.built)
    # Fallback isolation: small pins fallback [] → parent's chain must NOT leak in;
    # big's chain is exactly the on-disk profile's, not the parent's.
    assert not cap.built[0]["fallback_model"]
    assert cap.built[1]["fallback_model"] == [BIG_FALLBACK]
    assert parent._fallback_chain not in ([b["fallback_model"] for b in cap.built],)


# ── E2E-2: default_profile from the real file ────────────────────────────────

def test_e2e_default_profile_applies_from_real_config(real_home):
    from tools.delegate_tool import _handle_model_call

    with _CaptureAIAgent() as cap:
        _handle_model_call(
            {"tasks": [{"goal": "Do the maintenance work carefully"}]},
            parent_agent=_make_parent(), background=False,
        )
    assert len(cap.built) == 1
    assert cap.built[0]["model"] == SMALL_MODEL  # default_profile: small, from disk


# ── E2E-3: gate on/off schema honesty against the on-disk file ───────────────

def test_e2e_gate_flip_rewrites_schema_and_rejects_fabricated_arg(real_home):
    """Gate ON (file as written): the dynamic schema advertises model_profile with the
    real profile names. Flip the FILE to agent_routing: false and re-read through the
    real loader: the schema must not mention model_profile at all, and a fabricated
    model_profile arg is rejected with a clean tool_error before any construction."""
    from tools.delegate_tool import _build_dynamic_schema_overrides, _handle_model_call

    on = _build_dynamic_schema_overrides()["parameters"]["properties"]
    assert on["model_profile"]["enum"] == ["big", "small"]
    assert on["tasks"]["items"]["properties"]["model_profile"]["enum"] == ["big", "small"]

    _write_config(real_home, CONFIG_GATE_OFF)

    off = _build_dynamic_schema_overrides()["parameters"]["properties"]
    assert "model_profile" not in off
    assert "model_profile" not in off["tasks"]["items"]["properties"]

    with _CaptureAIAgent() as cap:
        result = _handle_model_call(
            {"tasks": [{"goal": "Summarize the release notes", "model_profile": "small"}]},
            parent_agent=_make_parent(), background=False,
        )
    assert "model_profile is not enabled" in result
    assert cap.built == []  # rejected BEFORE any child construction


# ── E2E-4: malformed profile — config-check lane + clean spawn refusal ────────

def test_e2e_malformed_profile_reported_by_config_check_and_spawn(real_home):
    _write_config(real_home, CONFIG_MALFORMED)

    # hermes config check lane: the real validator over the real on-disk config.
    from hermes_cli.config import load_config_readonly, validate_config_structure
    issues = validate_config_structure(load_config_readonly())
    messages = [str(getattr(i, "message", i)) for i in issues]
    assert any("temperature" in m and "small" in m for m in messages), messages

    # Spawn path: an unparseable profiles section hides the gate entirely
    # (_routable_profile_names → None on parse failure), so a model-supplied profile is
    # refused with the clean gate tool_error BEFORE any resolution or construction —
    # the malformed file never reaches a child spawn.
    from tools.delegate_tool import _handle_model_call
    with _CaptureAIAgent() as cap:
        result = _handle_model_call(
            {"tasks": [{"goal": "Summarize the notes", "model_profile": "small"}]},
            parent_agent=_make_parent(), background=False,
        )
    assert "model_profile is not enabled" in result
    assert '"success": true' not in result.lower()
    assert cap.built == []


# ── E2E-5: lifecycle parity through the same on-disk config ──────────────────

def test_e2e_lifecycle_launch_resolves_same_route_as_delegate_task(real_home, monkeypatch):
    """SubagentLaunchRequest(model_profile='small') resolves through the SAME real
    config.yaml + loader + resolver to the same model as E2E-1's small task."""
    from agent.subagent_lifecycle import SubagentLaunchRequest, SubagentLifecycleService

    # Keep the executor-side run inert; construction (the seam under test) is synchronous.
    monkeypatch.setattr(
        "tools.delegate_tool._run_child_lifecycle",
        lambda *a, **k: {"status": "completed", "summary": "ok", "api_calls": 1, "duration_seconds": 0.0},
    )
    parent = _make_parent()
    service = SubagentLifecycleService(lambda: parent)
    with _CaptureAIAgent() as cap:
        handle = service.launch(SubagentLaunchRequest(goal="parity check", model_profile="small"))
    assert len(cap.built) == 1
    assert cap.built[0]["model"] == SMALL_MODEL  # same route as E2E-1's small task
    assert cap.built[0]["api_key"] == "sk-or-e2e-test-key"
    assert not cap.built[0]["fallback_model"]  # small's fallback: [] honored here too
    assert handle.requested_profile == "small"  # spawn-time provenance must be stamped


# ── E2E-6: unresolvable profile provider → clean tool_error, no child spawned ─

CONFIG_ANTHROPIC_PROFILE = """
model:
  default: test-model
delegation:
  agent_routing: true
  profiles:
    small:
      provider: anthropic
      model: claude-haiku-e2e
"""


def test_e2e_unresolvable_provider_credentials_yield_clean_tool_error(real_home, monkeypatch):
    """A profile pointing at a provider with no credentials must surface as a clean
    tool_error string from _handle_model_call — never an escaped AuthError/RuntimeError —
    exactly like the legacy delegation.provider branch."""
    _write_config(real_home, CONFIG_ANTHROPIC_PROFILE)
    for var in ("ANTHROPIC_API_KEY", "CLAUDE_API_KEY"):
        monkeypatch.delenv(var, raising=False)

    from tools.delegate_tool import _handle_model_call
    with _CaptureAIAgent() as cap:
        result = _handle_model_call(
            {"tasks": [{"goal": "x", "model_profile": "small"}]},
            parent_agent=_make_parent(), background=False,
        )
    assert isinstance(result, str)
    assert "small" in result
    assert '"success": true' not in result.lower()
    assert cap.built == []  # refused before any child construction
