"""``<subsystem>.write_approval_scope``: gate only background writes.

``write_approval`` stays the boolean on/off gate. ``write_approval_scope: background`` exempts the
user's own turns (write origin ``assistant_tool``, or the unbound default ``foreground``) from an
enabled gate. Every forked writer keeps being gated: the post-turn review fork, ``/refine`` and the
curator's consolidation fork (all ``background_review``), plugin session forks, side questions, and
any origin added later. Unset or unknown scopes keep today's behavior (``all``).
"""

import json

import pytest

SKILL = "---\nname: scoped-skill\ndescription: Use when testing scoped write approval.\n---\n\nBody.\n"


@pytest.fixture
def home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with a skills dir that skill_manage writes to."""
    from agent import skill_utils
    from tools import skill_manager_tool

    hermes_home = tmp_path / ".hermes"
    skills_dir = hermes_home / "skills"
    skills_dir.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(skill_manager_tool, "SKILLS_DIR", skills_dir)
    monkeypatch.setattr(skill_utils, "get_all_skills_dirs", lambda: [skills_dir])
    return hermes_home


def _configure(subsystem, *, enabled, scope=None):
    import hermes_cli.config as cfg

    c = cfg.load_config()
    section = c.setdefault(subsystem, {})
    section["write_approval"] = enabled
    if scope is not None:
        section["write_approval_scope"] = scope
    cfg.save_config(c)


def _with_origin(origin, fn):
    from tools.skill_provenance import reset_current_write_origin, set_current_write_origin

    token = set_current_write_origin(origin)
    try:
        return fn()
    finally:
        reset_current_write_origin(token)


def _create_skill():
    from tools.skill_manager_tool import skill_manage

    return json.loads(skill_manage(action="create", name="scoped-skill", content=SKILL))


def _add_memory(content):
    from tools.memory_tool import MemoryStore, memory_tool

    store = MemoryStore()
    store.load_from_disk()
    return store, json.loads(memory_tool("add", "memory", content, store=store))


@pytest.fixture
def prompt_calls():
    """Record (and approve) any inline approval prompt, so a test can assert none was shown."""
    from tools.terminal_tool import set_approval_callback

    calls = []

    def _callback(*args, **kwargs):
        calls.append((args, kwargs))
        return "once"

    set_approval_callback(_callback)
    yield calls
    set_approval_callback(None)


# --- scope resolution ---

@pytest.mark.parametrize(("raw", "narrowed"), [
    (None, False), ("all", False), ("background", True), (" Background ", True),
    ("background_review", True), ("background_only", True), ("bogus", False), ("", False),
    (True, False), (1, False), ([], False),
])
def test_only_a_background_scope_narrows_the_gate(home, raw, narrowed):
    """Unset, unknown and non-string scopes keep today's gate (the safer direction: more gating):
    a foreground skill write is allowed only when the scope really is ``background`` (or one of its
    two accepted spellings: the issue's ``background_review`` and #53412's ``background_only``)."""
    from tools import write_approval as wa

    _configure("skills", enabled=True, scope=raw)
    decision = _with_origin("assistant_tool", lambda: wa.evaluate_gate(wa.SKILLS))
    assert (decision.allow, decision.stage) == ((True, False) if narrowed else (False, True))


def test_scope_is_read_per_subsystem(home):
    from tools import write_approval as wa

    _configure("skills", enabled=True, scope="background")
    _configure("memory", enabled=True)
    assert _with_origin("assistant_tool", lambda: wa.evaluate_gate(wa.SKILLS)).allow is True
    assert _with_origin("background_review", lambda: wa.evaluate_gate(wa.MEMORY)).stage is True
    _configure("memory", enabled=True, scope="background")
    _configure("skills", enabled=True, scope="all")
    assert _with_origin("assistant_tool", lambda: wa.evaluate_gate(wa.SKILLS)).stage is True


# --- skills ---

def test_unset_scope_still_stages_foreground_skill_writes(home):
    from tools import write_approval as wa

    _configure("skills", enabled=True)
    result = _with_origin("assistant_tool", _create_skill)
    assert result.get("staged") is True, result
    assert wa.pending_count(wa.SKILLS) == 1
    assert not (home / "skills" / "scoped-skill").exists()


@pytest.mark.parametrize("origin", ["assistant_tool", "foreground"])
def test_background_scope_applies_the_users_own_skill_writes(home, origin):
    """``assistant_tool`` is every main-agent turn (CLI, gateway, cron, subagents); ``foreground`` is
    the ContextVar default when nothing is bound."""
    from tools import write_approval as wa

    _configure("skills", enabled=True, scope="background")
    result = _with_origin(origin, _create_skill)
    assert result.get("success") is True and not result.get("staged"), result
    assert wa.pending_count(wa.SKILLS) == 0
    assert (home / "skills" / "scoped-skill" / "SKILL.md").exists()


def test_background_scope_stages_background_review_skill_writes(home):
    """Covers the post-turn review fork, /refine AND the curator's consolidation fork: all three
    bind the ``background_review`` write origin."""
    from tools import write_approval as wa

    _configure("skills", enabled=True, scope="background")
    result = _with_origin("background_review", _create_skill)
    assert result.get("staged") is True, result
    (record,) = wa.list_pending(wa.SKILLS)
    assert record["origin"] == "background_review"
    assert not (home / "skills" / "scoped-skill").exists()


@pytest.mark.parametrize("origin", ["assistant_tool", "background_review"])
def test_gate_off_ignores_the_scope(home, origin):
    from tools import write_approval as wa

    _configure("skills", enabled=False, scope="background")
    result = _with_origin(origin, _create_skill)
    assert result.get("success") is True and not result.get("staged"), result
    assert wa.pending_count(wa.SKILLS) == 0


@pytest.mark.parametrize("origin", ["plugin_fork:demo", "side_question", "some_future_fork"])
def test_background_scope_keeps_gating_every_other_fork(home, origin):
    """The exemption is an allowlist of the user's own turns, not a match on ``background_review``:
    a plugin session fork whitelisted for skill_manage, a side question, or an origin added later
    stays gated exactly as with scope ``all``."""
    from tools import write_approval as wa

    _configure("skills", enabled=True, scope="background")
    decision = _with_origin(origin, lambda: wa.evaluate_gate(wa.SKILLS))
    assert (decision.allow, decision.stage) == (False, True)
    _configure("skills", enabled=True, scope="all")
    assert _with_origin(origin, lambda: wa.evaluate_gate(wa.SKILLS)) == decision


# --- memory ---

def test_background_scope_applies_foreground_memory_without_a_prompt(home, prompt_calls):
    from tools import write_approval as wa

    _configure("memory", enabled=True, scope="background")
    store, result = _with_origin("assistant_tool", lambda: _add_memory("foreground fact"))
    assert result.get("success") is True and not result.get("staged"), result
    assert prompt_calls == []
    assert store.memory_entries == ["foreground fact"]
    assert wa.pending_count(wa.MEMORY) == 0


def test_background_scope_stages_background_memory_writes(home, prompt_calls):
    from tools import write_approval as wa

    _configure("memory", enabled=True, scope="background")
    store, result = _with_origin("background_review", lambda: _add_memory("review fact"))
    assert result.get("staged") is True, result
    assert prompt_calls == []
    assert store.memory_entries == []
    assert wa.pending_count(wa.MEMORY) == 1


def test_all_scope_keeps_the_foreground_memory_prompt(home, prompt_calls):
    """Unchanged behavior: with scope ``all`` a foreground memory write still goes through the inline
    prompt (approved here), proving the narrowing above is the scope's doing."""
    _configure("memory", enabled=True, scope="all")
    store, result = _with_origin("assistant_tool", lambda: _add_memory("prompted fact"))
    assert len(prompt_calls) == 1
    assert result.get("success") is True
    assert store.memory_entries == ["prompted fact"]


# --- curator coverage ---

def test_curator_consolidation_fork_binds_the_background_review_origin(tmp_path, monkeypatch):
    """The scope's curator coverage rests on this: the consolidation fork binds the same write
    origin as the post-turn review fork, so its skill writes stage under scope ``background``."""
    import importlib
    from pathlib import Path

    home = tmp_path / ".hermes"
    (home / "skills").mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    from agent import curator
    importlib.reload(curator)
    seen = {}

    class _StubAgent:
        def __init__(self, *args, **kwargs):
            self._memory_write_origin = "assistant_tool"
            self._memory_nudge_interval = self._skill_nudge_interval = 0
            self._session_messages = []

        def run_conversation(self, user_message=None, **kwargs):
            seen["origin"] = self._memory_write_origin
            return {"final_response": "ok"}

        def close(self):
            pass

    cfg = {"model": {"provider": "openai-api", "default": "gpt-4o-mini"}}
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: cfg)
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: cfg)
    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider",
                        lambda **kw: {"provider": "openai-api", "api_key": "k", "base_url": "https://x.test/v1"})
    monkeypatch.setattr("run_agent.AIAgent", _StubAgent)

    assert curator._run_llm_review("review prompt").get("error") is None
    assert seen == {"origin": "background_review"}


# --- generic truthiness readers ---

def test_skills_command_stays_on_messaging_gateways_with_background_scope(home):
    """A boolean gate plus a scope key keeps every truthiness reader right: the gateway shows
    /skills whenever skills.write_approval is on, whatever the scope."""
    from hermes_cli.commands import _resolve_config_gates

    _configure("skills", enabled=True, scope="background")
    assert "skills" in _resolve_config_gates()
