"""Unit tests for per-task Hermes profile delegation routing.

Tests:
1. _detect_task_profile:
   - Explicit 'profile' key
   - Goal prefix '@<profile>:'
   - Non-profile @-mentions ignored (fallback to None)
   - Plain goals without @-mention (fallback to None)
   - Case-insensitive profile name matching
2. _resolve_profile_task_credentials:
   - Successfully reads model, provider, base_url, api_key from profile config.yaml
   - Returns None for non-existent profile
   - Gracefully handles corrupt / unreadable config.yaml without raising
3. Schema advertisement:
   - DELEGATE_TASK_SCHEMA exposes 'profile' on tasks items
4. Isolation:
   - Profile-scoped children activate memory and context files
"""

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from tools.delegate_tool import (
    DELEGATE_TASK_SCHEMA,
    ProfileRealization,
    _extract_task_profile_target,
    _resolve_delegated_profile_authority,
)


# ── Profile Extraction Tests ────────────────────────────────────────────────


def test_extract_task_profile_explicit():
    """Explicit 'profile' key takes strict precedence."""
    assert _extract_task_profile_target({"profile": "piglet", "goal": "check logs"}) == "piglet"
    assert _extract_task_profile_target({"profile": "TIGGER", "goal": "run tests"}) == "tigger"
    assert _extract_task_profile_target({"profile": "unknown", "goal": "run tests"}) == "unknown"


def test_extract_task_profile_explicit_precedence():
    """Explicit 'profile' takes strict precedence over goal prefix."""
    # When both are present and disagree, explicit 'profile' wins
    assert _extract_task_profile_target({"profile": "piglet", "goal": "@tigger: run tests"}) == "piglet"
    # Even if invalid/unknown, explicit profile overrides the goal prefix
    assert _extract_task_profile_target({"profile": "unknown", "goal": "@tigger: run tests"}) == "unknown"


def test_extract_task_profile_goal_prefix():
    """Goal prefix '@<profile>:' extracts the profile name if no explicit profile is provided."""
    assert _extract_task_profile_target({"goal": "@piglet: review the changes"}) == "piglet"
    assert _extract_task_profile_target({"goal": "@Eeyore: check error trace"}) == "eeyore"
    assert _extract_task_profile_target({"goal": "   @tigger:   write unit tests"}) == "tigger"


def test_extract_task_profile_plain_goal():
    """Goals without any @-mention return None."""
    assert _extract_task_profile_target({"goal": "clean up temporary files"}) is None
    assert _extract_task_profile_target({}) is None


# ── Credential Resolution & Authorization Tests ─────────────────────────────


def test_resolve_delegated_profile_authority_success(tmp_path, monkeypatch):
    """Succeeds when profile exists, is allowed, and config is valid."""
    profile_dir = tmp_path / "profiles" / "mock_persona"
    profile_dir.mkdir(parents=True)
    cfg_file = profile_dir / "config.yaml"
    cfg_file.write_text(
        """
model:
  default: test-model-v1
  provider: mock_provider
providers:
  mock_provider:
    base_url: http://10.0.0.1:11434/v1
    api_key: secret-key-abc
    api_mode: chat
    request_overrides:
      extra_body:
        temperature: 0.2
"""
    )

    monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda name: name == "mock_persona")
    monkeypatch.setattr("hermes_cli.profiles.get_profile_dir", lambda name: profile_dir)

    delegation_cfg = {"allowed_profiles": ["mock_persona"]}
    creds, err = _resolve_delegated_profile_authority("mock_persona", delegation_cfg, parent_agent=None)
    
    assert err is None
    assert creds is not None
    assert creds["model"] == "test-model-v1"
    assert creds["provider"] == "mock_provider"
    assert creds["base_url"] == "http://10.0.0.1:11434/v1"
    assert creds["api_key"] == "secret-key-abc"
    assert creds["api_mode"] == "chat"
    assert creds["profile"] == "mock_persona"
    assert creds["profile_dir"] == profile_dir
    assert creds["request_overrides"] == {"extra_body": {"temperature": 0.2}}


def test_resolve_delegated_profile_authority_refuses_unallowed(monkeypatch):
    """Refuses pre-effect if profile is not in allowed_profiles."""
    monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda name: True)
    accessed_dirs = []
    monkeypatch.setattr("hermes_cli.profiles.get_profile_dir", lambda name: accessed_dirs.append(name))

    delegation_cfg = {"allowed_profiles": ["tigger"]}
    creds, err = _resolve_delegated_profile_authority("mock_persona", delegation_cfg, parent_agent=None)
    
    assert creds is None
    assert err == "Delegation refused: profile 'mock_persona' is not in delegation.allowed_profiles"
    # Refusal must occur before target profile config/home is accessed
    assert accessed_dirs == []


def test_resolve_delegated_profile_authority_refuses_empty_allowlist(monkeypatch):
    """Refuses pre-effect if allowed_profiles is not configured."""
    monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda name: True)

    creds, err = _resolve_delegated_profile_authority("mock_persona", {}, parent_agent=None)
    
    assert creds is None
    assert err == "Delegation refused: profile 'mock_persona' requested, but delegation.allowed_profiles is not configured (no profiles are granted for delegation)"


def test_resolve_delegated_profile_authority_refuses_nonexistent(monkeypatch):
    """Refuses pre-effect if profile does not exist."""
    monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda name: False)

    delegation_cfg = {"allowed_profiles": ["nonexistent"]}
    creds, err = _resolve_delegated_profile_authority("nonexistent", delegation_cfg, parent_agent=None)
    
    assert creds is None
    assert err == "Delegation refused: target profile 'nonexistent' does not exist"


def test_resolve_delegated_profile_authority_refuses_corrupt_yaml(tmp_path, monkeypatch):
    """Refuses pre-effect on unreadable or invalid YAML config."""
    profile_dir = tmp_path / "profiles" / "broken"
    profile_dir.mkdir(parents=True)
    cfg_file = profile_dir / "config.yaml"
    cfg_file.write_text("[\ninvalid yaml\n:")

    monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda name: name == "broken")
    monkeypatch.setattr("hermes_cli.profiles.get_profile_dir", lambda name: profile_dir)

    delegation_cfg = {"allowed_profiles": ["broken"]}
    creds, err = _resolve_delegated_profile_authority("broken", delegation_cfg, parent_agent=None)
    
    assert creds is None
    assert err == "Delegation refused: target profile 'broken' config.yaml is invalid YAML"


def test_resolve_delegated_profile_authority_refuses_missing_config(tmp_path, monkeypatch):
    """Refuses pre-effect if config.yaml is missing."""
    profile_dir = tmp_path / "profiles" / "missing_cfg"
    profile_dir.mkdir(parents=True)

    monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda name: name == "missing_cfg")
    monkeypatch.setattr("hermes_cli.profiles.get_profile_dir", lambda name: profile_dir)

    delegation_cfg = {"allowed_profiles": ["missing_cfg"]}
    creds, err = _resolve_delegated_profile_authority("missing_cfg", delegation_cfg, parent_agent=None)
    
    assert creds is None
    assert err == "Delegation refused: target profile 'missing_cfg' config.yaml is missing"


def test_resolve_delegated_profile_authority_refuses_model_only(tmp_path, monkeypatch):
    """Refuses pre-effect if profile specifies a model but no provider."""
    profile_dir = tmp_path / "profiles" / "model_only"
    profile_dir.mkdir(parents=True)
    cfg_file = profile_dir / "config.yaml"
    cfg_file.write_text("model: custom-worker\n")

    monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda name: name == "model_only")
    monkeypatch.setattr("hermes_cli.profiles.get_profile_dir", lambda name: profile_dir)

    delegation_cfg = {"allowed_profiles": ["model_only"]}
    creds, err = _resolve_delegated_profile_authority("model_only", delegation_cfg, parent_agent=None)
    
    assert creds is None
    assert err == "Delegation refused: target profile 'model_only' specifies no provider (cannot ambiently inherit parent provider)"


def test_resolve_delegated_profile_authority_refuses_missing_api_key(tmp_path, monkeypatch):
    """Refuses pre-effect if provider requires an API key and none is configured."""
    profile_dir = tmp_path / "profiles" / "no_key"
    profile_dir.mkdir(parents=True)
    cfg_file = profile_dir / "config.yaml"
    cfg_file.write_text("""
model:
  default: worker-model
  provider: custom
providers:
  custom:
    base_url: https://worker.example/v1
""")

    monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda name: name == "no_key")
    monkeypatch.setattr("hermes_cli.profiles.get_profile_dir", lambda name: profile_dir)

    delegation_cfg = {"allowed_profiles": ["no_key"]}
    creds, err = _resolve_delegated_profile_authority("no_key", delegation_cfg, parent_agent=None)
    
    assert creds is None
    assert err == "Delegation refused: target profile 'no_key' has no API key configured for provider 'custom'"


def test_resolve_delegated_profile_authority_keyless_local_provider(tmp_path, monkeypatch):
    """Succeeds for keyless provider and sets api_key to empty string sentinel."""
    profile_dir = tmp_path / "profiles" / "local_ollama"
    profile_dir.mkdir(parents=True)
    cfg_file = profile_dir / "config.yaml"
    cfg_file.write_text("""
model:
  default: llama3.2:latest
  provider: ollama
providers:
  ollama:
    base_url: http://127.0.0.1:11434/v1
""")

    monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda name: name == "local_ollama")
    monkeypatch.setattr("hermes_cli.profiles.get_profile_dir", lambda name: profile_dir)

    delegation_cfg = {"allowed_profiles": ["local_ollama"]}
    realization, err = _resolve_delegated_profile_authority("local_ollama", delegation_cfg, parent_agent=None)
    
    assert err is None
    assert realization is not None
    assert realization.credentials["api_key"] == ""
    assert realization.credentials["is_keyless"] is True


def test_resolve_child_runtime_never_leaks_parent_api_key_to_profile():
    """Asserts that child runtime resolution never borrows parent's API key for profile delegation."""
    from tools.delegate_tool_config import _resolve_child_runtime

    parent = MagicMock()
    parent.model = "parent-model"
    parent.provider = "parent-provider"
    parent.base_url = "https://parent.example/v1"

    # Profile child with no key on keyless target: gets empty string, NEVER parent key
    rt_profile = _resolve_child_runtime(
        parent, {}, "PARENT_SECRET_TOKEN",
        model="worker-model", override_provider="custom",
        override_base_url="https://worker.example/v1",
        override_api_key=None, override_api_mode=None,
        override_acp_command=None, override_acp_args=None,
        profile_name="worker",
    )
    assert rt_profile["api_key"] == ""
    assert rt_profile["api_key"] != "PARENT_SECRET_TOKEN"

    # Ordinary non-profile child: inherits parent key
    rt_ordinary = _resolve_child_runtime(
        parent, {}, "PARENT_SECRET_TOKEN",
        model="worker-model", override_provider=None,
        override_base_url=None, override_api_key=None,
        override_api_mode=None, override_acp_command=None,
        override_acp_args=None, profile_name=None,
    )
    assert rt_ordinary["api_key"] == "PARENT_SECRET_TOKEN"


def test_profile_realization_lifecycle_barrier(tmp_path, monkeypatch):
    """ProfileRealization detects disappearance and generation changes."""
    profile_dir = tmp_path / "profiles" / "worker"
    profile_dir.mkdir(parents=True)
    cfg_file = profile_dir / "config.yaml"
    cfg_file.write_text("model:\n  default: m\n  provider: ollama\n")

    monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda name: name == "worker")
    monkeypatch.setattr("hermes_cli.profiles.get_profile_dir", lambda name: profile_dir)

    delegation_cfg = {"allowed_profiles": ["worker"]}
    realization, err = _resolve_delegated_profile_authority("worker", delegation_cfg, parent_agent=None)
    assert err is None
    assert realization is not None

    # Unchanged control
    ok, err = realization.verify()
    assert ok is True
    assert err is None

    # Disappearance
    cfg_file.unlink()
    profile_dir.rmdir()
    ok, err = realization.verify()
    assert ok is False
    assert "no longer exists" in err

    # Recreated / replaced (Generation B) with different inode
    profile_dir.mkdir(parents=True)
    cfg_file.write_text("model:\n  default: m\n  provider: ollama\n")
    mismatched_realization = ProfileRealization(
        name=realization.name,
        path=realization.path,
        device_inode=(realization.device_inode[0], realization.device_inode[1] + 9999),
        config_mtime_ns=realization.config_mtime_ns,
        config_hash=realization.config_hash,
        credentials=realization.credentials,
    )
    ok, err = mismatched_realization.verify()
    assert ok is False
    assert err is not None
    assert "generation mismatch" in err


def test_build_children_multitask_batch_fails_preflight(tmp_path, monkeypatch):
    """A multi-task batch with an invalid second task refuses pre-effect with zero children constructed."""
    from tools.delegate_tool import _build_children

    profile_dir = tmp_path / "profiles" / "valid_worker"
    profile_dir.mkdir(parents=True)
    cfg_file = profile_dir / "config.yaml"
    cfg_file.write_text("model:\n  default: m\n  provider: ollama\n")

    monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda name: name == "valid_worker")
    monkeypatch.setattr("hermes_cli.profiles.get_profile_dir", lambda name: profile_dir)

    task_list = [
        {"goal": "task 1", "profile": "valid_worker"},
        {"goal": "task 2", "profile": "unallowed_worker"},
    ]
    routing_cfg = {"allowed_profiles": ["valid_worker"]}
    parent = MagicMock()
    parent.model = "m"
    parent.provider = "p"
    parent.session_id = "sess"

    children, err = _build_children(
        task_list=task_list,
        task_schemas=[None, None],
        creds={"model": "m", "provider": "p", "base_url": None, "api_key": "k", "api_mode": None},
        top_role="leaf",
        max_iterations=1,
        parent_agent=parent,
        routing_cfg=routing_cfg,
        live_deleg_id=None,
        live_writers=[],
    )

    assert children == []
    assert "is not in delegation.allowed_profiles" in err


# ── Schema Tests ──────────────────────────────────────────────────────────


def test_delegate_task_schema_advertises_profile():
    """DELEGATE_TASK_SCHEMA declares 'profile' on task items."""
    task_items = DELEGATE_TASK_SCHEMA["parameters"]["properties"]["tasks"]["items"]
    assert "profile" in task_items["properties"]
    prop = task_items["properties"]["profile"]
    assert prop["type"] == "string"
    assert "Hermes profile" in prop["description"]


# ── Attribution and Lineage Tests ─────────────────────────────────────────


def test_subagent_profile_attribution_and_lineage(tmp_path, monkeypatch):
    """Subagent records delegated_profile and delegate_identity, preserving parent DB handle and lineage."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    import pm.environments
    monkeypatch.setattr(pm.environments, "activate_dependencies", lambda *a, **kw: None)
    import json
    from run_agent import _gateway_origin_json
    from tools.delegate_tool import _build_child_agent

    profile_dir = tmp_path / "profiles" / "persona_x"
    profile_dir.mkdir(parents=True)
    cfg_file = profile_dir / "config.yaml"
    cfg_file.write_text("model:\n  default: test-m\n  provider: mock_p\nproviders:\n  mock_p:\n    api_key: mock_k\n")

    monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda name: name == "persona_x")
    monkeypatch.setattr("hermes_cli.profiles.get_profile_dir", lambda name: profile_dir)
    def mock_build_client(agent, *args, **kwargs):
        agent._client_kwargs = {}
    monkeypatch.setattr("agent.agent_init._build_client", mock_build_client)

    mock_db = MagicMock()
    mock_db.db_path = str(tmp_path / "state.db")
    mock_db._own_profile_name = MagicMock(return_value="default")

    parent = MagicMock()
    parent.session_id = "parent_sess_123"
    parent.base_url = "http://localhost:11434/v1"
    parent.provider = "mock_p"
    parent.api_key = "mock_key"
    parent.model = "test-m"
    parent._session_db = mock_db
    parent._delegate_depth = 0
    parent._toolsets = []
    parent.prefill_messages = None
    parent.request_overrides = {}
    parent._print_fn = None

    monkeypatch.setattr("tools.delegate_tool._open_child_session_db", lambda p: mock_db)

    realization, err = _resolve_delegated_profile_authority("persona_x", {"allowed_profiles": ["persona_x"]}, parent_agent=parent)
    assert err is None

    child = _build_child_agent(
        task_index=0,
        goal="do task",
        context=None,
        toolsets=None,
        model="test-m",
        max_iterations=1,
        task_count=1,
        parent_agent=parent,
        profile_name="persona_x",
        profile_realization=realization,
    )

    try:
        assert child._parent_session_id == "parent_sess_123"
        assert child._session_db is mock_db
        assert child._delegated_profile == "persona_x"
        assert child._delegate_identity == "profile:persona_x"
        assert child._session_init_model_config["delegated_profile"] == "persona_x"
        assert child._session_init_model_config["delegate_identity"] == "profile:persona_x"
        assert child._session_init_model_config["_delegate_from"] == "parent_sess_123"

        origin_str = _gateway_origin_json(child)
        assert origin_str is not None
        origin = json.loads(origin_str)
        assert origin["delegated_profile"] == "persona_x"
        assert origin["delegate_identity"] == "profile:persona_x"
    finally:
        if hasattr(child, "close"):
            child.close()
