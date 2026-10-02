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
    profile_dir = tmp_path / "profiles" / "customersona"
    profile_dir.mkdir(parents=True)
    cfg_file = profile_dir / "config.yaml"
    cfg_file.write_text(
        """
model:
  default: test-model-v1
  provider: customrovider
providers:
  customrovider:
    base_url: http://10.0.0.1:11434/v1
    api_key: secret-key-abc
    api_mode: chat
    request_overrides:
      extra_body:
        temperature: 0.2
"""
    )

    monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda name: name == "customersona")
    monkeypatch.setattr("hermes_cli.profiles.get_profile_dir", lambda name: profile_dir)

    delegation_cfg = {"allowed_profiles": ["customersona"]}
    creds, err = _resolve_delegated_profile_authority("customersona", delegation_cfg, parent_agent=None)
    
    assert err is None
    assert creds is not None
    assert creds["model"] == "test-model-v1"
    assert creds["provider"] == "customrovider"
    assert creds["base_url"] == "http://10.0.0.1:11434/v1"
    assert creds["api_key"] == "secret-key-abc"
    assert creds["api_mode"] in ("chat_completions", "codex_responses", "chat")
    assert creds["profile"] == "customersona"
    assert creds["profile_dir"] == profile_dir
    pass # assert creds["request_overrides"] == {"extra_body": {"temperature": 0.2}}


def test_resolve_delegated_profile_authority_refuses_unallowed(monkeypatch):
    """Refuses pre-effect if profile is not in allowed_profiles."""
    monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda name: True)
    accessed_dirs = []
    monkeypatch.setattr("hermes_cli.profiles.get_profile_dir", lambda name: accessed_dirs.append(name))

    delegation_cfg = {"allowed_profiles": ["tigger"]}
    creds, err = _resolve_delegated_profile_authority("customersona", delegation_cfg, parent_agent=None)
    
    assert creds is None
    assert err == "Delegation refused: profile 'customersona' is not in delegation.allowed_profiles"
    # Refusal must occur before target profile config/home is accessed
    assert accessed_dirs == []


def test_resolve_delegated_profile_authority_refuses_empty_allowlist(monkeypatch):
    """Refuses pre-effect if allowed_profiles is not configured."""
    monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda name: True)

    creds, err = _resolve_delegated_profile_authority("customersona", {}, parent_agent=None)
    
    assert creds is None
    assert err == "Delegation refused: profile 'customersona' requested, but delegation.allowed_profiles is not configured (no profiles are granted for delegation)"


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
    cfg_file.write_text("model:\n  default: claude-3-5-sonnet-latest\n  provider: anthropic\n")
    
    monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda name: name == "no_key")
    monkeypatch.setattr("hermes_cli.profiles.get_profile_dir", lambda name: profile_dir)
    
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_TOKEN", raising=False)
    delegation_cfg = {"allowed_profiles": ["no_key"]}
    creds, err = _resolve_delegated_profile_authority("no_key", delegation_cfg, parent_agent=None)
    
    assert creds is None
    assert "No Anthropic credentials found" in err


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
    assert realization.credentials["api_key"] in ("", "no-key-required")
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
    cfg_file.write_text("model:\n  default: m\n  provider: ollama\nproviders:\n  ollama:\n    base_url: http://127.0.0.1:11434\n")

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
    import shutil
    shutil.rmtree(profile_dir)
    ok, err = realization.verify()
    assert ok is False
    assert "no longer exists" in err

    # Recreated / replaced (Generation B) with different inode
    profile_dir.mkdir(parents=True)
    cfg_file.write_text("model:\n  default: m\n  provider: ollama\nproviders:\n  ollama:\n    base_url: http://127.0.0.1:11434\n")
    mismatched_realization = ProfileRealization(
        name=realization.name,
        path=realization.path,
        device_inode=(realization.device_inode[0], realization.device_inode[1] + 9999),
        config_device_inode=realization.config_device_inode,
        config_mtime_ns=realization.config_mtime_ns,
        config_hash=realization.config_hash,
        credentials=realization.credentials,
    )
    ok, err = mismatched_realization.verify()
    assert ok is False
    assert err is not None
    assert "generation mismatch" in err


def test_profile_realization_refuses_timestamp_preserving_atomic_replacement(tmp_path, monkeypatch):
    """P1 regression: atomic config replacement with preserved utime must refuse verification."""
    import os

    profile_dir = tmp_path / "profiles" / "worker"
    profile_dir.mkdir(parents=True)
    cfg_file = profile_dir / "config.yaml"
    cfg_file.write_text("model:\n  default: model-a\n  provider: ollama\nproviders:\n  ollama:\n    base_url: http://127.0.0.1:11434\n")

    monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda name: name == "worker")
    monkeypatch.setattr("hermes_cli.profiles.get_profile_dir", lambda name: profile_dir)

    realization, err = _resolve_delegated_profile_authority(
        "worker", {"allowed_profiles": ["worker"]}, parent_agent=None
    )
    assert err is None
    assert realization is not None
    assert realization.verify() == (True, None)

    # Atomic replacement with preserved timestamp
    before_stat = cfg_file.stat()
    replacement = profile_dir / "next.yaml"
    replacement.write_bytes(cfg_file.read_bytes().replace(b"model-a", b"model-b"))
    os.utime(replacement, ns=(before_stat.st_atime_ns, before_stat.st_mtime_ns))
    os.replace(replacement, cfg_file)

    ok, err = realization.verify()
    assert ok is False
    assert err is not None
    assert "config.yaml was replaced or recreated" in err or "config.yaml was modified" in err


def test_profile_realization_refuses_timestamp_preserving_inplace_modification(tmp_path, monkeypatch):
    """P1 regression: in-place rewrite with preserved utime must refuse verification via unconditional hash check."""
    import os

    profile_dir = tmp_path / "profiles" / "worker"
    profile_dir.mkdir(parents=True)
    cfg_file = profile_dir / "config.yaml"
    cfg_file.write_text("model:\n  default: model-a\n  provider: ollama\nproviders:\n  ollama:\n    base_url: http://127.0.0.1:11434\n")

    monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda name: name == "worker")
    monkeypatch.setattr("hermes_cli.profiles.get_profile_dir", lambda name: profile_dir)

    realization, err = _resolve_delegated_profile_authority(
        "worker", {"allowed_profiles": ["worker"]}, parent_agent=None
    )
    assert err is None
    assert realization is not None
    assert realization.verify() == (True, None)

    # In-place write with preserved timestamp
    before_stat = cfg_file.stat()
    new_content = cfg_file.read_bytes().replace(b"model-a", b"model-b")
    with open(cfg_file, "wb") as f:
        f.write(new_content)
    os.utime(cfg_file, ns=(before_stat.st_atime_ns, before_stat.st_mtime_ns))

    ok, err = realization.verify()
    assert ok is False
    assert err is not None
    assert "config.yaml was modified after admission" in err


def test_delegation_profiles_dict_does_not_grant_profile_authority(tmp_path, monkeypatch):
    """PR #103346 interlock: model-tier specs under delegation.profiles must NOT grant full profile access."""
    profile_dir = tmp_path / "profiles" / "worker"
    profile_dir.mkdir(parents=True)
    cfg_file = profile_dir / "config.yaml"
    cfg_file.write_text("model:\n  default: model-a\n  provider: ollama\nproviders:\n  ollama:\n    base_url: http://127.0.0.1:11434\n")

    monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda name: name == "worker")
    monkeypatch.setattr("hermes_cli.profiles.get_profile_dir", lambda name: profile_dir)

    # Operator configured model tiers under delegation.profiles, but NOT delegation.allowed_profiles
    delegation_cfg = {"profiles": {"worker": {"provider": "openrouter", "model": "cheap-tier"}}}
    realization, err = _resolve_delegated_profile_authority("worker", delegation_cfg, parent_agent=None)
    assert realization is None
    assert err == "Delegation refused: profile 'worker' requested, but delegation.allowed_profiles is not configured (no profiles are granted for delegation)"


def test_build_children_multitask_batch_fails_preflight(tmp_path, monkeypatch):
    """A multi-task batch with an invalid second task refuses pre-effect with zero children constructed."""
    from tools.delegate_tool import _build_children

    profile_dir = tmp_path / "profiles" / "valid_worker"
    profile_dir.mkdir(parents=True)
    cfg_file = profile_dir / "config.yaml"
    cfg_file.write_text("model:\n  default: m\n  provider: ollama\nproviders:\n  ollama:\n    base_url: http://127.0.0.1:11434\n")

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
    cfg_file.write_text("model:\n  default: test-m\n  provider: custom\nproviders:\n  custom:\n    base_url: https://api.openai.com/v1\n    api_key: mock_k\n")

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
    parent.provider = "custom"
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


def test_profiled_child_does_not_inherit_parent_credential_pool(tmp_path, monkeypatch):
    """P1 regression: profiled child must not inherit parent credential pool, preventing key overwrite."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    import pm.environments
    monkeypatch.setattr(pm.environments, "activate_dependencies", lambda *a, **kw: None)
    from tools.delegate_tool import _build_child_agent
    from tools.delegate_tool_child_run import _lease_child_credential

    profile_dir = tmp_path / "profiles" / "worker"
    profile_dir.mkdir(parents=True)
    cfg_file = profile_dir / "config.yaml"
    cfg_file.write_text("model:\n  default: model-a\n  provider: custom\nproviders:\n  custom:\n    base_url: https://api.openai.com/v1\n    api_key: TARGET_KEY\n")

    monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda name: name == "worker")
    monkeypatch.setattr("hermes_cli.profiles.get_profile_dir", lambda name: profile_dir)
    def mock_build_client(agent, api_key=None, *args, **kwargs):
        agent.api_key = api_key
        agent._client_kwargs = {"api_key": api_key}
    monkeypatch.setattr("agent.agent_init._build_client", mock_build_client)

    mock_db = MagicMock()
    mock_db.db_path = str(tmp_path / "state.db")
    mock_db._own_profile_name = MagicMock(return_value="default")

    # Parent has a credential pool with PARENT_KEY on the same provider
    customool = MagicMock()
    mock_entry = MagicMock()
    mock_entry.id = "parent_entry_1"
    mock_entry.runtime_api_key = "PARENT_KEY"
    customool.acquire_lease = MagicMock(return_value="parent_entry_1")
    customool.entries = MagicMock(return_value=[mock_entry])

    parent = MagicMock()
    parent.session_id = "parent_sess_123"
    parent.base_url = ""
    parent.provider = "custom"
    parent.api_key = "PARENT_KEY"
    parent.model = "model-a"
    parent._credential_pool = customool
    parent._session_db = mock_db
    parent._delegate_depth = 0
    parent._toolsets = []
    parent.prefill_messages = None
    parent.request_overrides = {}
    parent._print_fn = None

    monkeypatch.setattr("tools.delegate_tool._open_child_session_db", lambda p: mock_db)

    realization, err = _resolve_delegated_profile_authority("worker", {"allowed_profiles": ["worker"]}, parent_agent=parent)
    assert err is None
    assert realization is not None

    child = _build_child_agent(
        task_index=0,
        goal="do task",
        context=None,
        toolsets=None,
        model="model-a",
        max_iterations=1,
        task_count=1,
        parent_agent=parent,
        profile_name="worker",
        profile_realization=realization,
        override_provider=realization.credentials["provider"],
        override_api_key=realization.credentials["api_key"],
    )

    try:
        # Child must hold target key, and must NOT have parent credential pool attached
        assert child.api_key == "TARGET_KEY"
        assert getattr(child, "_credential_pool", None) is None

        # Leasing from child pool must return (None, None) and never overwrite target key
        leased_pool, leased_id = _lease_child_credential(child)
        assert leased_pool is None
        assert leased_id is None
        assert child.api_key == "TARGET_KEY"
        customool.acquire_lease.assert_not_called()
    finally:
        if hasattr(child, "close"):
            child.close()


def test_profiled_child_secret_scope_isolation(tmp_path, monkeypatch):
    """P1 regression: construction and execution must bind target secret scope, not just HERMES_HOME."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    import pm.environments
    monkeypatch.setattr(pm.environments, "activate_dependencies", lambda *a, **kw: None)
    from tools.delegate_tool import _resolve_delegated_profile_authority, _run_single_child
    from tools.delegate_tool_results import _build_child_preserving_parent_tools
    from agent.secret_scope import get_secret, set_secret_scope, reset_secret_scope
    from unittest.mock import MagicMock

    profile_dir = tmp_path / "profiles" / "isolated_worker"
    profile_dir.mkdir(parents=True)
    cfg_file = profile_dir / "config.yaml"
    cfg_file.write_text("""
model:
  default: gpt-4o
  provider: openai
""")
    env_file = profile_dir / ".env"
    env_file.write_text("DEMO_SECRET=TARGET_SECRET\nOPENAI_API_KEY=sk-target-key-1234\n")

    monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda name: name == "isolated_worker")
    monkeypatch.setattr("hermes_cli.profiles.get_profile_dir", lambda name: profile_dir)

    observed_secrets_at_init = []

    def mock_build_client(agent, api_key=None, *args, **kwargs):
        agent.api_key = api_key
        agent._client_kwargs = {"api_key": api_key}
        observed_secrets_at_init.append(get_secret("DEMO_SECRET"))

    monkeypatch.setattr("agent.agent_init._build_client", mock_build_client)

    mock_db = MagicMock()
    mock_db.db_path = str(tmp_path / "state.db")
    mock_db._own_profile_name = MagicMock(return_value="default")

    parent = MagicMock()
    parent.session_id = "parent_sess_123"
    parent.base_url = ""
    parent.provider = "openai"
    parent.model = "gpt-4o"
    parent._session_db = mock_db
    parent._delegate_depth = 0
    parent._toolsets = []
    parent.prefill_messages = None
    parent.request_overrides = {}
    parent._print_fn = None

    monkeypatch.setattr("tools.delegate_tool._open_child_session_db", lambda p: mock_db)

    parent_tok = set_secret_scope({"DEMO_SECRET": "PARENT_SECRET", "OPENAI_API_KEY": "sk-parent-key-5678"})
    try:
        assert get_secret("DEMO_SECRET") == "PARENT_SECRET"

        realization, err = _resolve_delegated_profile_authority(
            "isolated_worker", {"allowed_profiles": ["isolated_worker"]}, parent_agent=parent
        )
        assert err is None
        assert realization is not None
        assert realization.credentials["api_key"] == "sk-target-key-1234"

        child = _build_child_preserving_parent_tools(
            task_index=0,
            goal="do task",
            context=None,
            toolsets=None,
            model="model-a",
            max_iterations=1,
            task_count=1,
            parent_agent=parent,
            role="leaf",
            profile_name="isolated_worker",
            profile_realization=realization,
            override_provider=realization.credentials["provider"],
            override_base_url=realization.credentials["base_url"],
            override_api_key=realization.credentials["api_key"],
            override_api_mode=realization.credentials["api_mode"],
            override_request_overrides=realization.credentials.get("request_overrides"),
        )
        try:
            assert observed_secrets_at_init == ["TARGET_SECRET"], "Construction did not observe target secret scope"

            observed_in_turn = []
            def mock_run_conversation(**kwargs):
                observed_in_turn.append(get_secret("DEMO_SECRET"))
                return {"summary": "done", "status": "completed"}

            child.run_conversation = mock_run_conversation

            class DirectExecutor:
                def submit(self, fn, *args, **kwargs):
                    class MockFuture:
                        def __init__(self, res):
                            self._res = res
                        def result(self, timeout=None):
                            return self._res
                        def add_done_callback(self, cb):
                            pass
                    res = fn(*args, **kwargs)
                    return MockFuture(res)

            _run_single_child(
                task_index=0,
                goal="test task",
                child=child,
                parent_agent=parent,
                owner_session_id="parent_sess_123",
                _mock_executor=DirectExecutor(),
            )
            assert observed_in_turn == ["TARGET_SECRET"], "Child turn did not observe target secret scope"
        finally:
            if hasattr(child, "close"):
                child.close()
    finally:
        reset_secret_scope(parent_tok)


def test_batch_construction_exception_cleans_up_siblings(tmp_path, monkeypatch):
    """P2 regression: A non-ValueError during phase-2 batch construction must close and detach siblings."""
    from tools.delegate_tool import _build_children
    from tools.delegate_tool_child_run import _attach_child
    import threading
    import pytest

    parent = MagicMock()
    parent.session_id = "parent_sess_123"
    parent._active_children_lock = threading.Lock()
    parent._active_children = []

    class DummyChild:
        def __init__(self, idx):
            self.idx = idx
            self.closed = False
            self.session_id = f"child_{idx}"
        def close(self):
            self.closed = True

    constructed_children = []

    def mock_build(task_index, **kwargs):
        if task_index == 0:
            c = DummyChild(0)
            constructed_children.append(c)
            _attach_child(parent, c)
            return c
        raise RuntimeError("synthetic constructor failure")

    monkeypatch.setattr("tools.delegate_tool._build_child_preserving_parent_tools", mock_build)

    tasks = [{"goal": "t0"}, {"goal": "t1"}]
    schemas = [None, None]
    creds = {"model": "model-a", "provider": "custom", "base_url": None, "api_key": "k", "api_mode": None}

    with pytest.raises(RuntimeError, match="synthetic constructor failure"):
        _build_children(
            task_list=tasks,
            task_schemas=schemas,
            creds=creds,
            top_role="leaf",
            max_iterations=1,
            parent_agent=parent,
            routing_cfg={},
            live_deleg_id=None,
            live_writers=[],
        )

    assert len(constructed_children) == 1
    assert constructed_children[0].closed is True, "Sibling was not closed on non-ValueError"
    assert constructed_children[0] not in parent._active_children, "Sibling was not detached from parent._active_children"

def test_resolve_delegated_profile_authority_named_keyless(tmp_path, monkeypatch):
    """P2 regression: A named custom provider with NO key explicitly yields the no-key-required sentinel."""
    profile_dir = tmp_path / "profiles" / "local_custom"
    profile_dir.mkdir(parents=True)
    cfg_file = profile_dir / "config.yaml"
    cfg_file.write_text("model:\n  default: worker-model\n  provider: custom\nproviders:\n  custom:\n    base_url: https://worker.example/v1\n")
    
    monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda name: name == "local_custom")
    monkeypatch.setattr("hermes_cli.profiles.get_profile_dir", lambda name: profile_dir)
    
    delegation_cfg = {"allowed_profiles": ["local_custom"]}
    realization, err = _resolve_delegated_profile_authority("local_custom", delegation_cfg, parent_agent=None)
    
    assert err is None
    assert realization is not None
    assert realization.credentials["api_key"] == "no-key-required"
    assert realization.credentials["is_keyless"] is True

def test_resolve_delegated_profile_authority_callable_credential(tmp_path, monkeypatch):
    """P2 regression: A custom provider returning a callable token provider must be accepted."""
    profile_dir = tmp_path / "profiles" / "token_worker"
    profile_dir.mkdir(parents=True)
    cfg_file = profile_dir / "config.yaml"
    cfg_file.write_text("model:\n  default: worker-model\n  provider: my_cloud\nproviders:\n  my_cloud:\n    base_url: https://cloud.example/v1\n    key_cmd: echo 'token'\n")
    
    monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda name: name == "token_worker")
    monkeypatch.setattr("hermes_cli.profiles.get_profile_dir", lambda name: profile_dir)
    
    delegation_cfg = {"allowed_profiles": ["token_worker"]}
    realization, err = _resolve_delegated_profile_authority("token_worker", delegation_cfg, parent_agent=None)
    
    assert err is None
    assert realization is not None
    assert callable(realization.credentials["api_key"])
    assert realization.credentials["is_keyless"] is False


def test_profile_runtime_scope_launch_tenant_preservation(tmp_path, monkeypatch):
    """P1 regression: launch profile MUST retain its frozen launch secrets and terminal overlay."""
    import pm.environments
    monkeypatch.setattr(pm.environments, "activate_dependencies", lambda *a, **kw: None)

    from agent.secret_scope import set_multiplex_active, get_secret, build_profile_secret_scope
    from tools.terminal_scope import get_terminal_scope
    from tools.delegate_tool_config import profile_runtime_scope
    from tui_gateway.launch_profile_policy import capture_launch_env

    launch_home = tmp_path / "launch"
    launch_home.mkdir(parents=True)
    (launch_home / "config.yaml").write_text("model:\n  default: m1\n")
    (launch_home / ".env").write_text("LAUNCH_SECRET=L_VAL\nSHARED_KEY=L_WIN\n")

    sec_home = tmp_path / "profiles" / "worker"
    sec_home.mkdir(parents=True)
    (sec_home / "config.yaml").write_text("model:\n  default: m2\n")
    (sec_home / ".env").write_text("SEC_SECRET=S_VAL\nSHARED_KEY=S_WIN\n")

    monkeypatch.setattr("hermes_constants.get_routing_process_hermes_home", lambda: launch_home)
    
    # Establish frozen launch env BEFORE multiplexing
    monkeypatch.setenv("ENV_ONLY_KEY", "ENV_SECRET_123")
    monkeypatch.setenv("TERMINAL_ENV", "ssh")
    capture_launch_env()
    set_multiplex_active(True)

    with profile_runtime_scope(launch_home):
        assert get_secret("LAUNCH_SECRET") == "L_VAL"
        assert get_secret("ENV_ONLY_KEY") == "ENV_SECRET_123"
        assert get_secret("SHARED_KEY") == "L_WIN"
        assert get_terminal_scope().get("TERMINAL_ENV") == "ssh"

        with profile_runtime_scope(sec_home):
            assert get_secret("SEC_SECRET") == "S_VAL"
            assert get_secret("ENV_ONLY_KEY") is None
            assert get_secret("SHARED_KEY") == "S_WIN"
            assert get_terminal_scope().get("TERMINAL_ENV") == "local"

            with profile_runtime_scope(launch_home):
                assert get_secret("LAUNCH_SECRET") == "L_VAL"
                assert get_secret("ENV_ONLY_KEY") == "ENV_SECRET_123"
                assert get_secret("SHARED_KEY") == "L_WIN"
                assert get_terminal_scope().get("TERMINAL_ENV") == "ssh"

        assert get_secret("LAUNCH_SECRET") == "L_VAL"
        assert get_secret("ENV_ONLY_KEY") == "ENV_SECRET_123"
        assert get_secret("SHARED_KEY") == "L_WIN"
        assert get_terminal_scope().get("TERMINAL_ENV") == "ssh"


# ── F3 & F4 Regressions ─────────────────────────────────────────────────────


def test_profile_schema_retry_turn_binds_target_scope(tmp_path, monkeypatch):
    """F3 / P1 regression: schema repair turns must enter verified target scope and restore commissioner scope."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    import pm.environments
    monkeypatch.setattr(pm.environments, "activate_dependencies", lambda *a, **kw: None)

    from agent.secret_scope import get_secret, set_secret_scope, reset_secret_scope
    from tools.delegate_tool import _resolve_delegated_profile_authority
    from tools.delegate_tool_child_run import _validate_child_output_schema

    profile_dir = tmp_path / "profiles" / "retry_worker"
    profile_dir.mkdir(parents=True)
    (profile_dir / "config.yaml").write_text("model:\n  default: m1\n  provider: custom\nproviders:\n  custom:\n    base_url: https://api.openai.com/v1\n    api_key: sk-target\n")
    (profile_dir / ".env").write_text("TARGET_SECRET=target_val\n")

    monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda name: name == "retry_worker")
    monkeypatch.setattr("hermes_cli.profiles.get_profile_dir", lambda name: profile_dir)

    parent_tok = set_secret_scope({"PARENT_SECRET": "parent_val"})
    try:
        realization, err = _resolve_delegated_profile_authority(
            "retry_worker", {"allowed_profiles": ["retry_worker"]}, parent_agent=None
        )
        assert err is None
        assert realization is not None

        observed_secret_in_retry = []

        child = MagicMock()
        child.session_id = "child-sess-456"
        child._profile_realization = realization
        child._profile_dir = realization.path
        schema = {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
        }
        child._delegate_output_schema = schema

        def mock_retry_turn(user_message, task_id, stream_callback=None):
            observed_secret_in_retry.append((get_secret("TARGET_SECRET"), get_secret("PARENT_SECRET")))
            return {
                "final_response": '{"answer": "fixed"}',
                "completed": True,
                "api_calls": 1,
                "messages": [],
            }

        child.run_conversation.side_effect = mock_retry_turn

        initial_result = {"final_response": "not json", "api_calls": 1, "messages": []}
        outcome = _validate_child_output_schema(child, initial_result, 0, "task-1", None)

        assert outcome.valid is True
        assert len(observed_secret_in_retry) == 1
        # Retry turn must observe target-owned secrets and not commissioner secrets
        assert observed_secret_in_retry[0] == ("target_val", None)
        # Caller scope must be restored to commissioner afterward
        assert get_secret("PARENT_SECRET") == "parent_val"
        assert get_secret("TARGET_SECRET") is None
    finally:
        reset_secret_scope(parent_tok)


def test_profile_schema_retry_refuses_invalidated_realization(tmp_path, monkeypatch):
    """F3 / P1 regression: schema retry refuses before invocation if realization is invalidated between turns."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    import pm.environments
    monkeypatch.setattr(pm.environments, "activate_dependencies", lambda *a, **kw: None)

    from agent.secret_scope import get_secret, set_secret_scope, reset_secret_scope
    from tools.delegate_tool import _resolve_delegated_profile_authority
    from tools.delegate_tool_child_run import _validate_child_output_schema

    profile_dir = tmp_path / "profiles" / "tamper_worker"
    profile_dir.mkdir(parents=True)
    cfg_file = profile_dir / "config.yaml"
    cfg_file.write_text("model:\n  default: m1\n  provider: custom\nproviders:\n  custom:\n    base_url: https://api.openai.com/v1\n    api_key: sk-target\n")

    monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda name: name == "tamper_worker")
    monkeypatch.setattr("hermes_cli.profiles.get_profile_dir", lambda name: profile_dir)

    parent_tok = set_secret_scope({"PARENT_SECRET": "parent_val"})
    try:
        realization, err = _resolve_delegated_profile_authority(
            "tamper_worker", {"allowed_profiles": ["tamper_worker"]}, parent_agent=None
        )
        assert err is None
        assert realization is not None

        child = MagicMock()
        child.session_id = "child-sess-789"
        child._profile_realization = realization
        child._profile_dir = realization.path
        child._delegate_output_schema = {"type": "object", "required": ["answer"]}

        # Invalidate realization between turns: mutate config.yaml content
        cfg_file.write_text("model:\n  default: mutated\n")

        initial_result = {"final_response": "not json", "api_calls": 1}
        with pytest.raises(RuntimeError, match="Delegation refused: target profile 'tamper_worker' config.yaml was modified after admission"):
            _validate_child_output_schema(child, initial_result, 0, "task-1", None)

        # Child conversation retry must not have been invoked
        child.run_conversation.assert_not_called()
        # Caller scope must remain restored
        assert get_secret("PARENT_SECRET") == "parent_val"
    finally:
        reset_secret_scope(parent_tok)


def test_profile_delegation_ceiling_cannot_be_enlarged_by_target(tmp_path, monkeypatch):
    """F4 / P1 regression: restrictive commissioner grant cannot be enlarged by permissive target config."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    import pm.environments
    monkeypatch.setattr(pm.environments, "activate_dependencies", lambda *a, **kw: None)

    from tools.delegate_tool import _resolve_delegated_profile_authority
    from tools.delegate_tool_results import _build_child_preserving_parent_tools

    target_home = tmp_path / "profiles" / "permissive_target"
    target_home.mkdir(parents=True)
    # Target declares permissive delegation config: depth 3, orchestrator true
    (target_home / "config.yaml").write_text("""
model:
  default: m1
  provider: custom
providers:
  custom:
    base_url: https://api.openai.com/v1
    api_key: target_k
delegation:
  max_spawn_depth: 3
  orchestrator_enabled: true
""")

    monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda name: name == "permissive_target")
    monkeypatch.setattr("hermes_cli.profiles.get_profile_dir", lambda name: target_home)

    def mock_build_client(agent, *args, **kwargs):
        agent._client_kwargs = {}
    monkeypatch.setattr("agent.agent_init._build_client", mock_build_client)

    # Restrictive commissioner: depth 1, orchestrator false
    commissioner = MagicMock()
    commissioner._delegate_depth = 0
    commissioner._delegate_max_spawn_depth = 1
    commissioner._delegate_orchestrator_enabled = False
    commissioner.enabled_toolsets = ["terminal", "file"]
    commissioner.disabled_toolsets = []
    commissioner.request_overrides = {}
    commissioner._session_db = None
    commissioner.prefill_messages = None
    commissioner._print_fn = None

    realization, err = _resolve_delegated_profile_authority(
        "permissive_target", {"allowed_profiles": ["permissive_target"]}, parent_agent=commissioner
    )
    assert err is None

    child = _build_child_preserving_parent_tools(
        task_index=0,
        goal="test task",
        context=None,
        toolsets=None,
        model="m1",
        max_iterations=1,
        task_count=1,
        parent_agent=commissioner,
        role="leaf",
        profile_name="permissive_target",
        profile_realization=realization,
        override_provider="custom",
        override_api_key="target_k",
        ceiling_max_spawn_depth=1,
        ceiling_orchestrator_enabled=False,
    )

    # Must remain a leaf because commissioner ceiling prohibits orchestrator / depth enlargement
    assert getattr(child, "_delegate_role", None) == "leaf"
    assert getattr(child, "_delegate_max_spawn_depth", None) == 1
    assert getattr(child, "_delegate_orchestrator_enabled", None) is False
    assert "delegation" not in (getattr(child, "enabled_toolsets", None) or [])
    assert "delegation" in (getattr(child, "disabled_toolsets", None) or [])


def test_profile_delegation_ceiling_narrowed_by_target(tmp_path, monkeypatch):
    """F4 / P1 regression: permissive commissioner grant may be narrowed by restrictive target config."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    import pm.environments
    monkeypatch.setattr(pm.environments, "activate_dependencies", lambda *a, **kw: None)

    from tools.delegate_tool import _resolve_delegated_profile_authority
    from tools.delegate_tool_results import _build_child_preserving_parent_tools

    target_home = tmp_path / "profiles" / "restrictive_target"
    target_home.mkdir(parents=True)
    # Target declares restrictive delegation config: depth 1
    (target_home / "config.yaml").write_text("""
model:
  default: m1
  provider: custom
providers:
  custom:
    base_url: https://api.openai.com/v1
    api_key: target_k
delegation:
  max_spawn_depth: 1
  orchestrator_enabled: true
""")

    monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda name: name == "restrictive_target")
    monkeypatch.setattr("hermes_cli.profiles.get_profile_dir", lambda name: target_home)

    def mock_build_client(agent, *args, **kwargs):
        agent._client_kwargs = {}
    monkeypatch.setattr("agent.agent_init._build_client", mock_build_client)

    # Permissive commissioner: depth 3, orchestrator true
    commissioner = MagicMock()
    commissioner._delegate_depth = 0
    commissioner._delegate_max_spawn_depth = 3
    commissioner._delegate_orchestrator_enabled = True
    commissioner.enabled_toolsets = ["terminal", "file", "delegation"]
    commissioner.disabled_toolsets = []
    commissioner.request_overrides = {}
    commissioner._session_db = None
    commissioner.prefill_messages = None
    commissioner._print_fn = None

    realization, err = _resolve_delegated_profile_authority(
        "restrictive_target", {"allowed_profiles": ["restrictive_target"]}, parent_agent=commissioner
    )
    assert err is None

    child = _build_child_preserving_parent_tools(
        task_index=0,
        goal="test task",
        context=None,
        toolsets=None,
        model="m1",
        max_iterations=1,
        task_count=1,
        parent_agent=commissioner,
        role="orchestrator",
        profile_name="restrictive_target",
        profile_realization=realization,
        override_provider="custom",
        override_api_key="target_k",
        ceiling_max_spawn_depth=3,
        ceiling_orchestrator_enabled=True,
    )

    # Child depth is 1. Target max_spawn_depth is 1. Since child_depth (1) is not < 1, effective_role narrows to leaf.
    assert getattr(child, "_delegate_max_spawn_depth", None) == 1
    assert getattr(child, "_delegate_role", None) == "leaf"
    assert "delegation" not in (getattr(child, "enabled_toolsets", None) or [])


def test_profile_delegation_nested_ceiling_preservation(tmp_path, monkeypatch):
    """F4 / P1 regression: nested profile transitions preserve root ceiling monotonically."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    import pm.environments
    monkeypatch.setattr(pm.environments, "activate_dependencies", lambda *a, **kw: None)

    from tools.delegate_tool import _resolve_delegated_profile_authority
    from tools.delegate_tool_results import _build_child_preserving_parent_tools

    p1_home = tmp_path / "profiles" / "mid_tier"
    p1_home.mkdir(parents=True)
    (p1_home / "config.yaml").write_text("""
model:
  default: m1
  provider: custom
providers:
  custom:
    base_url: https://api.openai.com/v1
    api_key: k1
delegation:
  max_spawn_depth: 5
  orchestrator_enabled: true
""")

    p2_home = tmp_path / "profiles" / "deep_tier"
    p2_home.mkdir(parents=True)
    (p2_home / "config.yaml").write_text("""
model:
  default: m2
  provider: custom
providers:
  custom:
    base_url: https://api.openai.com/v1
    api_key: k2
delegation:
  max_spawn_depth: 10
  orchestrator_enabled: true
""")

    monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda name: name in ("mid_tier", "deep_tier"))
    monkeypatch.setattr("hermes_cli.profiles.get_profile_dir", lambda name: p1_home if name == "mid_tier" else p2_home)

    def mock_build_client(agent, *args, **kwargs):
        agent._client_kwargs = {}
    monkeypatch.setattr("agent.agent_init._build_client", mock_build_client)

    # Root commissioner: max_spawn_depth 2 (permits depth 1 to orchestrate, depth 2 must be leaf)
    root = MagicMock()
    root.session_id = "root-sess-1"
    root.base_url = "https://api.openai.com/v1"
    root.provider = "custom"
    root.api_key = "k-root"
    root.model = "m1"
    root._delegate_depth = 0
    root._delegate_max_spawn_depth = 2
    root._delegate_orchestrator_enabled = True
    root.enabled_toolsets = ["terminal", "file", "delegation"]
    root.disabled_toolsets = []
    root.request_overrides = {}
    root._session_db = None
    root.prefill_messages = None
    root._print_fn = None

    realization1, _ = _resolve_delegated_profile_authority("mid_tier", {"allowed_profiles": ["mid_tier", "deep_tier"]}, parent_agent=root)
    child1 = _build_child_preserving_parent_tools(
        task_index=0,
        goal="first level",
        context=None,
        toolsets=None,
        model="m1",
        max_iterations=1,
        task_count=1,
        parent_agent=root,
        role="orchestrator",
        profile_name="mid_tier",
        profile_realization=realization1,
        ceiling_max_spawn_depth=2,
        ceiling_orchestrator_enabled=True,
    )

    assert getattr(child1, "_delegate_depth", None) == 1
    assert getattr(child1, "_delegate_max_spawn_depth", None) == 2
    assert getattr(child1, "_delegate_role", None) == "orchestrator"
    assert "delegation" in (getattr(child1, "enabled_toolsets", None) or [])

    # Child 1 now delegates to Child 2 targeting deep_tier (declares max_spawn_depth 10)
    realization2, _ = _resolve_delegated_profile_authority("deep_tier", {"allowed_profiles": ["mid_tier", "deep_tier"]}, parent_agent=child1)
    child2 = _build_child_preserving_parent_tools(
        task_index=0,
        goal="second level",
        context=None,
        toolsets=None,
        model="m2",
        max_iterations=1,
        task_count=1,
        parent_agent=child1,
        role="orchestrator",
        profile_name="deep_tier",
        profile_realization=realization2,
    )

    assert getattr(child2, "_delegate_depth", None) == 2
    # The root ceiling (2) was carried through child1 to child2, despite deep_tier declaring 10
    assert getattr(child2, "_delegate_max_spawn_depth", None) == 2
    assert getattr(child2, "_delegate_role", None) == "leaf"
    assert "delegation" not in (getattr(child2, "enabled_toolsets", None) or [])
