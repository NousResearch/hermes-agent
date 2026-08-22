"""Behavioral contracts for the opt-in safe configuration tool."""

from __future__ import annotations

import json
import threading
import time
from argparse import Namespace
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import cast

import pytest
import yaml

from hermes_cli.tools_config import _get_platform_tools, tools_disable_enable_command
from model_tools import get_tool_definitions
from toolsets import TOOLSETS, _HERMES_CORE_TOOLS
from tools.config_set_tool import (
    CONFIG_SET_TOOL_SCHEMA,
    WRITABLE_CONFIG_KEYS,
    _is_credential_shaped,
    _is_whitelisted,
    _safe_audit_value,
    config_set_value,
)


_VALID_VALUE_BY_KEY = {
    "compression.enabled": "false",
    "compression.threshold": "0.75",
    "display.show_reasoning": "true",
    "display.skin": "default",
    "display.tool_progress": "off",
    "stt.local.model": "tiny",
    "tts.deepinfra.voice": "default",
    "tts.edge.voice": "en-US-AriaNeural",
    "tts.elevenlabs.voice_id": "pNInz6obpgDQGcFmaJgB",
    "tts.gemini.voice": "Kore",
    "tts.kittentts.voice": "Jasper",
    "tts.minimax.voice_id": "English_expressive_narrator",
    "tts.mistral.voice_id": "c69964a6-ab8b-4f8a-9465-ec0925096ec8",
    "tts.openai.voice": "alloy",
    "tts.xai.voice_id": "eve",
}


def _write_config(home: Path, text: str = "{}\n") -> Path:
    home.mkdir(parents=True, exist_ok=True)
    path = home / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def _install_mismatched_writer(monkeypatch, transform=lambda value: value.upper()):
    """Install a canonical writer that persists a value the validator never
    approved, deterministically forcing a read-back verification mismatch."""
    import hermes_cli.config as config_module

    real_writer = config_module.set_config_value

    def mismatched_writer(key, value):
        real_writer(key, transform(value))

    monkeypatch.setattr(config_module, "set_config_value", mismatched_writer)


def _call(key: str, value, session_id: str = "test-session") -> dict:
    return json.loads(config_set_value(key, value, session_id=session_id))


@pytest.mark.parametrize("key", sorted(WRITABLE_CONFIG_KEYS))
def test_every_approved_leaf_is_authorized(key: str):
    assert _is_whitelisted(key) is True


@pytest.mark.parametrize(
    "key",
    [
        "",
        "stt",
        "stt.enabled.extra",
        "stt.future_setting",
        "STT.ENABLED",
        " stt.enabled",
        "stt.enabled ",
        ".stt.enabled",
        "stt..enabled",
        "stt[enabled]",
        "stt.0.enabled",
        "displayed.skin",
        "compression_extra.enabled",
        "mcp_servers.context7.command",
        "mcp_servers.context7.args",
        "mcp_servers.context7.env.API_KEY",
        "mcp_servers.context7.url",
        "mcp_servers.context7.headers.Authorization",
        "stt.enabled",
        "stt.provider",
        "tts.provider",
        "tts.piper.voice",
        "custom_providers.example.command",
        "platform_toolsets.cli",
        "auxiliary.approval.enabled",
        "approvals.mode",
        "security.redact_secrets",
        "terminal.backend",
        "delegation.max_spawn_depth",
        "model.default",
        "webhook.enabled",
    ],
)
def test_unknown_sibling_descendant_and_trust_boundary_keys_are_denied(key: str):
    assert _is_whitelisted(key) is False


def test_schema_enumerates_only_the_reviewed_leaves():
    key_schema = CONFIG_SET_TOOL_SCHEMA["parameters"]["properties"]["key"]
    assert set(key_schema["enum"]) == WRITABLE_CONFIG_KEYS
    assert CONFIG_SET_TOOL_SCHEMA["parameters"]["additionalProperties"] is False
    assert set(_VALID_VALUE_BY_KEY) == WRITABLE_CONFIG_KEYS


def test_default_off_enable_disable_uses_official_tool_configuration(
    tmp_path, monkeypatch
):
    home = tmp_path / "home"
    _write_config(home)
    monkeypatch.setenv("HERMES_HOME", str(home))

    assert "config" not in _get_platform_tools(
        {}, "cli", include_default_mcp_servers=False
    )

    tools_disable_enable_command(
        Namespace(tools_action="enable", names=["config"], platform="cli")
    )
    enabled_config = yaml.safe_load((home / "config.yaml").read_text(encoding="utf-8"))
    enabled = _get_platform_tools(
        enabled_config, "cli", include_default_mcp_servers=False
    )
    assert "config" in enabled

    enabled_defs = get_tool_definitions(
        enabled_toolsets=sorted(enabled),
        quiet_mode=True,
        skip_tool_search_assembly=True,
    )
    assert "hermes_config_set" in {item["function"]["name"] for item in enabled_defs}

    tools_disable_enable_command(
        Namespace(tools_action="disable", names=["config"], platform="cli")
    )
    disabled_config = yaml.safe_load((home / "config.yaml").read_text(encoding="utf-8"))
    disabled = _get_platform_tools(
        disabled_config, "cli", include_default_mcp_servers=False
    )
    assert "config" not in disabled


def test_tool_has_no_default_core_schema_footprint():
    assert "hermes_config_set" not in _HERMES_CORE_TOOLS
    core_bundle_tools = cast(list[str], TOOLSETS["hermes-cli"]["tools"])
    assert "hermes_config_set" not in core_bundle_tools

    default_defs = get_tool_definitions(
        enabled_toolsets=["hermes-cli"],
        quiet_mode=True,
        skip_tool_search_assembly=True,
    )
    assert "hermes_config_set" not in {
        item["function"]["name"] for item in default_defs
    }


@pytest.mark.parametrize(
    ("key", "value", "initial"),
    [
        ("display.show_reasoning", "true", "display:\n  show_reasoning: false\n"),
        ("compression.threshold", "0.75", "compression:\n  threshold: 0.80\n"),
        ("display.tool_progress", "off", "{}\n"),
        ("tts.edge.voice", "en-US-AriaNeural", "{}\n"),
    ],
)
def test_real_tool_write_persists_and_matches_canonical_writer(
    key, value, initial, tmp_path, monkeypatch
):
    tool_home = tmp_path / "tool-home"
    direct_home = tmp_path / "direct-home"
    _write_config(tool_home, initial)
    _write_config(direct_home, initial)

    monkeypatch.setenv("HERMES_HOME", str(tool_home))
    result = _call(key, value)
    assert result["success"] is True
    assert result["key"] == key
    assert result["audit_logged"] is True
    tool_config = yaml.safe_load(
        (tool_home / "config.yaml").read_text(encoding="utf-8")
    )
    monkeypatch.setenv("HERMES_HOME", str(direct_home))
    from hermes_cli.config import set_config_value

    set_config_value(key, value)
    direct_config = yaml.safe_load(
        (direct_home / "config.yaml").read_text(encoding="utf-8")
    )
    assert tool_config == direct_config


def test_concurrent_writes_are_serialized_and_both_persist(tmp_path, monkeypatch):
    home = tmp_path / "home"
    _write_config(home)
    monkeypatch.setenv("HERMES_HOME", str(home))

    from hermes_cli.config import set_config_value as canonical_writer

    counter_lock = threading.Lock()
    active = 0
    max_active = 0

    def observed_writer(key, value):
        nonlocal active, max_active
        with counter_lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.05)
        try:
            canonical_writer(key, value)
        finally:
            with counter_lock:
                active -= 1

    monkeypatch.setattr("hermes_cli.config.set_config_value", observed_writer)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(
            pool.map(
                lambda pair: _call(*pair),
                [
                    ("display.show_reasoning", "true"),
                    ("compression.enabled", "false"),
                ],
            )
        )

    config = yaml.safe_load((home / "config.yaml").read_text(encoding="utf-8"))
    assert all(result["success"] is True for result in results)
    assert max_active == 1
    assert config["display"]["show_reasoning"] is True
    assert config["compression"]["enabled"] is False


def test_denied_mutation_is_byte_identical_and_audited(tmp_path, monkeypatch):
    home = tmp_path / "home"
    config_path = _write_config(
        home,
        "approvals:\n  mode: manual\nmcp_servers:\n  context7:\n    command: npx\n",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    before = config_path.read_bytes()

    result = _call("mcp_servers.context7.command", "malicious-command")

    assert result["success"] is False
    assert result["blocked"] is True
    assert result["audit_logged"] is True
    assert config_path.read_bytes() == before
    audit = (home / "logs" / "config_changes.jsonl").read_text(encoding="utf-8")
    assert '"status": "denied"' in audit


@pytest.mark.parametrize(
    "value",
    [
        "sk-proj-AbCdEfGhIjKlMnOpQrStUvWxYz0123456789",
        "sk-***",
        "sk-...",
        "sk-proj-AbC...xyz7890",
        "sk-<api-key>",
        "sk-PLACEHOLDER",
        "sk-CHANGEME",
        "Bearer this-is-a-secret-token",
        "ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdef",
        "AKIAABCDEFGHIJKLMNOP",
        "hf_abcdefghijklmnopqrstuvwxyz1234567890",
        "«redacted:sk-…»",
    ],
)
def test_secrets_and_redaction_sentinels_are_blocked_and_not_logged(
    value: str, tmp_path, monkeypatch
):
    home = tmp_path / "home"
    config_path = _write_config(home, "tts:\n  edge:\n    voice: safe-voice\n")
    monkeypatch.setenv("HERMES_HOME", str(home))
    before = config_path.read_bytes()

    result = _call("tts.edge.voice", value)

    assert result["success"] is False
    assert result["blocked"] is True
    assert config_path.read_bytes() == before
    audit = (home / "logs" / "config_changes.jsonl").read_text(encoding="utf-8")
    assert value not in audit
    assert "redacted" in audit.lower()


@pytest.mark.parametrize(
    "value",
    [
        "a normal benign voice name that is deliberately longer than thirty-two characters",
        "https://voice.example.test/path",
        "anthropic/claude-sonnet-4",
        "c69964a6-ab8b-4f8a-9465-ec0925096ec8",
        "voices/en_US-lessac-medium.onnx",
    ],
)
def test_benign_long_url_model_voice_and_path_strings_do_not_false_positive(value: str):
    assert _is_credential_shaped(value) is False


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("stt.enabled", "yes"),
        ("stt.provider", "local_command"),
        ("stt.provider", "custom-command-provider"),
        ("tts.provider", "custom-command-provider"),
        ("stt.local.model", "future-model"),
        ("display.tool_progress", "future-mode"),
        ("compression.threshold", "nan"),
        ("compression.threshold", "0.49"),
        ("compression.threshold", "0.96"),
        ("tts.edge.voice", {"nested": "value"}),
        ("tts.edge.voice", "voice\x7fhidden"),
        ("tts.edge.voice", "voice\x85hidden"),
        ("tts.edge.voice", "voice\u200bhidden"),
        ("tts.edge.voice", "voice\u202eevil"),
    ],
)
def test_allowed_keys_still_require_safe_typed_values(
    key, value, tmp_path, monkeypatch
):
    home = tmp_path / "home"
    config_path = _write_config(home)
    monkeypatch.setenv("HERMES_HOME", str(home))
    before = config_path.read_bytes()

    result = _call(key, value)

    assert result["success"] is False
    assert result["blocked"] is True
    assert config_path.read_bytes() == before


@pytest.mark.parametrize(
    "value",
    [
        "on",
        "off",
        "true",
        "false",
        "yes",
        "no",
        "ON",
        "False",
        "YeS",
    ],
)
def test_bool_like_tokens_are_rejected_for_bounded_string_keys_before_persistence(
    value: str, tmp_path, monkeypatch
):
    """Bool-like tokens must never reach the canonical writer, whose coercion
    for default-less keys could persist ``voice_id: true`` instead of the
    intended string. The rejection must happen at the validator (fail-closed),
    leave the file untouched, and never masquerade as a verification failure."""
    home = tmp_path / "home"
    config_path = _write_config(home)
    monkeypatch.setenv("HERMES_HOME", str(home))
    before = config_path.read_bytes()

    result = _call("tts.xai.voice_id", value)

    assert result["success"] is False
    assert result["blocked"] is True
    assert "boolean-like" in result["error"].lower()
    assert "could not be verified" not in result["error"].lower()
    assert config_path.read_bytes() == before
    audit = (home / "logs" / "config_changes.jsonl").read_text(encoding="utf-8")
    assert '"status": "invalid_value"' in audit
    assert '"rollback"' not in audit


def test_bool_like_rejection_does_not_affect_legitimate_voice_names(
    tmp_path, monkeypatch
):
    home = tmp_path / "home"
    _write_config(home)
    monkeypatch.setenv("HERMES_HOME", str(home))

    for value in ["alloy", "Kore", "eve", "voice-2-future"]:
        result = _call("tts.openai.voice", value)
        assert result["success"] is True
        assert result["key"] == "tts.openai.voice"


@pytest.mark.parametrize(
    ("key", "value", "applies"),
    [
        ("display.skin", "default", "new_session"),
        ("display.tool_progress", "off", "new_session"),
        ("compression.enabled", "false", "new_session"),
        ("stt.local.model", "tiny", "next_invocation"),
        ("tts.edge.voice", "en-US-AriaNeural", "next_invocation"),
    ],
)
def test_response_reports_per_setting_application_semantics(
    key: str, value: str, applies: str, tmp_path, monkeypatch
):
    home = tmp_path / "home"
    _write_config(home)
    monkeypatch.setenv("HERMES_HOME", str(home))

    result = _call(key, value)

    assert result["success"] is True
    assert result["applies"] == applies
    assert result["requires_process_restart"] is False


def test_prefix_collision_denial_with_secret_never_leaks_to_audit(
    tmp_path, monkeypatch
):
    home = tmp_path / "home"
    _write_config(home)
    monkeypatch.setenv("HERMES_HOME", str(home))
    secret = "sk-proj-AbCdEfGhIjKlMnOpQrStUvWxYz0123456789"

    result = _call("mcp_servers_v2.command", secret)

    assert result["success"] is False
    audit = (home / "logs" / "config_changes.jsonl").read_text(encoding="utf-8")
    assert secret not in audit


def test_structured_secret_attempt_is_rejected_and_redacted(tmp_path, monkeypatch):
    home = tmp_path / "home"
    _write_config(home)
    monkeypatch.setenv("HERMES_HOME", str(home))
    secret = "sk-proj-AbCdEfGhIjKlMnOpQrStUvWxYz0123456789"

    result = _call("tts.edge.voice", {"nested": {"token": secret}})

    assert result["success"] is False
    audit = (home / "logs" / "config_changes.jsonl").read_text(encoding="utf-8")
    assert secret not in audit


def test_denied_secret_key_is_redacted_in_audit(tmp_path, monkeypatch):
    home = tmp_path / "home"
    _write_config(home)
    monkeypatch.setenv("HERMES_HOME", str(home))
    secret_key = "sk-proj-AbCdEfGhIjKlMnOpQrStUvWxYz0123456789"

    result = _call(secret_key, "benign")

    assert result["success"] is False
    audit = (home / "logs" / "config_changes.jsonl").read_text(encoding="utf-8")
    assert secret_key not in audit


def test_writer_failure_cannot_return_unrelated_process_output(
    tmp_path, monkeypatch, capsys
):
    home = tmp_path / "home"
    _write_config(home)
    monkeypatch.setenv("HERMES_HOME", str(home))

    def noisy_failure(_key, _value):
        print("OTHER_SESSION_PRIVATE_OUTPUT")
        raise RuntimeError("private writer detail")

    monkeypatch.setattr("hermes_cli.config.set_config_value", noisy_failure)
    result = _call("display.show_reasoning", "true")

    assert result["success"] is False
    assert result["error"] == "Configuration write failed."
    assert "OTHER_SESSION_PRIVATE_OUTPUT" not in json.dumps(result)
    assert "OTHER_SESSION_PRIVATE_OUTPUT" in capsys.readouterr().out


def test_audit_bounds_malformed_values_and_rotates(tmp_path, monkeypatch):
    home = tmp_path / "home"
    _write_config(home)
    log_dir = home / "logs"
    log_dir.mkdir()
    log_path = log_dir / "config_changes.jsonl"
    log_path.write_bytes(b"x" * 1_000_000)
    monkeypatch.setenv("HERMES_HOME", str(home))

    result = _call("unknown." + "k" * 20_000, {"items": ["v" * 20_000] * 50})

    assert result["success"] is False
    assert (log_dir / "config_changes.jsonl.1").stat().st_size == 1_000_000
    assert log_path.stat().st_size < 10_000


def test_audit_bounds_many_key_mapping(tmp_path, monkeypatch):
    home = tmp_path / "home"
    _write_config(home)
    monkeypatch.setenv("HERMES_HOME", str(home))

    result = _call("unknown", {f"key-{index}": "value" for index in range(50_000)})

    assert result["success"] is False
    audit = (home / "logs" / "config_changes.jsonl").read_text(encoding="utf-8")
    assert "key-19" in audit
    assert "key-20" not in audit
    assert len(audit) < 10_000


def test_audit_redacts_and_bounds_session_id(tmp_path, monkeypatch):
    home = tmp_path / "home"
    _write_config(home)
    monkeypatch.setenv("HERMES_HOME", str(home))
    secret = "sk-proj-AbCdEfGhIjKlMnOpQrStUvWxYz0123456789"

    result = _call("unknown", "value", session_id=secret + "x" * 20_000)

    assert result["success"] is False
    audit = (home / "logs" / "config_changes.jsonl").read_text(encoding="utf-8")
    assert secret not in audit
    assert len(audit) < 2_000


@pytest.mark.parametrize("key", sorted(WRITABLE_CONFIG_KEYS))
def test_each_allowed_setting_persists_with_accurate_application_semantics(
    key: str, tmp_path, monkeypatch
):
    home = tmp_path / "home"
    _write_config(home)
    monkeypatch.setenv("HERMES_HOME", str(home))

    result = _call(key, _VALID_VALUE_BY_KEY[key])

    assert result["success"] is True
    if key.startswith(("stt.", "tts.")):
        assert result["applies"] == "next_invocation"
    else:
        assert result["applies"] == "new_session"
    assert result["requires_process_restart"] is False


# ---------------------------------------------------------------------------
# Finding 2: a failed verification must not leave the config mutated.
# ---------------------------------------------------------------------------


def test_verification_failure_restores_original_value_when_key_existed(
    tmp_path, monkeypatch
):
    """Case A: the key existed before the failed write; rollback must restore
    the exact previous value and never report success."""
    home = tmp_path / "home"
    config_path = _write_config(home, "tts:\n  openai:\n    voice: alloy\n")
    monkeypatch.setenv("HERMES_HOME", str(home))
    _install_mismatched_writer(monkeypatch)

    result = _call("tts.openai.voice", "alloy")

    assert result["success"] is False
    assert result["rollback"] == "restored"
    assert "restored" in result["error"]
    assert "could not be verified" in result["error"]
    parsed = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert parsed == {"tts": {"openai": {"voice": "alloy"}}}
    audit = (home / "logs" / "config_changes.jsonl").read_text(encoding="utf-8")
    assert '"status": "verification_failed"' in audit
    assert '"rollback": "restored"' in audit


def test_verification_failure_removes_newly_created_leaf_and_parents(
    tmp_path, monkeypatch
):
    """Case B: the key did not exist before the failed write; rollback must
    remove the created leaf and leave no empty parent mappings behind."""
    home = tmp_path / "home"
    config_path = _write_config(home, "display:\n  skin: default\n")
    monkeypatch.setenv("HERMES_HOME", str(home))
    _install_mismatched_writer(monkeypatch)

    result = _call("tts.openai.voice", "alloy")

    assert result["success"] is False
    assert result["rollback"] == "restored"
    parsed = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert parsed == {"display": {"skin": "default"}}
    assert "tts" not in parsed


def test_verification_failure_rollback_preserves_unrelated_siblings(
    tmp_path, monkeypatch
):
    """Case C: rollback must restore the full pre-mutation file so unrelated
    sibling keys are never touched."""
    home = tmp_path / "home"
    config_path = _write_config(
        home,
        "display:\n  skin: default\n  show_reasoning: false\n"
        "compression:\n  enabled: true\n",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    _install_mismatched_writer(monkeypatch)

    result = _call("tts.edge.voice", "en-US-AriaNeural")

    assert result["success"] is False
    assert result["rollback"] == "restored"
    parsed = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert parsed == {
        "display": {"skin": "default", "show_reasoning": False},
        "compression": {"enabled": True},
    }


def test_rollback_failure_is_reported_distinctly(tmp_path, monkeypatch):
    """Case E: if the rollback write itself fails, the caller must get a
    distinct rollback_failed result — never a plain success, never a claim
    that the original state was restored."""
    home = tmp_path / "home"
    _write_config(home)
    monkeypatch.setenv("HERMES_HOME", str(home))
    _install_mismatched_writer(monkeypatch)

    import utils as utils_module

    calls = {"n": 0}
    real_write = utils_module.atomic_yaml_write

    def flaky_write(path, data, **kwargs):
        calls["n"] += 1
        if calls["n"] >= 2:  # first call = the mutation, second = the rollback
            raise OSError("simulated disk failure")
        return real_write(path, data, **kwargs)

    monkeypatch.setattr(utils_module, "atomic_yaml_write", flaky_write)

    result = _call("tts.openai.voice", "alloy")

    assert result["success"] is False
    assert result["rollback"] == "failed"
    assert "inconsistent" in result["error"].lower()
    audit = (home / "logs" / "config_changes.jsonl").read_text(encoding="utf-8")
    assert '"status": "rollback_failed"' in audit
    assert '"rollback": "failed"' in audit


def test_verification_failure_rollback_keeps_lock_serialization(tmp_path, monkeypatch):
    """Case D: rollback happens inside the same mutation-lock scope; concurrent
    tool calls stay serialized and each outcome is self-consistent."""
    home = tmp_path / "home"
    _write_config(home)
    monkeypatch.setenv("HERMES_HOME", str(home))
    _install_mismatched_writer(monkeypatch)

    from hermes_cli.config import set_config_value as canonical_writer

    counter_lock = threading.Lock()
    active = 0
    max_active = 0

    def observed_writer(key, value):
        nonlocal active, max_active
        with counter_lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.05)
        try:
            canonical_writer(key, value)
        finally:
            with counter_lock:
                active -= 1

    monkeypatch.setattr("hermes_cli.config.set_config_value", observed_writer)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(
            pool.map(
                lambda pair: _call(*pair),
                [
                    ("tts.openai.voice", "alloy"),  # fails + rolls back
                    ("display.show_reasoning", "true"),  # succeeds
                ],
            )
        )

    config = yaml.safe_load((home / "config.yaml").read_text(encoding="utf-8"))
    assert max_active == 1
    by_success = {result["success"] for result in results}
    assert by_success == {True, False}
    failing = next(result for result in results if result["success"] is False)
    assert failing["rollback"] == "restored"
    assert "tts" not in config
    assert config["display"]["show_reasoning"] is True


# ---------------------------------------------------------------------------
# Finding 3: audit dict keys must stay distinct, bounded, and non-leaky.
# ---------------------------------------------------------------------------


def test_audit_non_string_dict_keys_are_type_tagged():
    out = _safe_audit_value({1: "int", 1.5: "float", None: "none"})
    assert out == {"«int:1»": "int", "«float:1.5»": "float", "«none»": "none"}
    out_bool = _safe_audit_value({True: "bool"})
    assert out_bool == {"«bool:true»": "bool"}
    # a string key "1" must never merge with the int key 1
    out_mixed = _safe_audit_value({1: "int", "1": "str"})
    assert set(out_mixed) == {"«int:1»", "1"}


def test_audit_oversized_string_keys_stay_distinct_and_bounded():
    import hashlib
    import json

    k1 = "a" * 300
    k2 = "a" * 299 + "b"
    out = _safe_audit_value({k1: 1, k2: 2})
    keys = list(out)
    assert len(keys) == 2
    assert keys[0] != keys[1]  # no collapse onto one placeholder
    assert all(key.startswith("a" * 256) and "…«" in key for key in keys)
    # deterministic
    assert _safe_audit_value({k1: 1, k2: 2}) == out
    # bounded
    assert len(json.dumps(out, ensure_ascii=False)) < 2_000
    # fingerprint actually distinguishes the two keys
    d1 = hashlib.sha256(k1.encode("utf-8")).hexdigest()[:8]
    d2 = hashlib.sha256(k2.encode("utf-8")).hexdigest()[:8]
    assert d1 != d2
    assert d1 in keys[0] and d2 in keys[1]


def test_audit_nested_non_string_keys_are_tagged_and_bounded():
    import hashlib
    import json

    digest = hashlib.sha256(repr((1, 2)).encode("utf-8")).hexdigest()[:8]
    out = _safe_audit_value({"outer": {(1, 2): {"deep": "v"}, 42: "answer"}})
    inner = out["outer"]
    assert set(inner) == {f"«tuple:{digest}»", "«int:42»"}
    assert len(json.dumps(out, ensure_ascii=False)) < 2_000


def test_audit_dict_keys_never_leak_secret_shaped_input():
    import json

    secret = "sk-proj-AbCdEfGhIjKlMnOpQrStUvWxYz0123456789"
    out = _safe_audit_value({secret: 1})
    assert json.dumps(out, ensure_ascii=False) == '{"«redacted-secret»": 1}'


def test_audit_string_dict_keys_keep_existing_behavior():
    out = _safe_audit_value({"display": {"skin": "default"}})
    assert out == {"display": {"skin": "default"}}


# ---------------------------------------------------------------------------
# Round 2 (independent review): targeted rollback + CAS ownership.
# ---------------------------------------------------------------------------


def test_verification_failure_preserves_unrelated_change_added_after_snapshot(
    tmp_path, monkeypatch
):
    """An unrelated config key written by another writer after our write must
    survive the rollback. A full-snapshot restore would silently drop it; the
    targeted rollback must not."""
    home = tmp_path / "home"
    config_path = _write_config(home, "display:\n  skin: default\n")
    monkeypatch.setenv("HERMES_HOME", str(home))

    import hermes_cli.config as config_module
    from utils import atomic_yaml_write

    real_writer = config_module.set_config_value

    def writer_with_external_unrelated_add(key, value):
        # Our mutation write, then an unrelated writer adds compression.
        real_writer(key, value.upper())
        cfg = config_module.read_raw_config()
        cfg.setdefault("compression", {})["enabled"] = True
        atomic_yaml_write(config_module.get_config_path(), cfg, sort_keys=False)

    monkeypatch.setattr(
        config_module, "set_config_value", writer_with_external_unrelated_add
    )

    result = _call("tts.openai.voice", "alloy")

    assert result["success"] is False
    assert result["rollback"] == "restored"
    parsed = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert parsed == {
        "display": {"skin": "default"},
        "compression": {"enabled": True},
    }
    assert "tts" not in parsed


def test_verification_failure_does_not_overwrite_same_key_concurrent_change(
    tmp_path, monkeypatch
):
    """If another writer changes the SAME key between our verification and the
    rollback, the CAS ownership check must refuse to overwrite it."""
    home = tmp_path / "home"
    config_path = _write_config(home)
    monkeypatch.setenv("HERMES_HOME", str(home))
    _install_mismatched_writer(monkeypatch)

    import hermes_cli.config as config_module
    from utils import atomic_yaml_write

    real_read = config_module.read_raw_config
    calls = {"n": 0}

    def read_with_concurrent_same_key_change():
        calls["n"] += 1
        # Call order in _config_set_value_locked: 1 = pre-write capture,
        # 2 = post-write verification read, 3 = rollback re-read. Inject the
        # external same-key write right before the rollback re-read so the
        # rollback observes a value different from the one verification saw.
        if calls["n"] == 3:
            cfg = real_read()
            cfg["tts"] = {"openai": {"voice": "Kore"}}
            atomic_yaml_write(config_module.get_config_path(), cfg, sort_keys=False)
        return real_read()

    monkeypatch.setattr(
        config_module, "read_raw_config", read_with_concurrent_same_key_change
    )

    result = _call("tts.openai.voice", "alloy")

    assert result["success"] is False
    assert result["rollback"] == "failed"
    assert "not overwritten" in result["error"]
    parsed = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert parsed == {"tts": {"openai": {"voice": "Kore"}}}
    audit = (home / "logs" / "config_changes.jsonl").read_text(encoding="utf-8")
    assert '"status": "rollback_skipped"' in audit
    assert '"rollback": "failed"' in audit


def test_verification_failure_restores_missing_config_file_to_absent(
    tmp_path, monkeypatch
):
    """When config.yaml did not exist before the failed mutation, a successful
    rollback must leave no config.yaml behind (not an empty {} file)."""
    home = tmp_path / "home"
    home.mkdir(parents=True, exist_ok=True)
    config_path = home / "config.yaml"
    assert not config_path.exists()
    monkeypatch.setenv("HERMES_HOME", str(home))
    _install_mismatched_writer(monkeypatch)

    result = _call("tts.openai.voice", "alloy")

    assert result["success"] is False
    assert result["rollback"] == "restored"
    assert not config_path.exists()
    audit = (home / "logs" / "config_changes.jsonl").read_text(encoding="utf-8")
    assert '"status": "verification_failed"' in audit


def test_verification_failure_missing_file_keeps_concurrent_unrelated_add(
    tmp_path, monkeypatch
):
    """Even when the original config.yaml did not exist, an unrelated value
    added by another writer during the failed mutation must be preserved — the
    file must NOT be unlinked wholesale."""
    home = tmp_path / "home"
    home.mkdir(parents=True, exist_ok=True)
    config_path = home / "config.yaml"
    monkeypatch.setenv("HERMES_HOME", str(home))

    import hermes_cli.config as config_module
    from utils import atomic_yaml_write

    real_writer = config_module.set_config_value

    def writer_with_external_unrelated_add(key, value):
        real_writer(key, value.upper())
        cfg = config_module.read_raw_config()
        cfg.setdefault("display", {})["skin"] = "default"
        atomic_yaml_write(config_module.get_config_path(), cfg, sort_keys=False)

    monkeypatch.setattr(
        config_module, "set_config_value", writer_with_external_unrelated_add
    )

    result = _call("tts.openai.voice", "alloy")

    assert result["success"] is False
    assert result["rollback"] == "restored"
    assert config_path.exists()
    parsed = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert parsed == {"display": {"skin": "default"}}
    assert "tts" not in parsed


def test_targeted_rollback_prunes_only_empty_parents(tmp_path, monkeypatch):
    """Removing the failed leaf must not delete sibling keys under the same
    parent; only empty parent mappings are pruned."""
    home = tmp_path / "home"
    config_path = _write_config(home, "tts:\n  openai:\n    other_setting: keep-me\n")
    monkeypatch.setenv("HERMES_HOME", str(home))
    _install_mismatched_writer(monkeypatch)

    result = _call("tts.openai.voice", "alloy")

    assert result["success"] is False
    assert result["rollback"] == "restored"
    parsed = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert parsed == {"tts": {"openai": {"other_setting": "keep-me"}}}
    assert "voice" not in parsed["tts"]["openai"]


def test_audit_huge_numeric_key_is_bounded():
    """A gigantic integer dict key must not blow up audit size (str() of a
    100k-digit int raises in CPython 3.11+); it must stay bounded and
    deterministic."""
    import json

    huge = 10**100_000
    out = _safe_audit_value({huge: "value"})
    (key,) = out
    assert key.startswith("«int:")
    assert len(key) <= 64
    assert len(json.dumps(out, ensure_ascii=False)) < 2_000
    assert _safe_audit_value({huge: "value"}) == out  # deterministic
    # moderately huge (still over the primitive threshold) and normal keys
    big = 10**400
    assert len(list(_safe_audit_value({big: 1}))) == 1
    assert _safe_audit_value({1.5: "x"}) == {"«float:1.5»": "x"}


def test_writable_voice_write_skips_env_sync_and_completes_post_write_path(
    tmp_path, monkeypatch
):
    """Representative behavioral evidence for the §13 post-write investigation
    (classification A). This is NOT an exhaustive proof that no Python runtime
    exception can ever occur in the post-write code: it exercises one real
    writable write (tts.edge.voice) and proves the env-sync hook is never
    invoked for a non-terminal key, while the real post-write code path (cron
    drift no-op, skin-touch skip) completes successfully. The exhaustive claim
    rests on the code trace in the review package: env sync is gated to
    ``terminal.*`` keys, the cron drift warning returns early for non-model
    keys, and the display.skin touch is try/except-wrapped."""
    home = tmp_path / "home"
    _write_config(home)
    monkeypatch.setenv("HERMES_HOME", str(home))

    import hermes_cli.config as config_module

    calls = {"save_env": 0}

    def boom(*args, **kwargs):
        calls["save_env"] += 1
        raise AssertionError("save_env_value must not run for writable config keys")

    monkeypatch.setattr(config_module, "save_env_value", boom)

    # Real write: exercises the real env-sync gate, cron no-op, and skin-touch
    # path; must complete with success and without invoking save_env_value.
    result = _call("tts.edge.voice", "en-US-AriaNeural")

    assert result["success"] is True
    assert calls == {"save_env": 0}


# ---------------------------------------------------------------------------
# Round 3 (final repair): presence-aware CAS + signed huge-int audit keys.
# ---------------------------------------------------------------------------


def test_verification_failure_presence_change_missing_to_null_skips_rollback(
    tmp_path, monkeypatch
):
    """CAS must treat 'key absent' and 'key: null' as different states. When
    verification observed the key ABSENT and another writer later adds
    ``key: null``, the rollback must be skipped and the external null kept."""
    home = tmp_path / "home"
    config_path = _write_config(home)
    monkeypatch.setenv("HERMES_HOME", str(home))

    import hermes_cli.config as config_module
    from utils import atomic_yaml_write

    def writer_that_leaves_target_absent(key, value):
        # Failed write that never created the target leaf: persist an unrelated
        # change so the file moves while the target key stays absent.
        atomic_yaml_write(
            config_path, {"compression": {"enabled": False}}, sort_keys=False
        )

    monkeypatch.setattr(
        config_module, "set_config_value", writer_that_leaves_target_absent
    )

    real_read = config_module.read_raw_config
    calls = {"n": 0}

    def read_with_external_null_add():
        calls["n"] += 1
        # 1 = pre-write capture, 2 = verification read, 3 = rollback re-read.
        # External writer adds target key with explicit null before rollback.
        if calls["n"] == 3:
            cfg = real_read()
            cfg.setdefault("tts", {}).setdefault("openai", {})["voice"] = None
            atomic_yaml_write(config_module.get_config_path(), cfg, sort_keys=False)
        return real_read()

    monkeypatch.setattr(config_module, "read_raw_config", read_with_external_null_add)

    result = _call("tts.openai.voice", "alloy")

    assert result["success"] is False
    assert result["rollback"] == "failed"
    assert "not overwritten" in result["error"]
    parsed = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    # external null preserved; unrelated compression preserved
    assert parsed == {
        "compression": {"enabled": False},
        "tts": {"openai": {"voice": None}},
    }
    audit = (home / "logs" / "config_changes.jsonl").read_text(encoding="utf-8")
    assert '"status": "rollback_skipped"' in audit
    assert '"rollback": "failed"' in audit


def test_verification_failure_presence_change_null_to_missing_skips_rollback(
    tmp_path, monkeypatch
):
    """When verification observed the target key present with an explicit null
    value and another writer later REMOVES the key, the CAS must detect the
    presence change, skip the rollback, and not recreate the key."""
    home = tmp_path / "home"
    config_path = _write_config(home)
    monkeypatch.setenv("HERMES_HOME", str(home))

    import hermes_cli.config as config_module
    from utils import atomic_yaml_write

    def writer_that_persists_null(key, value):
        # Failed write persisted the target leaf as explicit YAML null.
        atomic_yaml_write(
            config_path, {"tts": {"openai": {"voice": None}}}, sort_keys=False
        )

    monkeypatch.setattr(config_module, "set_config_value", writer_that_persists_null)

    real_read = config_module.read_raw_config
    calls = {"n": 0}

    def read_with_external_key_removal():
        calls["n"] += 1
        # 3 = rollback re-read: external writer removes the target key.
        if calls["n"] == 3:
            cfg = real_read()
            cfg.pop("tts", None)
            atomic_yaml_write(config_module.get_config_path(), cfg, sort_keys=False)
        return real_read()

    monkeypatch.setattr(
        config_module, "read_raw_config", read_with_external_key_removal
    )

    result = _call("tts.openai.voice", "alloy")

    assert result["success"] is False
    assert result["rollback"] == "failed"
    assert "not overwritten" in result["error"]
    parsed = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert "tts" not in parsed  # key stays absent; rollback did not recreate it
    audit = (home / "logs" / "config_changes.jsonl").read_text(encoding="utf-8")
    assert '"status": "rollback_skipped"' in audit
    assert '"rollback": "failed"' in audit


def test_audit_huge_negative_integer_key_is_bounded():
    """A huge negative integer dict key must not raise OverflowError from
    unsigned to_bytes(); it must stay bounded and deterministic, with a digest
    distinct from the same-magnitude positive integer."""
    import json

    huge_negative = -(10**100_000)
    out = _safe_audit_value({huge_negative: "value"})
    (key,) = out
    assert key.startswith("«int:")
    assert len(key) <= 64
    assert len(json.dumps(out, ensure_ascii=False)) < 2_000
    assert _safe_audit_value({huge_negative: "value"}) == out  # deterministic

    # Same magnitude, opposite sign → distinct digest.
    huge_positive = 10**100_000
    pos_out = _safe_audit_value({huge_positive: "value"})
    assert list(pos_out)[0] != key

    # Small signed ints keep the short tagged representation.
    assert _safe_audit_value({-1: "x"}) == {"«int:-1»": "x"}
    assert _safe_audit_value({42: "x"}) == {"«int:42»": "x"}
