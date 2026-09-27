"""Composer effort pins must not shadow per-model config overrides.

Regression shape (2026-09, AssemblyAI gateway): the Desktop composer seeds its effort
atom from the profile's global ``agent.reasoning_effort`` and ships that value on every
``session.create``. Treated as a pin, it defeated ``agent.reasoning_overrides`` — a
model configured in config.yaml to run effort "none" with tools got the global "high"
on the wire and the provider 500'd. An effort that merely restates the global default
is not a user choice and must not pin; only a differing value is a real override.
"""
import pytest


@pytest.fixture
def overrides():
    """The bind_module-rebound copy (server globals), as production dispatches it."""
    import tui_gateway.server as srv

    fn = getattr(srv, "_create_overrides", None)
    if fn is None:  # direct module call in isolation (no server import side effects)
        import tui_gateway.methods_session as methods_session

        return methods_session._create_overrides
    return fn


@pytest.fixture
def cfg_effort(monkeypatch):
    """Stub server._load_cfg's agent.reasoning_effort (the profile default)."""
    def _install(global_effort):
        import tui_gateway.server as srv

        monkeypatch.setattr(srv, "_load_cfg", lambda: {"agent": {"reasoning_effort": global_effort}}, raising=False)

    return _install


def test_mirror_of_global_default_is_not_a_pin(overrides, cfg_effort):
    cfg_effort("high")
    _, reasoning, _ = overrides({"reasoning_effort": "high"})
    assert reasoning is None  # config resolution (incl. per-model overrides) decides


def test_differing_effort_is_a_real_pin(overrides, cfg_effort):
    cfg_effort("high")
    _, reasoning, _ = overrides({"reasoning_effort": "low"})
    assert reasoning == {"enabled": True, "effort": "low"}


def test_none_differs_from_global_effort_is_a_pin(overrides, cfg_effort):
    cfg_effort("high")
    _, reasoning, _ = overrides({"reasoning_effort": "none"})
    assert reasoning == {"enabled": False}


def test_case_variant_of_default_is_not_a_pin(overrides, cfg_effort):
    cfg_effort("high")
    _, reasoning, _ = overrides({"reasoning_effort": "HIGH"})
    assert reasoning is None


def test_no_global_configured_pin_still_applies(overrides, cfg_effort):
    cfg_effort("")  # profile has no agent.reasoning_effort
    _, reasoning, _ = overrides({"reasoning_effort": "medium"})
    assert reasoning == {"enabled": True, "effort": "medium"}


def test_alias_disabled_config_vs_none_payload_is_echo(overrides, cfg_effort):
    """config 'disabled' and composer 'none' are the same state — echo, not a pin."""
    cfg_effort("disabled")
    _, reasoning, _ = overrides({"reasoning_effort": "none"})
    assert reasoning is None


def test_alias_yaml_false_config_vs_none_payload_is_echo(overrides, cfg_effort):
    """YAML `reasoning_effort: false` loads as Python False; 'none' means the same."""
    cfg_effort(False)
    _, reasoning, _ = overrides({"reasoning_effort": "none"})
    assert reasoning is None


def test_whitespace_variant_of_default_is_echo(overrides, cfg_effort):
    cfg_effort(" medium ")  # parse_reasoning_effort strips
    _, reasoning, _ = overrides({"reasoning_effort": "medium"})
    assert reasoning is None


def test_model_override_and_service_tier_untouched(overrides, cfg_effort):
    cfg_effort("high")
    model, reasoning, tier = overrides(
        {"model": "gpt-5.6-sol", "provider": "assemblyai", "reasoning_effort": "high", "fast": True})
    assert model == {"model": "gpt-5.6-sol", "provider": "assemblyai"}
    assert reasoning is None
    assert tier == "priority"


def _other_profile(tmp_path, effort: str):
    """A real secondary-profile home (its own config.yaml) — the create TARGETS this profile."""
    home = tmp_path / "other-profile"
    home.mkdir()
    (home / "config.yaml").write_text(f"agent:\n  reasoning_effort: {effort}\n")
    return home


def test_secondary_profile_echo_is_not_a_pin(tmp_path, overrides, cfg_effort):
    """App-global remote mode: one backend serves several profiles. The shipped value is the TARGET
    profile's default, so it is an echo even though the LAUNCH profile's default differs."""
    cfg_effort("high")  # launch profile default
    other = _other_profile(tmp_path, "low")
    _, reasoning, _ = overrides({"profile": "other-profile", "reasoning_effort": "low"}, other)
    assert reasoning is None


def test_pick_matching_launch_global_in_another_profile_is_a_pin(tmp_path, overrides, cfg_effort):
    """The comparison must read the TARGET profile's config: a deliberate pick that differs from the
    target's default stays a pin even when it happens to equal the launch profile's default."""
    cfg_effort("high")  # launch profile default — must NOT decide this
    other = _other_profile(tmp_path, "low")
    _, reasoning, _ = overrides({"profile": "other-profile", "reasoning_effort": "high"}, other)
    assert reasoning == {"enabled": True, "effort": "high"}
