"""Under ``gateway.multiplex_profiles`` one process serves every profile; each routed turn runs with
a context-local HERMES_HOME override. Tool-side state resolved once at import, or cached in a
single unkeyed slot, would hand the launch profile's paths/limits to every other profile.

Each test warms the site under profile A, flips the override to profile B with different
config, and asserts B sees its own values (real temp homes, real config.yaml, no mocks).
"""

import json

import pytest

from hermes_constants import reset_hermes_home_override, set_hermes_home_override


@pytest.fixture
def two_profiles(tmp_path, monkeypatch):
    prof_a, prof_b = tmp_path / "profA", tmp_path / "profB"
    for home, limits in ((prof_a, (222, 33, 3300)), (prof_b, (888, 77, 7700))):
        home.mkdir()
        max_bytes, timeout, threshold = limits
        (home / "config.yaml").write_text(
            f"file_read_max_chars: {max_bytes}\ntool_output:\n  max_bytes: {max_bytes}\n"
            f"browser:\n  command_timeout: {timeout}\n  snapshot_threshold: {threshold}\n",
            encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(prof_a))
    return prof_a, prof_b


def _under(home, fn):
    token = set_hermes_home_override(str(home))
    try:
        return fn()
    finally:
        reset_hermes_home_override(token)


def test_checkpoint_and_snapshot_paths_follow_active_profile(two_profiles):
    import tools.process_registry as pr
    from tools.environments import modal, singularity

    prof_a, prof_b = two_profiles
    _under(prof_a, pr._checkpoint_path)  # warm under A
    assert _under(prof_b, pr._checkpoint_path) == prof_b / "processes.json"
    assert _under(prof_b, modal._snapshot_store) == prof_b / "modal_snapshots.json"
    assert _under(prof_b, singularity._snapshot_store) == prof_b / "singularity_snapshots.json"
    assert _under(prof_a, pr._checkpoint_path) == prof_a / "processes.json"


def test_config_caches_are_keyed_by_profile(two_profiles):
    import tools.browser_camofox as cam
    import tools.browser_tool as bt
    import tools.file_tools as ft
    import tools.tool_output_limits as tol
    from tools.browser_tool_lifecycle import cleanup_all_browsers

    prof_a, prof_b = two_profiles
    tol._reset_tool_output_limits_cache()
    cleanup_all_browsers()
    cam._cmd_timeout_resolved, cam._cached_cmd_timeout = False, None

    def read_all():
        return (tol.get_tool_output_limits()["max_bytes"], ft._get_max_read_chars(),
                bt._get_command_timeout(), bt.get_browser_snapshot_threshold(), cam._get_command_timeout())

    assert _under(prof_a, read_all) == (222, 222, 33, 3300, 33)
    assert _under(prof_b, read_all) == (888, 888, 77, 7700, 77)
    # Per-profile slots stay hot — switching back is not a single-slot ping-pong.
    assert _under(prof_a, read_all) == (222, 222, 33, 3300, 33)


def test_schema_path_hints_follow_active_profile(two_profiles):
    import tools.cronjob_tools  # noqa: F401  (registers cronjob_manage)
    import tools.skill_manager_tool  # noqa: F401
    import tools.tts_tool  # noqa: F401
    from tools.registry import registry

    prof_a, prof_b = two_profiles
    names = {"cronjob_manage", "text_to_speech", "skill_manage"}

    def definitions():
        return json.dumps(registry.get_definitions(names, quiet=True))

    for_a, for_b = _under(prof_a, definitions), _under(prof_b, definitions)
    assert "profA" in for_a and "profB" not in for_a
    assert "profB" in for_b and "profA" not in for_b
    for fn in json.loads(for_b):
        name, text = fn["function"]["name"], json.dumps(fn["function"])
        assert name in names and "profB" in text, name
