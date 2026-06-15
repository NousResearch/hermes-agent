from toolsets import TOOLSETS, resolve_toolset


def test_slack_history_toolset_is_configurable_and_in_messaging():
    assert "slack" in TOOLSETS
    assert "slack_history" in TOOLSETS["slack"]["tools"]
    assert "slack_history" in TOOLSETS["messaging"]["tools"]


def test_default_core_toolset_includes_slack_history():
    assert "slack_history" in resolve_toolset("hermes-cli")
