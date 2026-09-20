"""The provider-default reasoning value is enabled without a manufactured effort."""

from hermes_constants import parse_reasoning_effort


def test_auto_is_a_neutral_explicit_reasoning_config():
    assert parse_reasoning_effort("auto") == {"enabled": True}
