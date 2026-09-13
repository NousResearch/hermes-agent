"""Opt-in bundled plugins remain discoverable in Desktop's backend inventory."""
from tui_gateway.methods_tools import _plugin_rows


def test_plugin_rows_publish_activation_default_separately_from_current_status():
    row = next(row for row in _plugin_rows() if row['name'] == 'hermes-realms')
    assert row['source'] == 'bundled'
    assert row.get('default_enabled') is False
    assert row['status'] == 'not enabled'
