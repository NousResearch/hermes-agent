"""Opt-in bundled plugins remain discoverable in Desktop's backend inventory."""
from tui_gateway.methods_tools import _plugin_rows


def _bundled(root, name, kind):
    plugin = root / name
    plugin.mkdir(parents=True)
    (plugin / "plugin.yaml").write_text(f"name: {name}\nversion: 1.0.0\nkind: {kind}\n", encoding="utf-8")
    (plugin / "__init__.py").write_text("def register(ctx):\n    pass\n", encoding="utf-8")


def test_plugin_rows_publish_activation_default_separately_from_current_status(tmp_path, monkeypatch):
    bundled = tmp_path / "bundled"
    _bundled(bundled, "sample-target", "standalone")
    _bundled(bundled, "sample-backend", "backend")
    monkeypatch.setenv("HERMES_BUNDLED_PLUGINS", str(bundled))
    rows = {row["name"]: row for row in _plugin_rows()}
    row = rows["sample-target"]
    assert row["source"] == "bundled"
    assert row.get("default_enabled") is False
    assert row["status"] == "not enabled"
    assert rows["sample-backend"].get("default_enabled") is True
