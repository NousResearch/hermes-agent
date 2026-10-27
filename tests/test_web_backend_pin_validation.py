"""A web backend pin must be validated against what plugins actually declare (#145698).

The pin is strict: when the named provider is absent from the process registry, every
web_search/web_extract call fails instead of degrading. Config validation is the only place that
can say so before the first call, so it must not stay silent about a name nobody provides.
"""
import sys
import types

import pytest

from hermes_cli import config_value_checks


@pytest.fixture
def issues():
    return []


def _messages(issues):
    return " | ".join(i.message for i in issues)


def test_unknown_backend_name_warns(issues, monkeypatch):
    monkeypatch.setattr(config_value_checks, "_declared_web_backends", lambda: {"firecrawl": "web-firecrawl"})
    config_value_checks._validate_web_backends({"web": {"extract_backend": "firecarwl"}}, issues)
    assert issues, "a backend nobody provides must be reported"
    assert "firecarwl" in _messages(issues)
    assert issues[0].severity == "warning"


def test_declared_backend_is_quiet_when_registry_is_not_loaded(issues, monkeypatch):
    """Cold start: plugins load after config validation, so silence is correct here."""
    monkeypatch.setattr(config_value_checks, "_declared_web_backends", lambda: {"firecrawl": "web-firecrawl"})
    monkeypatch.delitem(sys.modules, "agent.web_search_registry", raising=False)
    config_value_checks._validate_web_backends({"web": {"extract_backend": "firecrawl"}}, issues)
    assert issues == []


def test_declared_but_unregistered_provider_warns_to_restart(issues, monkeypatch):
    """The #145698 case: plugin installed and enabled, serving process never loaded it."""
    monkeypatch.setattr(config_value_checks, "_declared_web_backends",
                        lambda: {"local-extract": "web-local-extract"})
    fake = types.ModuleType("agent.web_search_registry")
    fake.get_provider = lambda name: None
    monkeypatch.setitem(sys.modules, "agent.web_search_registry", fake)
    config_value_checks._validate_web_backends({"web": {"extract_backend": "local-extract"}}, issues)
    assert issues, "an unregistered provider of an installed plugin must be reported"
    text = _messages(issues)
    assert "local-extract" in text and "web-local-extract" in text
    assert "restart" in text.lower()
    assert issues[0].severity == "warning"


def test_registered_provider_is_quiet(issues, monkeypatch):
    monkeypatch.setattr(config_value_checks, "_declared_web_backends",
                        lambda: {"local-extract": "web-local-extract"})
    fake = types.ModuleType("agent.web_search_registry")
    fake.get_provider = lambda name: object()
    monkeypatch.setitem(sys.modules, "agent.web_search_registry", fake)
    config_value_checks._validate_web_backends({"web": {"extract_backend": "local-extract"}}, issues)
    assert issues == []


def test_empty_selection_is_not_checked(issues, monkeypatch):
    monkeypatch.setattr(config_value_checks, "_declared_web_backends", dict)
    config_value_checks._validate_web_backends({"web": {"backend": "", "extract_backend": ""}}, issues)
    assert issues == []


def test_declared_backends_reads_manifests(tmp_path, monkeypatch):
    """The manifest scan is the only source of truth; it must read both layouts."""
    (tmp_path / "hermes_cli").mkdir()
    bundled = tmp_path / "plugins" / "web" / "demo"
    bundled.mkdir(parents=True)
    (bundled / "plugin.yaml").write_text(
        "name: web-demo\nprovides_web_providers:\n  - demo\n  - demo-alt\nkind: backend\n",
        encoding="utf-8")
    installed = tmp_path / "plugins" / "local"
    installed.mkdir(parents=True)
    (installed / "plugin.yaml").write_text(
        "name: local\nprovides_web_providers:\n  - local-extract\n", encoding="utf-8")
    unrelated = tmp_path / "plugins" / "other"
    unrelated.mkdir(parents=True)
    (unrelated / "plugin.yaml").write_text("name: other\nkind: tool\n", encoding="utf-8")
    monkeypatch.setattr(config_value_checks, "__file__",
                        str(tmp_path / "hermes_cli" / "config_value_checks.py"))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    declared = config_value_checks._declared_web_backends()
    assert declared.get("demo") == "demo"
    assert declared.get("demo-alt") == "demo"
    assert declared.get("local-extract") == "local"
