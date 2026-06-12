"""Regression tests for the Firecrawl web provider config resolver."""


def test_direct_config_reads_hermes_env_file(monkeypatch, tmp_path):
    """Firecrawl should see keys saved in Hermes' .env, not just exported env."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)
    monkeypatch.delenv("FIRECRAWL_API_URL", raising=False)
    (tmp_path / ".env").write_text(
        "FIRECRAWL_API_KEY=fc-test-key\n"
        "FIRECRAWL_API_URL=https://firecrawl.example.com/\n"
    )

    from plugins.web.firecrawl.provider import _get_direct_firecrawl_config

    config = _get_direct_firecrawl_config()

    assert config == (
        {
            "api_key": "fc-test-key",
            "api_url": "https://firecrawl.example.com",
        },
        ("direct", "https://firecrawl.example.com", "fc-test-key"),
    )


def test_direct_config_falls_back_to_process_env_when_config_loader_unavailable(monkeypatch):
    """Keep plugin import/use robust if hermes_cli.config cannot be imported."""
    monkeypatch.setenv("FIRECRAWL_API_KEY", "fc-process-key")
    monkeypatch.setenv("FIRECRAWL_API_URL", "https://firecrawl.process/")

    import plugins.web.firecrawl.provider as firecrawl_provider

    real_import = firecrawl_provider.__builtins__["__import__"]

    def fake_import(name, *args, **kwargs):
        if name == "hermes_cli.config":
            raise ImportError("config unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setitem(firecrawl_provider.__builtins__, "__import__", fake_import)

    config = firecrawl_provider._get_direct_firecrawl_config()

    assert config == (
        {
            "api_key": "fc-process-key",
            "api_url": "https://firecrawl.process",
        },
        ("direct", "https://firecrawl.process", "fc-process-key"),
    )


def test_direct_config_unset_when_neither_env_source_has_firecrawl(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)
    monkeypatch.delenv("FIRECRAWL_API_URL", raising=False)
    (tmp_path / ".env").write_text("OTHER_KEY=value\n")

    from plugins.web.firecrawl.provider import _get_direct_firecrawl_config

    assert _get_direct_firecrawl_config() is None
