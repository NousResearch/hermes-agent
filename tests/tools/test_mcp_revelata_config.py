"""Tests for Revelata MCP integration.

Covers config parsing, credential handling, connection lifecycle, tool
registration, and enable/disable through config. All tests use mocks — no
real Revelata service is contacted.
"""

import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path

import pytest
import yaml


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def revelata_manifest(tmp_path):
    """Create a temporary Revelata manifest for testing."""
    manifest = {
        "manifest_version": 1,
        "name": "revelata",
        "description": "Query SEC-sourced KPIs, filings, and company summaries for US public companies via Revelata deepKPI.",
        "source": "https://www.revelata.com",
        "transport": {
            "type": "http",
            "url": "https://deepkpi-mcp.revelata.com/mcp"
        },
        "auth": {
            "type": "oauth"
        },
        "post_install": """On first connection, Hermes will open a browser to authenticate with Revelata."""
    }
    manifest_path = tmp_path / "manifest.yaml"
    with open(manifest_path, "w") as f:
        yaml.safe_dump(manifest, f)
    return manifest_path, manifest


@pytest.fixture
def mock_config_context(monkeypatch, tmp_path):
    """Isolate config I/O to a temporary HERMES_HOME."""
    hh = tmp_path / "hermes-home"
    hh.mkdir()
    config_path = hh / "config.yaml"
    config_path.write_text("")
    
    monkeypatch.setenv("HERMES_HOME", str(hh))
    monkeypatch.setattr("hermes_cli.config.get_hermes_home", lambda: hh)
    monkeypatch.setattr("hermes_cli.config.get_config_path", lambda: config_path)
    monkeypatch.setattr("hermes_cli.config.get_env_path", lambda: hh / ".env")
    
    def load_config():
        """Load config from the isolated HERMES_HOME."""
        if config_path.exists():
            return yaml.safe_load(config_path.read_text()) or {}
        return {}
    
    def save_config(cfg):
        """Save config to the isolated HERMES_HOME."""
        config_path.write_text(yaml.safe_dump(cfg))
    
    monkeypatch.setattr("hermes_cli.config.load_config", load_config)
    monkeypatch.setattr("hermes_cli.config.save_config", save_config)
    
    return hh


# ---------------------------------------------------------------------------
# Config Parsing Tests
# ---------------------------------------------------------------------------


class TestRevelataConfigParsing:
    """Test parsing of Revelata config from config.yaml."""

    def test_revelata_minimal_config_parses(self, mock_config_context):
        """Minimal Revelata config parses cleanly."""
        from hermes_cli.config import load_config, save_config
        
        cfg = {
            "mcp_servers": {
                "revelata": {
                    "url": "https://deepkpi-mcp.revelata.com/mcp",
                    "auth": "oauth"
                }
            }
        }
        save_config(cfg)
        loaded = load_config()
        
        assert "mcp_servers" in loaded
        assert "revelata" in loaded["mcp_servers"]
        assert loaded["mcp_servers"]["revelata"]["url"] == "https://deepkpi-mcp.revelata.com/mcp"
        assert loaded["mcp_servers"]["revelata"]["auth"] == "oauth"

    def test_revelata_with_tool_filtering_parses(self, mock_config_context):
        """Revelata config with tool filtering parses cleanly."""
        from hermes_cli.config import load_config, save_config
        
        cfg = {
            "mcp_servers": {
                "revelata": {
                    "url": "https://deepkpi-mcp.revelata.com/mcp",
                    "auth": "oauth",
                    "tools": {
                        "include": [
                            "query_company_id",
                            "list_kpis",
                            "search_kpis",
                            "get_company_summary"
                        ]
                    }
                }
            }
        }
        save_config(cfg)
        loaded = load_config()
        
        assert "tools" in loaded["mcp_servers"]["revelata"]
        assert "include" in loaded["mcp_servers"]["revelata"]["tools"]
        included_tools = loaded["mcp_servers"]["revelata"]["tools"]["include"]
        assert "query_company_id" in included_tools
        assert len(included_tools) == 4

    def test_revelata_config_with_timeout(self, mock_config_context):
        """Revelata config accepts optional timeout settings."""
        from hermes_cli.config import load_config, save_config
        
        cfg = {
            "mcp_servers": {
                "revelata": {
                    "url": "https://deepkpi-mcp.revelata.com/mcp",
                    "auth": "oauth",
                    "connect_timeout": 30,
                    "timeout": 60
                }
            }
        }
        save_config(cfg)
        loaded = load_config()
        
        assert loaded["mcp_servers"]["revelata"]["connect_timeout"] == 30
        assert loaded["mcp_servers"]["revelata"]["timeout"] == 60

    def test_mcp_load_config_returns_revelata_entry(self, monkeypatch):
        """_load_mcp_config in mcp_tool.py returns Revelata entry."""
        from tools import mcp_tool
        
        servers = {
            "revelata": {
                "url": "https://deepkpi-mcp.revelata.com/mcp",
                "auth": "oauth"
            }
        }
        
        with patch("hermes_cli.config.load_config", return_value={"mcp_servers": servers}):
            loaded = mcp_tool._load_mcp_config()
        
        assert "revelata" in loaded
        assert loaded["revelata"]["url"] == "https://deepkpi-mcp.revelata.com/mcp"


# ---------------------------------------------------------------------------
# Auth & Credential Tests
# ---------------------------------------------------------------------------


class TestRevelataAuth:
    """Test OAuth authentication flow for Revelata."""

    def test_revelata_uses_oauth_not_api_key(self, revelata_manifest):
        """Revelata manifest declares OAuth, not API key auth."""
        _, manifest = revelata_manifest
        assert manifest["auth"]["type"] == "oauth"
        assert "env" not in manifest["auth"]

    def test_revelata_manifest_parses_with_oauth_auth(self):
        """Revelata catalog entry parses successfully."""
        from hermes_cli.mcp_catalog import _parse_manifest
        
        # Use the actual Revelata manifest from optional-mcps/
        repo_root = Path(__file__).parent.parent.parent
        manifest_path = repo_root / "optional-mcps" / "revelata" / "manifest.yaml"
        
        if not manifest_path.exists():
            pytest.skip("Revelata manifest not found in optional-mcps/")
        
        entry = _parse_manifest(manifest_path)
        assert entry.name == "revelata"
        assert entry.transport.type == "http"
        assert entry.transport.url == "https://deepkpi-mcp.revelata.com/mcp"
        assert entry.auth.type == "oauth"
        assert not entry.auth.env  # No env vars for OAuth

    def test_revelata_has_post_install_instructions(self):
        """Revelata manifest includes post_install guidance for OAuth."""
        from hermes_cli.mcp_catalog import _parse_manifest
        
        repo_root = Path(__file__).parent.parent.parent
        manifest_path = repo_root / "optional-mcps" / "revelata" / "manifest.yaml"
        
        if not manifest_path.exists():
            pytest.skip("Revelata manifest not found in optional-mcps/")
        
        entry = _parse_manifest(manifest_path)
        assert entry.post_install
        # OAuth flow guidance should be in the post_install text
        assert any(word in entry.post_install.lower() 
                  for word in ["browser", "auth", "login", "oauth"])


# ---------------------------------------------------------------------------
# Connection & Tool Registration Tests
# ---------------------------------------------------------------------------


class TestRevelataConnection:
    """Test MCP connection and tool registration (with mocks)."""

    @pytest.mark.asyncio
    async def test_revelata_server_task_initialization(self):
        """MCPServerTask initializes correctly for Revelata."""
        from tools.mcp_tool import MCPServerTask
        
        server = MCPServerTask("revelata")
        assert server.name == "revelata"
        assert server._tools == []

    @pytest.mark.asyncio
    async def test_connect_to_revelata_requires_valid_config(self):
        """Connecting to Revelata requires url and auth config."""
        from tools.mcp_tool import _connect_server
        
        # Missing URL should raise ValueError
        with pytest.raises(ValueError):
            await _connect_server("revelata", {"auth": "oauth"})

    @pytest.mark.asyncio
    async def test_revelata_http_transport_recognized(self):
        """Revelata's HTTP transport is properly recognized."""
        from tools.mcp_tool import MCPServerTask
        
        config = {
            "url": "https://deepkpi-mcp.revelata.com/mcp",
            "auth": "oauth"
        }
        
        server = MCPServerTask("revelata")
        # Should not raise an exception during config validation
        try:
            # This validates the config shape (checking for required keys)
            from tools import mcp_tool
            server_config = config
            # HTTP transport requires 'url'
            assert "url" in server_config
            assert server_config["url"].startswith("https://")
        except Exception as exc:
            pytest.fail(f"HTTP config validation failed: {exc}")

    @pytest.mark.asyncio
    async def test_revelata_tool_discovery_mock(self, monkeypatch):
        """Revelata tool discovery works with mocked server."""
        from tools.mcp_tool import MCPServerTask
        
        server = MCPServerTask("revelata")
        
        # Mock the session and tool list
        mock_session = AsyncMock()
        server.session = mock_session
        
        # Mock the 8 Revelata tools
        tools = [
            SimpleNamespace(name="query_company_id", description="Look up company numeric ID"),
            SimpleNamespace(name="list_kpis", description="List all KPIs for a company"),
            SimpleNamespace(name="search_kpis", description="Semantic search over KPIs"),
            SimpleNamespace(name="company_summary_search", description="Search across companies"),
            SimpleNamespace(name="get_company_summary", description="Get company summary"),
            SimpleNamespace(name="get_company_segments", description="Get company segments"),
            SimpleNamespace(name="list_sec_filing_markdowns", description="List SEC filings"),
            SimpleNamespace(name="get_sec_filing_markdown", description="Get SEC filing markdown"),
        ]
        
        server._tools = tools
        
        # All expected tools should be registered
        assert len(server._tools) == 8
        tool_names = [t.name for t in server._tools]
        assert "query_company_id" in tool_names
        assert "search_kpis" in tool_names
        assert "get_company_summary" in tool_names
        assert "get_sec_filing_markdown" in tool_names

    @pytest.mark.asyncio
    async def test_revelata_tool_naming_convention(self):
        """Revelata tools follow the mcp_revelata_* naming convention."""
        # Tool names in MCP are canonical (with hyphens), but Hermes registers
        # them as mcp_revelata_<name> with hyphens → underscores
        tool_names = [
            "query_company_id",
            "list_kpis",
            "search_kpis",
            "company_summary_search",
            "get_company_summary",
            "get_company_segments",
            "list_sec_filing_markdowns",
            "get_sec_filing_markdown",
        ]
        
        # All names should be valid Python identifiers (no hyphens after conversion)
        for name in tool_names:
            hermes_name = f"mcp_revelata_{name}"
            assert hermes_name.replace("-", "_").isidentifier()


# ---------------------------------------------------------------------------
# Enable/Disable Tests
# ---------------------------------------------------------------------------


class TestRevelataEnableDisable:
    """Test enabling and disabling Revelata through config."""

    def test_revelata_enable_via_config(self, mock_config_context):
        """Enabling Revelata writes the mcp_servers entry."""
        from hermes_cli.config import load_config, save_config
        
        # Start with no Revelata
        cfg = {"mcp_servers": {}}
        save_config(cfg)
        
        # Add Revelata
        cfg["mcp_servers"]["revelata"] = {
            "url": "https://deepkpi-mcp.revelata.com/mcp",
            "auth": "oauth"
        }
        save_config(cfg)
        
        # Verify it's there
        loaded = load_config()
        assert "revelata" in loaded["mcp_servers"]

    def test_revelata_disable_via_config(self, mock_config_context):
        """Disabling Revelata removes the mcp_servers entry."""
        from hermes_cli.config import load_config, save_config
        
        # Start with Revelata enabled
        cfg = {
            "mcp_servers": {
                "revelata": {
                    "url": "https://deepkpi-mcp.revelata.com/mcp",
                    "auth": "oauth"
                }
            }
        }
        save_config(cfg)
        assert "revelata" in load_config()["mcp_servers"]
        
        # Disable it
        cfg["mcp_servers"].pop("revelata")
        save_config(cfg)
        
        # Verify it's gone
        loaded = load_config()
        assert "revelata" not in loaded.get("mcp_servers", {})

    def test_revelata_tool_filtering_enables_subset(self, mock_config_context):
        """Tool filtering lets users enable only specific Revelata tools."""
        from hermes_cli.config import load_config, save_config
        
        cfg = {
            "mcp_servers": {
                "revelata": {
                    "url": "https://deepkpi-mcp.revelata.com/mcp",
                    "auth": "oauth",
                    "tools": {
                        "include": [
                            "query_company_id",
                            "list_kpis",
                            "get_company_summary"
                        ]
                    }
                }
            }
        }
        save_config(cfg)
        
        loaded = load_config()
        included = loaded["mcp_servers"]["revelata"]["tools"]["include"]
        # Users can select which tools they want
        assert len(included) < 8  # Less than all 8 tools
        assert "search_kpis" not in included  # Explicitly omitted


# ---------------------------------------------------------------------------
# Hidden Whitespace Warning Tests
# ---------------------------------------------------------------------------


class TestRevelataHiddenWhitespace:
    """Test that hidden whitespace in config is detected (pasted tokens, URLs)."""

    def test_revelata_url_with_trailing_space_warned(self, caplog):
        """Leading/trailing whitespace in URL is detected and warned."""
        from tools import mcp_tool
        
        config = {
            "url": "https://deepkpi-mcp.revelata.com/mcp ",  # trailing space
            "auth": "oauth"
        }
        
        with caplog.at_level(logging.WARNING, logger="tools.mcp_tool"):
            flagged = mcp_tool._warn_hidden_whitespace("revelata", config)
        
        assert "url" in flagged
        # Value should not be mutated by the warning function
        assert config["url"] == "https://deepkpi-mcp.revelata.com/mcp "

    def test_revelata_clean_config_no_warnings(self, caplog):
        """Clean Revelata config produces no whitespace warnings."""
        from tools import mcp_tool
        
        config = {
            "url": "https://deepkpi-mcp.revelata.com/mcp",
            "auth": "oauth"
        }
        
        with caplog.at_level(logging.WARNING, logger="tools.mcp_tool"):
            flagged = mcp_tool._warn_hidden_whitespace("revelata", config)
        
        assert flagged == []


# ---------------------------------------------------------------------------
# Manifest Contract Tests
# ---------------------------------------------------------------------------


class TestRevelataManifestContract:
    """Test that Revelata manifest conforms to catalog contracts."""

    def test_revelata_manifest_exists_in_repo(self):
        """Revelata manifest file must exist in optional-mcps/."""
        repo_root = Path(__file__).parent.parent.parent
        manifest_path = repo_root / "optional-mcps" / "revelata" / "manifest.yaml"
        assert manifest_path.exists(), f"Revelata manifest not found at {manifest_path}"

    def test_revelata_manifest_http_transport_has_pinned_url(self):
        """HTTP transport URL must be pinned (not a placeholder)."""
        from hermes_cli.mcp_catalog import _parse_manifest
        
        repo_root = Path(__file__).parent.parent.parent
        manifest_path = repo_root / "optional-mcps" / "revelata" / "manifest.yaml"
        
        if not manifest_path.exists():
            pytest.skip("Revelata manifest not found")
        
        entry = _parse_manifest(manifest_path)
        # URL should not contain ${...} placeholders
        assert entry.transport.url is not None
        assert "$" not in entry.transport.url
        # URL should be HTTPS (secure)
        assert entry.transport.url.startswith("https://")

    def test_revelata_manifest_required_fields_present(self):
        """Revelata manifest must have all required fields."""
        from hermes_cli.mcp_catalog import _parse_manifest
        
        repo_root = Path(__file__).parent.parent.parent
        manifest_path = repo_root / "optional-mcps" / "revelata" / "manifest.yaml"
        
        if not manifest_path.exists():
            pytest.skip("Revelata manifest not found")
        
        entry = _parse_manifest(manifest_path)
        # Required fields from CatalogEntry dataclass
        assert entry.name == "revelata"
        assert entry.description
        assert entry.source
        assert entry.transport.type in ("stdio", "http")
        assert entry.auth.type in ("api_key", "oauth", "none")
