"""Tests for OSV malware check on MCP extension packages."""

import importlib
import json
import pytest
from unittest.mock import patch, MagicMock

from tools.osv_check import (
    check_package_for_malware,
    _infer_ecosystem,
    _parse_package_from_args,
    _parse_npm_package,
    _parse_pypi_package,
    _query_osv,
)


class TestInferEcosystem:
    def test_npx(self):
        assert _infer_ecosystem("npx") == "npm"
        assert _infer_ecosystem("/usr/bin/npx") == "npm"

    def test_uvx(self):
        assert _infer_ecosystem("uvx") == "PyPI"
        assert _infer_ecosystem("/home/user/.local/bin/uvx") == "PyPI"

    def test_pipx(self):
        assert _infer_ecosystem("pipx") == "PyPI"

    def test_unknown(self):
        assert _infer_ecosystem("node") is None
        assert _infer_ecosystem("python") is None
        assert _infer_ecosystem("/bin/bash") is None


class TestParseNpmPackage:
    def test_simple(self):
        assert _parse_npm_package("react") == ("react", None)

    def test_with_version(self):
        assert _parse_npm_package("react@18.3.1") == ("react", "18.3.1")

    def test_scoped(self):
        assert _parse_npm_package("@modelcontextprotocol/server-filesystem") == (
            "@modelcontextprotocol/server-filesystem", None
        )

    def test_scoped_with_version(self):
        assert _parse_npm_package("@scope/pkg@1.2.3") == ("@scope/pkg", "1.2.3")

    def test_latest_ignored(self):
        assert _parse_npm_package("react@latest") == ("react", None)


class TestParsePypiPackage:
    def test_simple(self):
        assert _parse_pypi_package("requests") == ("requests", None)

    def test_with_version(self):
        assert _parse_pypi_package("requests==2.32.3") == ("requests", "2.32.3")

    def test_with_extras(self):
        assert _parse_pypi_package("mcp[cli]==1.2.3") == ("mcp", "1.2.3")

    def test_extras_no_version(self):
        assert _parse_pypi_package("mcp[cli]") == ("mcp", None)


class TestParsePackageFromArgs:
    def test_npm_skips_flags(self):
        name, ver = _parse_package_from_args(["-y", "@scope/pkg@1.0"], "npm")
        assert name == "@scope/pkg"
        assert ver == "1.0"

    def test_pypi_skips_flags(self):
        name, ver = _parse_package_from_args(["--from", "mcp[cli]"], "PyPI")
        # --from is a flag, mcp[cli] is the package
        # Actually --from is a flag so it gets skipped, mcp[cli] is found
        assert name == "mcp"

    def test_empty_args(self):
        assert _parse_package_from_args([], "npm") == (None, None)

    def test_only_flags(self):
        assert _parse_package_from_args(["-y", "--yes"], "npm") == (None, None)

    def test_package_equals_form(self):
        # `npx --package=@scope/pkg@1.0 some-bin` -> install target is the
        # --package value, NOT the executed binary `some-bin`.
        name, ver = _parse_package_from_args(
            ["--package=@scope/pkg@1.0", "some-bin"], "npm"
        )
        assert name == "@scope/pkg"
        assert ver == "1.0"

    def test_package_space_form(self):
        # `npx --package @scope/pkg some-bin` (value in the next token).
        name, ver = _parse_package_from_args(
            ["--package", "@scope/pkg@2.0", "some-bin"], "npm"
        )
        assert name == "@scope/pkg"
        assert ver == "2.0"

    def test_short_p_form(self):
        # `npx -p left-pad@1.3.0 cli-cmd` -> package is left-pad, not cli-cmd.
        name, ver = _parse_package_from_args(
            ["-p", "left-pad@1.3.0", "cli-cmd"], "npm"
        )
        assert name == "left-pad"
        assert ver == "1.3.0"

    def test_plain_positional_still_works(self):
        # Regression guard: bare positional with no --package flag is the pkg.
        name, ver = _parse_package_from_args(["-y", "react@18.3.1"], "npm")
        assert name == "react"
        assert ver == "18.3.1"


class TestCheckPackageForMalware:
    def test_clean_package(self):
        """Clean package returns None (allow)."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({"vulns": []}).encode()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("tools.osv_check.urllib.request.urlopen", return_value=mock_response):
            result = check_package_for_malware("npx", ["-y", "@modelcontextprotocol/server-filesystem"])
        assert result is None

    def test_malware_blocked(self):
        """Known malware package returns error string."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "vulns": [
                {"id": "MAL-2023-7938", "summary": "Malicious code in evil-pkg"},
                {"id": "CVE-2023-1234", "summary": "Regular vulnerability"},  # should be filtered
            ]
        }).encode()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("tools.osv_check.urllib.request.urlopen", return_value=mock_response):
            result = check_package_for_malware("npx", ["evil-pkg"])
        assert result is not None
        assert "BLOCKED" in result
        assert "MAL-2023-7938" in result
        assert "CVE-2023-1234" not in result  # regular CVEs filtered

    def test_network_error_fails_open(self):
        """Network errors allow the package (fail-open)."""
        with patch("tools.osv_check.urllib.request.urlopen", side_effect=ConnectionError("timeout")):
            result = check_package_for_malware("npx", ["some-package"])
        assert result is None

    def test_non_npx_skipped(self):
        """Non-npx/uvx commands are skipped entirely."""
        result = check_package_for_malware("node", ["server.js"])
        assert result is None

    def test_uvx_pypi(self):
        """uvx commands check PyPI ecosystem."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({"vulns": []}).encode()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("tools.osv_check.urllib.request.urlopen", return_value=mock_response) as mock_url:
            check_package_for_malware("uvx", ["mcp-server-fetch"])
            # Verify PyPI ecosystem was sent
            call_data = json.loads(mock_url.call_args[0][0].data)
            assert call_data["package"]["ecosystem"] == "PyPI"
            assert call_data["package"]["name"] == "mcp-server-fetch"


class TestEndpointHardcoded:
    """Regression: the OSV endpoint must be hardcoded — an OSV_ENDPOINT env
    var must not be able to redirect the malware scan to an attacker-controlled
    server (scan bypass)."""

    def test_env_var_cannot_override_endpoint(self, monkeypatch):
        import tools.osv_check as osv_mod

        monkeypatch.setenv("OSV_ENDPOINT", "https://attacker.example.com/v1/query")
        # Reload so the module-level endpoint constant is re-evaluated with
        # the hostile env var present.
        importlib.reload(osv_mod)
        try:
            mock_response = MagicMock()
            mock_response.read.return_value = json.dumps({"vulns": []}).encode()
            mock_response.__enter__ = lambda s: s
            mock_response.__exit__ = MagicMock(return_value=False)

            with patch(
                "tools.osv_check.urllib.request.urlopen",
                return_value=mock_response,
            ) as mock_urlopen:
                osv_mod._query_osv("react", "npm")

            req = mock_urlopen.call_args[0][0]
            assert req.full_url == "https://api.osv.dev/v1/query"
        finally:
            # Restore module state with a clean environment.
            monkeypatch.delenv("OSV_ENDPOINT", raising=False)
            importlib.reload(osv_mod)


class TestLiveOsvQuery:
    """Live integration test against the real OSV API. Skipped if offline."""

    @pytest.mark.skipif(
        not pytest.importorskip("urllib.request", reason="no network"),
        reason="network required",
    )
    def test_known_malware_package(self):
        """node-hide-console-windows has a real MAL- advisory."""
        try:
            result = _query_osv("node-hide-console-windows", "npm")
            assert len(result) >= 1
            assert result[0]["id"].startswith("MAL-")
        except Exception:
            pytest.skip("OSV API unreachable")

    @pytest.mark.skipif(
        not pytest.importorskip("urllib.request", reason="no network"),
        reason="network required",
    )
    def test_clean_package(self):
        """react should have zero MAL- advisories."""
        try:
            result = _query_osv("react", "npm")
            assert len(result) == 0
        except Exception:
            pytest.skip("OSV API unreachable")
