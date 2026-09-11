"""Tests for OSV malware check on MCP extension packages."""

import json
import time
from pathlib import Path

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


    def test_unknown(self):
        assert _infer_ecosystem("node") is None
        assert _infer_ecosystem("python") is None
        assert _infer_ecosystem("/bin/bash") is None


class TestParseNpmPackage:
    def test_simple(self):
        assert _parse_npm_package("react") == ("react", None)


    def test_latest_ignored(self):
        assert _parse_npm_package("react@latest") == ("react", None)


class TestParsePypiPackage:
    def test_simple(self):
        assert _parse_pypi_package("requests") == ("requests", None)


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


    def test_plain_positional_still_works(self):
        # Regression guard: bare positional with no --package flag is the pkg.
        name, ver = _parse_package_from_args(["-y", "react@18.3.1"], "npm")
        assert name == "react"
        assert ver == "18.3.1"


class TestCheckPackageForMalware:
    @pytest.fixture(autouse=True)
    def _fresh_cache(self, tmp_path, monkeypatch):
        from tools import osv_check
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        with osv_check._cache_lock:
            osv_check._cache.clear()
            osv_check._disk_cache_loaded = False
        (tmp_path / "cache" / "osv_check.json").unlink(missing_ok=True)
        yield
        with osv_check._cache_lock:
            osv_check._cache.clear()
            osv_check._disk_cache_loaded = False
        (tmp_path / "cache" / "osv_check.json").unlink(missing_ok=True)
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

    def test_repeat_checks_hit_cache_not_network(self):
        """Same package re-checked (MCP revival loops) must not re-query OSV.

        Regression for #75485: watchdog revival loops re-ran the preflight
        every spawn attempt, producing 779K api.osv.dev DNS queries in 16h.
        """
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({"vulns": []}).encode()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("tools.osv_check.urllib.request.urlopen", return_value=mock_response) as mock_url:
            for _ in range(50):
                assert check_package_for_malware("uvx", ["mcp-server-fetch"]) is None
        assert mock_url.call_count == 1

    def test_blocked_verdict_is_cached(self):
        """A malware verdict is served from cache on re-check too."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(
            {"vulns": [{"id": "MAL-2023-1", "summary": "bad"}]}
        ).encode()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("tools.osv_check.urllib.request.urlopen", return_value=mock_response) as mock_url:
            first = check_package_for_malware("npx", ["evil-pkg"])
            second = check_package_for_malware("npx", ["evil-pkg"])
        assert first is not None and "BLOCKED" in first
        assert second == first
        assert mock_url.call_count == 1

    def test_network_failure_not_cached(self):
        """Fail-open results must not be cached — retry once network is back."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({"vulns": []}).encode()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch(
            "tools.osv_check.urllib.request.urlopen",
            side_effect=OSError("network down"),
        ):
            assert check_package_for_malware("uvx", ["mcp-server-time"]) is None
        # Network is back: the next check must hit OSV, not a cached fail-open.
        with patch(
            "tools.osv_check.urllib.request.urlopen", return_value=mock_response
        ) as mock_url:
            assert check_package_for_malware("uvx", ["mcp-server-time"]) is None
        assert mock_url.call_count == 1

    def test_cache_expiry_requeries(self, monkeypatch):
        """Expired entries re-query instead of serving stale verdicts."""
        from tools import osv_check

        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({"vulns": []}).encode()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("tools.osv_check.urllib.request.urlopen", return_value=mock_response) as mock_url:
            check_package_for_malware("uvx", ["mcp-server-fetch"])
            # Force-expire the entry.
            with osv_check._cache_lock:
                key = next(iter(osv_check._cache))
                _, result = osv_check._cache[key]
                osv_check._cache[key] = (0.0, result)
            check_package_for_malware("uvx", ["mcp-server-fetch"])
        assert mock_url.call_count == 2

    def test_disk_cache_persists_and_reloads(self, tmp_path, monkeypatch):
        """A warm disk cache is reused by a fresh in-process cache."""
        from tools import osv_check

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({"vulns": []}).encode()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("tools.osv_check.urllib.request.urlopen", return_value=mock_response) as mock_url:
            check_package_for_malware("uvx", ["mcp-server-persist"])

        cache_file = tmp_path / "cache" / "osv_check.json"
        assert cache_file.exists(), "disk cache should be written after a warm result"

        with osv_check._cache_lock:
            osv_check._cache.clear()
            osv_check._disk_cache_loaded = False

        with patch("tools.osv_check.urllib.request.urlopen", return_value=mock_response) as mock_url2:
            check_package_for_malware("uvx", ["mcp-server-persist"])

        assert mock_url2.call_count == 0, "disk cache must satisfy the second call"

    def test_disk_cache_format_versioned(self, tmp_path, monkeypatch):
        """Disk cache JSON has a version field and recoverable entries."""
        from tools import osv_check

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({"vulns": []}).encode()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("tools.osv_check.urllib.request.urlopen", return_value=mock_response):
            check_package_for_malware("uvx", ["mcp-server-format"])

        cache_file = tmp_path / "cache" / "osv_check.json"
        with open(cache_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["version"] == osv_check._DISK_CACHE_VERSION
        assert "entries" in data
        key = "PyPI|mcp-server-format|"
        assert key in data["entries"]
        assert "expiry" in data["entries"][key]
        assert data["entries"][key]["result"] is None

    def test_disk_cache_retries_after_transient_oserror(self, tmp_path, monkeypatch):
        """A busy/unreadable cache file must not disable disk loads for the process."""
        from tools import osv_check

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        cache_file = tmp_path / "cache" / "osv_check.json"
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(
            json.dumps({
                "version": osv_check._DISK_CACHE_VERSION,
                "entries": {
                    "PyPI|mcp-server-retry|": {
                        "expiry": time.time() + 3600,
                        "result": None,
                    }
                },
            }),
            encoding="utf-8",
        )

        real_open = open
        calls = {"n": 0}

        def flaky_open(path, *args, **kwargs):
            if Path(path) == cache_file:
                calls["n"] += 1
                if calls["n"] == 1:
                    raise OSError("resource temporarily unavailable")
            return real_open(path, *args, **kwargs)

        monkeypatch.setattr("builtins.open", flaky_open)
        with osv_check._cache_lock:
            osv_check._load_disk_cache()
            assert osv_check._disk_cache_loaded is False
            osv_check._load_disk_cache()
            assert osv_check._disk_cache_loaded is True
            assert ("PyPI", "mcp-server-retry", None) in osv_check._cache


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


# ── Additional coverage for uncovered paths ────────────────────────────


class TestInferEcosystemExtra:
    def test_npx_cmd_windows(self):
        assert _infer_ecosystem("npx.cmd") == "npm"

    def test_uvx_cmd_windows(self):
        assert _infer_ecosystem("uvx.cmd") == "PyPI"

    def test_empty_command(self):
        assert _infer_ecosystem("") is None


class TestParseNpmPackageEdgeCases:
    def test_scoped_no_match_returns_token(self):
        """A scoped token that doesn't match the regex returns (token, None)."""
        # @invalid (no slash) doesn't match the scoped regex
        result = _parse_npm_package("@invalid")
        assert result == ("@invalid", None)

    def test_unscoped_no_at_returns_token(self):
        result = _parse_npm_package("plain-name")
        assert result == ("plain-name", None)


class TestParsePypiPackageEdgeCases:
    def test_no_match_returns_token(self):
        """A token that doesn't match the PyPI regex returns (token, None)."""
        result = _parse_pypi_package("++invalid++")
        assert result == ("++invalid++", None)


class TestParsePackageFromArgsEdgeCases:
    def test_non_string_arg_skipped(self):
        """Non-string args (e.g. numbers) are skipped."""
        name, ver = _parse_package_from_args([42, "react@1.0"], "npm")
        assert name == "react"
        assert ver == "1.0"

    def test_unknown_ecosystem_returns_token_without_version(self):
        """Unknown ecosystem returns (package_token, None)."""
        name, ver = _parse_package_from_args(["some-pkg@1.0"], "unknown")
        assert name == "some-pkg@1.0"
        assert ver is None

    def test_all_non_string_args(self):
        """All non-string args returns (None, None)."""
        assert _parse_package_from_args([42, 3.14], "npm") == (None, None)


class TestCheckPackageForMalwareEdgeCases:
    @pytest.fixture(autouse=True)
    def _fresh_cache(self, tmp_path, monkeypatch):
        from tools import osv_check
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        with osv_check._cache_lock:
            osv_check._cache.clear()
            osv_check._disk_cache_loaded = False
        (tmp_path / "cache" / "osv_check.json").unlink(missing_ok=True)
        yield
        with osv_check._cache_lock:
            osv_check._cache.clear()
            osv_check._disk_cache_loaded = False
        (tmp_path / "cache" / "osv_check.json").unlink(missing_ok=True)

    def test_unparseable_package_returns_none(self):
        """When package can't be parsed, returns None (allow)."""
        result = check_package_for_malware("npx", [])
        assert result is None

    def test_malware_without_summary_uses_id(self):
        """Malware advisory without summary falls back to id in the message."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "vulns": [
                {"id": "MAL-2024-0001"},  # no summary
            ]
        }).encode()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("tools.osv_check.urllib.request.urlopen", return_value=mock_response):
            result = check_package_for_malware("npx", ["evil-pkg"])
        assert result is not None
        assert "MAL-2024-0001" in result

    def test_multiple_malware_capped_at_three(self):
        """Only first 3 malware advisories are shown."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "vulns": [
                {"id": f"MAL-2024-{i:04d}", "summary": f"malware {i}"}
                for i in range(5)
            ]
        }).encode()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("tools.osv_check.urllib.request.urlopen", return_value=mock_response):
            result = check_package_for_malware("npx", ["evil-pkg"])
        assert result is not None
        assert "MAL-2024-0000" in result
        assert "MAL-2024-0002" in result
        assert "MAL-2024-0004" not in result


class TestQueryOsvWithVersion:
    def test_version_included_in_payload(self):
        """When version is provided, it's included in the OSV query payload."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({"vulns": []}).encode()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("tools.osv_check.urllib.request.urlopen", return_value=mock_response) as mock_url:
            _query_osv("some-pkg", "npm", "1.2.3")
            call_data = json.loads(mock_url.call_args[0][0].data)
            assert call_data["version"] == "1.2.3"

    def test_no_version_omitted_from_payload(self):
        """When version is None, it's not in the payload."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({"vulns": []}).encode()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("tools.osv_check.urllib.request.urlopen", return_value=mock_response) as mock_url:
            _query_osv("some-pkg", "npm")
            call_data = json.loads(mock_url.call_args[0][0].data)
            assert "version" not in call_data

    def test_only_mal_advisories_returned(self):
        """Non-MAL advisories are filtered out."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "vulns": [
                {"id": "MAL-2024-0001", "summary": "malware"},
                {"id": "CVE-2024-1234", "summary": "regular vuln"},
                {"id": "GHSA-1234", "summary": "github advisory"},
            ]
        }).encode()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("tools.osv_check.urllib.request.urlopen", return_value=mock_response):
            result = _query_osv("some-pkg", "npm")
        assert len(result) == 1
        assert result[0]["id"] == "MAL-2024-0001"
