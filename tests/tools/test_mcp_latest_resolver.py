"""${latest:<glob>} resolution in mcp_servers config (#153351).

Semantics under test (from the issue):
- newest version wins when path segments are version-like (sort -V semantics), else newest mtime;
- resolved at render time — i.e. per connection, not per gateway lifetime;
- no match keeps the literal placeholder (spawn fails loudly) and logs the unresolved
  pattern explicitly, instead of silently serving zero tools;
- composes with the existing ${VAR} interpolation in the same string.
"""

import os
import time

import pytest

from tools.mcp_tool_config import (
    _ENV_VAR_PATTERN,
    _interpolate_env_vars,
    _resolve_latest_pattern,
    _version_sort_key,
)


def _plugin_cache(tmp_path, *versions):
    """Mimic a Claude Code plugin cache: one directory per version, each holding the
    server entry script; the installer deletes old versions on upgrade."""
    for version in versions:
        scripts = tmp_path / "cache" / "thedotmack" / "claude-mem" / version / "scripts"
        scripts.mkdir(parents=True, exist_ok=True)
        (scripts / "mcp-server.cjs").write_text("console.log('ok')\n")
    return tmp_path / "cache" / "thedotmack" / "claude-mem"


class TestNewestMatchWins:
    def test_highest_semver_wins_regardless_of_mtime(self, tmp_path):
        cache = _plugin_cache(tmp_path, "15.13.0", "15.15.2")
        old = cache / "15.13.0" / "scripts" / "mcp-server.cjs"
        new = cache / "15.15.2" / "scripts" / "mcp-server.cjs"
        # An installer touching the OLD version must not outrank the higher number.
        os.utime(old, (time.time(), time.time() + 10_000))
        pattern = str(cache / "*" / "scripts" / "mcp-server.cjs")
        assert _resolve_latest_pattern(pattern) == str(new)

    def test_numeric_order_not_lexical(self, tmp_path):
        cache = _plugin_cache(tmp_path, "2.9.9", "2.10.0")
        pattern = str(cache / "*" / "scripts" / "mcp-server.cjs")
        assert _resolve_latest_pattern(pattern) == str(cache / "2.10.0" / "scripts" / "mcp-server.cjs")

    def test_mtime_fallback_for_non_versioned_paths(self, tmp_path):
        """No digits anywhere in the candidate names -> newest mtime wins outright,
        even when the lexically-greater name is the OLDER one (the reviewer's
        counterexample: 'zzz-old' must NOT beat 'aaa-new' just by sorting later)."""
        cache = _plugin_cache(tmp_path, "aaa-new", "zzz-old")
        newer = cache / "aaa-new" / "scripts" / "mcp-server.cjs"
        os.utime(newer, (time.time(), time.time() + 500))
        pattern = str(cache / "*" / "scripts" / "mcp-server.cjs")
        assert _resolve_latest_pattern(pattern) == str(newer)

    def test_versioned_beats_non_versioned_regardless_of_mtime(self, tmp_path):
        """Digit runs decide first: a version-bearing directory outranks a lexically
        later, non-versioned one even when the latter is newer."""
        cache = _plugin_cache(tmp_path, "15.13.0", "release")
        os.utime(cache / "release", (time.time(), time.time() + 10_000))
        pattern = str(cache / "*" / "scripts" / "mcp-server.cjs")
        assert _resolve_latest_pattern(pattern) == str(cache / "15.13.0" / "scripts" / "mcp-server.cjs")

    def test_version_sort_key_is_digit_runs(self):
        assert _version_sort_key("a/15.15.2/x") == (15, 15, 2)
        assert _version_sort_key("a/v10/x") > _version_sort_key("a/v2/x")
        assert _version_sort_key("a/2.10.0/x") > _version_sort_key("a/2.9.9/x")
        assert _version_sort_key("a/snapshot-b/x") == ()  # no digits: mtime territory


class TestRenderTimeResolution:
    def test_resolves_inside_args_list(self, tmp_path):
        cache = _plugin_cache(tmp_path, "15.13.0", "15.15.2")
        config = {"command": "/usr/bin/node",
                  "args": [f"${{latest:{cache / '*' / 'scripts' / 'mcp-server.cjs'}}}"]}
        rendered = _interpolate_env_vars(config)
        assert rendered["args"][0] == str(cache / "15.15.2" / "scripts" / "mcp-server.cjs")

    def test_re_render_after_upgrade_picks_new_version(self, tmp_path):
        """Per-connection semantics: the SAME template re-rendered after an upgrade
        resolves to the new version directory — no stale absolute path."""
        cache = _plugin_cache(tmp_path, "15.13.0")
        template = f"${{latest:{cache / '*' / 'scripts' / 'mcp-server.cjs'}}}"
        assert _interpolate_env_vars(template) == str(cache / "15.13.0" / "scripts" / "mcp-server.cjs")
        _plugin_cache(tmp_path, "15.16.0")  # upgrade: old dir kept, new one installed
        assert _interpolate_env_vars(template) == str(cache / "15.16.0" / "scripts" / "mcp-server.cjs")

    def test_composes_with_env_ref_in_same_config(self, tmp_path, monkeypatch):
        """${VAR} and ${latest:...} resolve independently within one config — each
        ref is expanded in place; nesting refs inside one another is not supported."""
        cache = _plugin_cache(tmp_path, "15.15.2")
        monkeypatch.setenv("MCP_HOME", str(tmp_path))
        config = {"command": "/usr/bin/node",
                  "args": ["--home", "${MCP_HOME}",
                           f"${{latest:{cache / '*' / 'scripts' / 'mcp-server.cjs'}}}"]}
        rendered = _interpolate_env_vars(config)
        assert rendered["args"][1] == str(tmp_path)
        assert rendered["args"][2] == str(cache / "15.15.2" / "scripts" / "mcp-server.cjs")


class TestNoMatchFailsLoudly:
    def test_no_match_keeps_literal_and_marks_missed(self, tmp_path, caplog):
        missing = str(tmp_path / "nope" / "*" / "server.cjs")
        rendered = _interpolate_env_vars(f"${{latest:{missing}}}")
        assert rendered == f"${{latest:{missing}}}"
        assert getattr(rendered, "missed", False) is True

    def test_no_match_warns_the_pattern_once(self, tmp_path, caplog):
        import logging

        missing = str(tmp_path / "never" / "*" / "server.cjs")
        template = f"${{latest:{missing}}}"
        with caplog.at_level(logging.WARNING, logger="tools.mcp_tool"):
            _interpolate_env_vars(template)
            _interpolate_env_vars(template)  # config loads repeat per discovery pass
        warnings = [r for r in caplog.records if "matched no existing path" in r.message]
        assert len(warnings) == 1  # once per process, naming the pattern
        assert missing in warnings[0].getMessage()

    def test_no_match_inside_dict_config(self, tmp_path):
        missing = str(tmp_path / "void" / "*" / "s.cjs")
        config = {"command": "/usr/bin/node", "args": [f"${{latest:{missing}}}"], "enabled": True}
        rendered = _interpolate_env_vars(config)
        assert rendered["args"][0] == f"${{latest:{missing}}}"


class TestNonGlobExactPath:
    def test_single_exact_match_resolves_verbatim(self, tmp_path):
        _plugin_cache(tmp_path, "15.15.2")
        exact = tmp_path / "cache" / "thedotmack" / "claude-mem" / "15.15.2" / "scripts" / "mcp-server.cjs"
        assert _resolve_latest_pattern(str(exact)) == str(exact)


class TestPatternGuardrails:
    def test_plain_env_refs_unaffected(self, monkeypatch):
        monkeypatch.setenv("PLAIN", "value")
        assert _interpolate_env_vars("${PLAIN}") == "value"

    def test_pattern_reach_is_the_braced_ref(self):
        # The resolver only ever sees text between ${ and } — never a bare glob.
        m = _ENV_VAR_PATTERN.search("prefix ${latest:/a/*/b} suffix")
        assert m is not None
        assert m.group(1) == "latest:/a/*/b"
