#!/usr/bin/env python3
"""
Plugin Guard — Security scanner for externally-installed plugins.

Inspired by Claude Cowork's skill & plugin security scanning (announced
2026-08-06: third-party skills and plugins are automatically checked for
malicious content when someone uploads or edits them, returning pass /
warn / fail). Hermes already scans hub-installed *skills* via
``tools/skills_guard.py``; this module extends the same static-analysis
engine to ``hermes plugins install`` and ``hermes plugins update``, which
previously cloned and executed arbitrary Git repositories unscanned.

Plugins are strictly more dangerous than skills — they run Python
in-process with the agent — but they are also *expected* to do things a
skill never should: read their own API keys from environment variables
(the documented ``requires_env`` pattern), call provider HTTP APIs with
those keys, and spawn subprocesses. A naive reuse of the skill threat
patterns would flag every legitimate provider plugin. So this scanner:

- Runs the full skills_guard pattern set on documentation/config files
  (README, after-install.md, plugin.yaml, ...), where prompt-injection
  and social-engineering content lives.
- Exempts the "reads own env secret" / "HTTP call with key" pattern
  family on *code* files, while keeping genuinely malicious signals:
  foreign credential-store access (~/.ssh, ~/.aws, ~/.hermes/.env),
  reverse shells, destructive commands, persistence mechanisms,
  obfuscated execution, and known exfiltration services.
- Applies plugin-sized structural limits and skips VCS/venv noise.

Verdict → install policy (Cowork's pass/warn/fail, adapted):

- ``safe``      → install normally.
- ``caution``   → warn; requires explicit confirmation (interactive
                  prompt, ``--force``, or a caller-supplied decision
                  callback).
- ``dangerous`` → blocked. ``--force`` does NOT override.

Usage:
    from tools.plugin_guard import scan_plugin, should_allow_plugin_install

    result = scan_plugin(Path("/tmp/clone/my-plugin"), source="owner/repo")
    allowed, reason = should_allow_plugin_install(result)
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

from tools.skills_guard import (
    Finding,
    ScanResult,
    SUSPICIOUS_BINARY_EXTENSIONS,
    _determine_verdict,
    format_scan_report,
    scan_file,
)

PLUGIN_SCANNER_VERSION = "plugin-guard-v1"

# Directories that are never scanned (VCS internals, caches, vendored envs).
EXCLUDED_DIRS = {
    ".git", "__pycache__", "node_modules", ".venv", "venv",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", ".tox",
}

# Code file extensions where "reads an env secret" / "HTTP call with a key
# variable" is the NORMAL, documented plugin pattern (provider plugins read
# their own API keys via requires_env and call their backend with them).
CODE_FILE_EXTENSIONS = {
    ".py", ".js", ".ts", ".sh", ".bash", ".rb", ".pl", ".php",
}

# Pattern ids from skills_guard.THREAT_PATTERNS that are exempt on code
# files. Each of these describes behavior every legitimate provider plugin
# exhibits. They still apply in full to documentation and config files,
# where such content is a strong injection/social-engineering signal.
CODE_EXEMPT_PATTERN_IDS = {
    "python_environ_get_secret",
    "python_getenv_secret",
    "python_os_environ",
    "node_process_env",
    "ruby_env_secret",
    "env_exfil_httpx",
    "env_exfil_requests",
    "env_exfil_fetch",
    "env_exfil_curl",
    "env_exfil_wget",
    # Agent-facing instruction patterns are meaningless inside code
    # (docstrings/comments about prompts trip them constantly).
    "context_exfil",
    "send_to_url",
    "fake_policy",
    # Plugins legitimately write their own settings into config.yaml during
    # post_setup, and encode credentials (e.g. HTTP Basic auth) with base64.
    "agent_config_mod",
    "encoded_exfil",
}

# Findings whose severity is remapped for plugins. Skills treat any bundled
# binary as critical (a skill is documentation and should never ship one);
# plugin repos occasionally vendor a compiled artifact legitimately, so a
# binary is a warn-tier signal instead of an instant block.
#
# ``hermes_env_access`` (a reference to ``~/.hermes/.env``) is the DOCUMENTED
# way plugins tell users where to put their API keys — nearly every legit
# plugin README mentions it. A mere reference is informational for plugins;
# actually READING the file still trips ``read_secrets_file`` (critical).
# ``curl | sh`` install instructions are common in plugin READMEs; keep them
# at warn tier (caution) rather than an unoverridable block.
SEVERITY_REMAP = {
    "binary_file": "high",
    "hermes_env_access": "medium",
    "curl_pipe_shell": "high",
}

# Structural limits — plugins are real codebases, far larger than skills.
MAX_PLUGIN_FILE_COUNT = 400
MAX_PLUGIN_TOTAL_SIZE_KB = 10 * 1024   # 10MB of scannable tree
MAX_PLUGIN_SINGLE_FILE_KB = 1024       # 1MB single file


def _is_excluded(rel_parts: Tuple[str, ...]) -> bool:
    return any(part in EXCLUDED_DIRS for part in rel_parts)


def _filter_findings(findings: List[Finding], rel_path: str) -> List[Finding]:
    """Apply plugin-specific exemptions and severity remaps to raw findings."""
    ext = Path(rel_path).suffix.lower()
    is_code = ext in CODE_FILE_EXTENSIONS
    out: List[Finding] = []
    for f in findings:
        if is_code and f.pattern_id in CODE_EXEMPT_PATTERN_IDS:
            continue
        remapped = SEVERITY_REMAP.get(f.pattern_id)
        if remapped:
            f.severity = remapped
        out.append(f)
    return out


def _check_plugin_structure(plugin_dir: Path) -> List[Finding]:
    """Structural checks sized for plugin repositories."""
    findings: List[Finding] = []
    file_count = 0
    total_size = 0

    for f in plugin_dir.rglob("*"):
        try:
            rel_parts = f.relative_to(plugin_dir).parts
        except ValueError:
            continue
        if _is_excluded(rel_parts):
            continue
        rel = "/".join(rel_parts)

        if f.is_symlink():
            file_count += 1
            try:
                resolved = f.resolve()
                if not resolved.is_relative_to(plugin_dir.resolve()):
                    findings.append(Finding(
                        pattern_id="symlink_escape",
                        severity="critical",
                        category="traversal",
                        file=rel,
                        line=0,
                        match=f"symlink -> {resolved}",
                        description="symlink points outside the plugin directory",
                    ))
            except OSError:
                findings.append(Finding(
                    pattern_id="broken_symlink",
                    severity="medium",
                    category="traversal",
                    file=rel,
                    line=0,
                    match="broken symlink",
                    description="broken or circular symlink",
                ))
            continue

        if not f.is_file():
            continue
        file_count += 1

        try:
            size = f.stat().st_size
        except OSError:
            continue
        total_size += size

        if size > MAX_PLUGIN_SINGLE_FILE_KB * 1024:
            findings.append(Finding(
                pattern_id="oversized_file",
                severity="medium",
                category="structural",
                file=rel,
                line=0,
                match=f"{size // 1024}KB",
                description=(
                    f"file is {size // 1024}KB "
                    f"(limit: {MAX_PLUGIN_SINGLE_FILE_KB}KB)"
                ),
            ))

        ext = f.suffix.lower()
        if ext in SUSPICIOUS_BINARY_EXTENSIONS:
            findings.append(Finding(
                pattern_id="binary_file",
                severity=SEVERITY_REMAP.get("binary_file", "high"),
                category="structural",
                file=rel,
                line=0,
                match=f"binary: {ext}",
                description=(
                    f"binary/executable file ({ext}) bundled in plugin "
                    f"(cannot be scanned)"
                ),
            ))

    if file_count > MAX_PLUGIN_FILE_COUNT:
        findings.append(Finding(
            pattern_id="too_many_files",
            severity="medium",
            category="structural",
            file="(directory)",
            line=0,
            match=f"{file_count} files",
            description=(
                f"plugin has {file_count} files "
                f"(limit: {MAX_PLUGIN_FILE_COUNT})"
            ),
        ))
    if total_size > MAX_PLUGIN_TOTAL_SIZE_KB * 1024:
        findings.append(Finding(
            pattern_id="oversized_bundle",
            severity="medium",
            category="structural",
            file="(directory)",
            line=0,
            match=f"{total_size // 1024}KB",
            description=(
                f"plugin is {total_size // 1024}KB total "
                f"(limit: {MAX_PLUGIN_TOTAL_SIZE_KB}KB)"
            ),
        ))

    return findings


def scan_plugin(plugin_dir: Path, source: str = "") -> ScanResult:
    """Scan a plugin directory for security threats.

    Args:
        plugin_dir: Path to the plugin directory (typically the temp clone,
            before it is moved into ``~/.hermes/plugins/``).
        source: Identifier for display (git URL or owner/repo shorthand).

    Returns:
        ScanResult with verdict ``safe`` | ``caution`` | ``dangerous``.
        Every externally installed plugin is ``community`` trust.
    """
    all_findings: List[Finding] = []

    if plugin_dir.is_dir():
        all_findings.extend(_check_plugin_structure(plugin_dir))
        for f in sorted(plugin_dir.rglob("*")):
            if not f.is_file() or f.is_symlink():
                continue
            try:
                rel_parts = f.relative_to(plugin_dir).parts
            except ValueError:
                continue
            if _is_excluded(rel_parts):
                continue
            rel = "/".join(rel_parts)
            raw = scan_file(f, rel_path=rel)
            all_findings.extend(_filter_findings(raw, rel))

    verdict = _determine_verdict(all_findings)
    from datetime import datetime, timezone

    result = ScanResult(
        skill_name=plugin_dir.name,
        source=source or plugin_dir.name,
        trust_level="community",
        verdict=verdict,
        findings=all_findings,
        scanned_at=datetime.now(timezone.utc).isoformat(),
    )
    if all_findings:
        categories = {f.category for f in all_findings}
        result.summary = (
            f"{plugin_dir.name}: {verdict} — {len(all_findings)} finding(s) "
            f"in {', '.join(sorted(categories))}"
        )
    else:
        result.summary = f"{plugin_dir.name}: clean scan, no threats detected"
    result.scan_provenance = {
        "scanner_version": PLUGIN_SCANNER_VERSION,
        "verdict": verdict,
        "source": result.source,
    }
    return result


def should_allow_plugin_install(
    result: ScanResult,
    force: bool = False,
) -> Tuple[Optional[bool], str]:
    """Map a plugin scan verdict to an install decision.

    Returns ``(allowed, reason)``:
      - ``(True, ...)``  install proceeds.
      - ``(None, ...)``  needs explicit confirmation (caution verdict).
      - ``(False, ...)`` blocked; ``force`` never overrides ``dangerous``.
    """
    if result.verdict == "safe":
        return True, "Allowed (clean scan)"
    if result.verdict == "caution":
        if force:
            return True, (
                f"Force-installed despite caution verdict "
                f"({len(result.findings)} findings)"
            )
        return None, (
            f"Requires confirmation (caution verdict, "
            f"{len(result.findings)} findings)"
        )
    return False, (
        f"Blocked (dangerous verdict, {len(result.findings)} findings). "
        f"--force does not override a dangerous verdict."
    )


__all__ = [
    "scan_plugin",
    "should_allow_plugin_install",
    "format_scan_report",
    "PLUGIN_SCANNER_VERSION",
]
