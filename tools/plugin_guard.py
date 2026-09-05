#!/usr/bin/env python3
"""Plugin Guard — ``skills_guard`` engine applied to ``hermes plugins install``/``update``.

Plugins run in-process but are *expected* to read their own env keys, call provider APIs
and spawn subprocesses, so: full pattern set on docs/config files (where prompt-injection
lives); the "reads own secret"/"HTTP call with key" family exempt on *code* files;
plugin-sized structural limits; VCS/venv noise skipped. ``safe`` installs, ``caution``
needs confirmation, ``dangerous`` is blocked and ``--force`` does NOT override.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

from tools.skills_guard import (
    Finding, ScanResult, SUSPICIOUS_BINARY_EXTENSIONS, _determine_verdict, format_scan_report,
    scan_file)

PLUGIN_SCANNER_VERSION = "plugin-guard-v2"

# Never scanned: VCS internals, caches, vendored envs.
EXCLUDED_DIRS = {
    ".git", "__pycache__", "node_modules", ".venv", "venv",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", ".tox"}

# Top-level test trees ARE scanned (``plugins_loader`` sets ``submodule_search_locations``
# to the plugin root, so ``from .tests import evil`` runs whatever lives there), but a
# critical found under one is capped at ``high``: fixtures deliberately hold hostile
# strings to prove the plugin rejects them, and an un-overridable ``dangerous`` made
# such plugins uninstallable and taught authors to obfuscate their own tests (#89610).
# The cap keeps the verdict at ``caution`` — blocked by default, ``--force`` overridable.
# Root-level names only: ``src/spec/handler.py`` is runtime code and gets no cap.
TEST_TREE_DIRS = {"tests", "test", "testing", "spec", "specs", "fixtures"}

# Code files, where "reads an env secret" / "HTTP call with a key" is normal (requires_env).
CODE_FILE_EXTENSIONS = {".py", ".js", ".ts", ".sh", ".bash", ".rb", ".pl", ".php"}

# Line-comment marker per code extension. Whole-line comments explain intent; hardening
# notes like "# a symlink could point at /etc/passwd" are prose *about* a defense.
COMMENT_PREFIXES_BY_EXTENSION = {
    ".py": "#", ".sh": "#", ".bash": "#", ".rb": "#", ".pl": "#", ".r": "#", ".jl": "#",
    ".js": "//", ".ts": "//", ".php": "//", ".css": "//"}

# One severity step down from the pattern's default.
_COMMENT_SEVERITY_CAP = {"critical": "high", "high": "medium"}

# History, not an agent-facing instruction surface: a hardening entry mentioning the threat
# it fixed ("A symlink could point at /etc/passwd, so ...") is documentation, not the attack.
CHANGELOG_FILENAMES = {"changelog.md"}

# Pattern ids exempt on code files (every legitimate provider plugin trips them); still
# applied in full to docs/config files.
CODE_EXEMPT_PATTERN_IDS = {
    "python_environ_get_secret", "python_getenv_secret", "python_os_environ", "node_process_env",
    "ruby_env_secret", "env_exfil_httpx", "env_exfil_requests", "env_exfil_fetch",
    "env_exfil_curl", "env_exfil_wget",
    # Agent-facing instruction patterns are meaningless inside code (prompt docstrings trip them).
    "context_exfil", "send_to_url", "fake_policy",
    # Plugins legitimately write config.yaml in post_setup and base64 credentials (Basic auth).
    "agent_config_mod", "agent_config_contract", "encoded_exfil"}

# Severity remaps: a bundled binary is warn-tier (repos occasionally vendor one); a mere
# ``~/.hermes/.env`` mention is how READMEs say where keys go (READING it still trips
# ``read_secrets_file``, critical); ``curl | sh`` in READMEs is caution, not a hard block.
SEVERITY_REMAP = {
    "binary_file": "high", "hermes_env_access": "medium", "curl_pipe_shell": "high"}

# In JS/TS, these text matches cannot distinguish a UI label or DNS lookup
# template from a write or exfiltration operation. Keep them visible and require
# confirmation; do not silently allow them. Shell commands and instructions keep
# their critical severity, as do separate credential-read/exfiltration findings.
JS_CAPABILITY_REMAP = {"dns_exfil": "high", "ssh_backdoor": "high"}

# Plugin scans gate a HOST install: what matters is what executes on the host. Two critical
# families describe the author's own dev workflow when they appear in documentation files, so
# they are demoted one tier (critical -> high) there instead of hard-blocking an otherwise
# auditable plugin; the same content in runtime code keeps its critical severity.
DOC_PROSE_EXTENSIONS = {".md", ".txt", ".rst", ".html"}
DOC_PROSE_DEMOTIONS = {
    # Prose modification bullets ("- Modify: `CLAUDE.md`") in plan/design docs describe the
    # repo's own files; only executable intent (shell writes, code) stays critical.
    "agent_config_mod": "high",
    # Example/demo credentials quoted in docs (placeholder hex, test tokens). Real token-shaped
    # literals (sk-, ghp_, AKIA, glpat-, private keys) keep their own critical patterns.
    "hardcoded_secret": "high",
}

# Structural limits — plugins are real codebases, far larger than skills.
MAX_PLUGIN_FILE_COUNT = 400
MAX_PLUGIN_TOTAL_SIZE_KB = 10 * 1024   # 10MB of scannable tree
MAX_PLUGIN_SINGLE_FILE_KB = 1024       # 1MB single file


def _walk(plugin_dir: Path) -> Iterator[Tuple[Path, str]]:
    """Yield (path, "a/b/c" relative path) for every non-excluded entry under plugin_dir."""
    for f in plugin_dir.rglob("*"):
        try:
            rel_parts = f.relative_to(plugin_dir).parts
        except ValueError:
            continue
        if not any(part in EXCLUDED_DIRS for part in rel_parts):
            yield f, "/".join(rel_parts)


def _finding(pattern_id: str, severity: str, category: str, file: str, match: str, description: str) -> Finding:
    return Finding(pattern_id, severity, category, file, 0, match, description)


def _filter_findings(findings: List[Finding], rel_path: str) -> List[Finding]:
    """Apply plugin-specific exemptions and severity remaps to raw findings."""
    is_code = Path(rel_path).suffix.lower() in CODE_FILE_EXTENSIONS
    in_test_tree = Path(rel_path).parts[0] in TEST_TREE_DIRS
    is_js = Path(rel_path).suffix.lower() in {".js", ".ts"}
    is_doc_prose = Path(rel_path).suffix.lower() in DOC_PROSE_EXTENSIONS
    out: List[Finding] = []
    for f in findings:
        if is_code and f.pattern_id in CODE_EXEMPT_PATTERN_IDS:
            continue
        f.severity = (
            (JS_CAPABILITY_REMAP.get(f.pattern_id) if is_js else None)
            or SEVERITY_REMAP.get(f.pattern_id) or f.severity
        )
        if is_doc_prose and f.pattern_id in DOC_PROSE_DEMOTIONS:
            f.severity = DOC_PROSE_DEMOTIONS[f.pattern_id]
        if in_test_tree and f.severity == "critical":
            f.severity = "high"
        if (
            _is_defensive_documentation(f, rel_path)
            and f.severity in _COMMENT_SEVERITY_CAP
        ):
            f.severity = _COMMENT_SEVERITY_CAP[f.severity]
        out.append(f)
    return out


def _is_defensive_documentation(finding: Finding, rel_path: str) -> bool:
    """A whole-line code comment or a changelog entry *describes* threats (the attack a
    defense rejects, the hardening a release shipped) instead of executing them, so its
    findings cap one severity step lower — visible and reviewable, never un-overridable
    ``dangerous`` from prose alone. Runtime code and agent-facing docs keep full severity.
    """
    if Path(rel_path).name.lower() in CHANGELOG_FILENAMES:
        return True
    prefix = COMMENT_PREFIXES_BY_EXTENSION.get(Path(rel_path).suffix.lower())
    if prefix is None or not finding.match:
        return False
    stripped = finding.match.lstrip()
    if not stripped.startswith(prefix):
        return False
    if prefix == "#" and stripped.startswith(("#!", "#:")):
        return False
    return True


def _dangerous_findings_summary(findings: List[Finding]) -> str:
    """Describe the critical findings that made a plugin install dangerous."""
    critical = [finding for finding in findings if finding.severity == "critical"]
    pattern_ids = sorted({finding.pattern_id for finding in critical})
    names = f" ({', '.join(pattern_ids)})" if pattern_ids else ""
    return f"{len(critical)} critical of {len(findings)} findings{names}"


def _check_plugin_structure(plugin_dir: Path) -> List[Finding]:
    """Structural checks sized for plugin repositories."""
    findings: List[Finding] = []
    file_count = 0
    total_size = 0
    resolved_root = plugin_dir.resolve()
    for f, rel in _walk(plugin_dir):
        if f.is_symlink():
            file_count += 1
            try:
                resolved = f.resolve()
            except OSError:
                findings.append(_finding("broken_symlink", "medium", "traversal", rel,
                                         "broken symlink", "broken or circular symlink"))
                continue
            if not resolved.is_relative_to(resolved_root):
                findings.append(_finding("symlink_escape", "critical", "traversal", rel,
                                         f"symlink -> {resolved}", "symlink points outside the plugin directory"))
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
            findings.append(_finding("oversized_file", "medium", "structural", rel, f"{size // 1024}KB",
                                     f"file is {size // 1024}KB (limit: {MAX_PLUGIN_SINGLE_FILE_KB}KB)"))
        ext = f.suffix.lower()
        if ext in SUSPICIOUS_BINARY_EXTENSIONS:
            findings.append(_finding("binary_file", SEVERITY_REMAP["binary_file"], "structural", rel,
                                     f"binary: {ext}", f"binary/executable file ({ext}) bundled in plugin (cannot be scanned)"))
    if file_count > MAX_PLUGIN_FILE_COUNT:
        findings.append(_finding("too_many_files", "medium", "structural", "(directory)", f"{file_count} files",
                                 f"plugin has {file_count} files (limit: {MAX_PLUGIN_FILE_COUNT})"))
    if total_size > MAX_PLUGIN_TOTAL_SIZE_KB * 1024:
        findings.append(_finding("oversized_bundle", "medium", "structural", "(directory)", f"{total_size // 1024}KB",
                                 f"plugin is {total_size // 1024}KB total (limit: {MAX_PLUGIN_TOTAL_SIZE_KB}KB)"))
    return findings


def scan_plugin(plugin_dir: Path, source: str = "") -> ScanResult:
    """Scan a plugin directory (typically the temp clone); every external plugin is ``community`` trust."""
    all_findings: List[Finding] = []
    if plugin_dir.is_dir():
        all_findings.extend(_check_plugin_structure(plugin_dir))
        for f, rel in sorted(_walk(plugin_dir)):
            if f.is_file() and not f.is_symlink():
                all_findings.extend(_filter_findings(scan_file(f, rel_path=rel), rel))
    verdict = _determine_verdict(all_findings)
    if all_findings:
        categories = sorted({f.category for f in all_findings})
        summary = f"{plugin_dir.name}: {verdict} — {len(all_findings)} finding(s) in {', '.join(categories)}"
    else:
        summary = f"{plugin_dir.name}: clean scan, no threats detected"
    result = ScanResult(
        skill_name=plugin_dir.name, source=source or plugin_dir.name, trust_level="community",
        verdict=verdict, findings=all_findings, scanned_at=datetime.now(timezone.utc).isoformat(),
        summary=summary)
    result.scan_provenance = {
        "scanner_version": PLUGIN_SCANNER_VERSION, "verdict": verdict, "source": result.source}
    return result


def should_allow_plugin_install(
    result: ScanResult, force: bool = False) -> Tuple[Optional[bool], str]:
    """Map a verdict to ``(allowed, reason)``: True installs, None asks to confirm, False blocks."""
    n = len(result.findings)
    if result.verdict == "safe":
        return True, "Allowed (clean scan)"
    if result.verdict == "caution":
        if force:
            return True, f"Force-installed despite caution verdict ({n} findings)"
        return None, f"Requires confirmation (caution verdict, {n} findings)"
    return False, (
        f"Blocked (dangerous verdict, {_dangerous_findings_summary(result.findings)}). "
        f"--force does not override a dangerous verdict.")


__all__ = [
    "scan_plugin", "should_allow_plugin_install", "format_scan_report", "PLUGIN_SCANNER_VERSION"]
