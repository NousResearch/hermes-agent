"""Syntax-lint and LSP-diagnostics tier for ``tools.file_operations``.

``ShellFileOperations`` inherits ``LintMixin``; module constants and in-process
linters are pure functions importable from this module.
"""

import ast
import json
import os
import tomllib
from typing import Callable, Dict, Optional

from tools.file_operations_common import LintResult

# Shell linters by extension (external toolchain). ``.tsx`` is deliberately absent:
# it hits the "No linter" skip and LSP covers it when enabled.
LINTERS = {
    '.py': 'python -m py_compile {file} 2>&1',
    '.js': 'node --check {file} 2>&1',
    '.ts': 'npx tsc --noEmit {file} 2>&1',
    '.go': 'go vet {file} 2>&1',
    '.rs': 'rustfmt --check {file} 2>&1',
}

# Per-file shell linters that flood phantom errors on real projects (single-file
# ``tsc`` ignores tsconfig, ``go vet`` fails outside a module, ``rustfmt --check``
# is style-only): skipped when an LSP server claims the file. py_compile /
# node --check are file-local and correct so always run.
_SHELL_LINTER_LSP_REDUNDANT = frozenset({'.ts', '.go', '.rs'})

# Output substrings (case-insensitive) meaning the linter binary exists but could
# not run → ``skipped`` so the write isn't flagged and the LSP tier still runs.
_LINTER_UNUSABLE_PATTERNS = {
    'npx': (
        'this is not the tsc command you are looking for',  # tsc not installed locally
        'could not determine executable to run',
        'not found in npm registry',
    ),
    'rustfmt': (
        'no input filename given',  # outside a Cargo project
        'error: not a workspace',
    ),
    'go': (
        'cannot find package',  # outside a module / GOPATH
        'go: cannot find main module',
    ),
}


def _looks_like_linter_unusable(base_cmd: str, output: str) -> bool:
    """True iff ``output`` from ``base_cmd`` (first word of the linter cmd) says the tool itself couldn't run."""
    patterns = _LINTER_UNUSABLE_PATTERNS.get(base_cmd)
    if not patterns:
        return False
    lower = output.lower()
    return any(p in lower for p in patterns)


def _lint_json_inproc(content: str) -> tuple[bool, str]:
    """In-process JSON syntax check. Returns (ok, error_message)."""
    try:
        json.loads(content)
        return True, ""
    except json.JSONDecodeError as e:
        return False, f"JSONDecodeError: {e.msg} (line {e.lineno}, column {e.colno})"
    except Exception as e:  # noqa: BLE001 — any parse failure is a lint failure
        return False, f"{type(e).__name__}: {e}"


def _lint_yaml_inproc(content: str) -> tuple[bool, str]:
    """In-process YAML syntax check; ``__SKIP__`` when PyYAML is missing. Syntax-only
    (``yaml.parse``), NOT ``safe_load``: loading rejects valid multi-doc streams and
    app tags (``!Sub``, ``!vault``), and this is a fail-closed WRITE gate."""
    try:
        import yaml as _yaml
    except ImportError:
        return True, "__SKIP__"
    try:
        for _event in _yaml.parse(content):
            pass
        return True, ""
    except _yaml.YAMLError as e:
        return False, f"YAMLError: {e}"
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"


def _lint_toml_inproc(content: str) -> tuple[bool, str]:
    """In-process TOML syntax check (stdlib tomllib)."""
    try:
        tomllib.loads(content)
        return True, ""
    except Exception as e:  # TOMLDecodeError is a ValueError subclass
        return False, f"{type(e).__name__}: {e}"


def _lint_python_inproc(content: str) -> tuple[bool, str]:
    """In-process Python syntax check via ast.parse (py_compile's scope, no subprocess)."""
    try:
        ast.parse(content)
        return True, ""
    except SyntaxError as e:
        loc = f" (line {e.lineno}, column {e.offset})" if e.lineno else ""
        return False, f"{type(e).__name__}: {e.msg}{loc}"
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"


# In-process linters, preferred over shell linters (no subprocess). Each returns
# (ok, error); error ``"__SKIP__"`` = unavailable dependency, counts as "no linter".
LINTERS_INPROC: Dict[str, Callable[[str], tuple[bool, str]]] = {
    '.py': _lint_python_inproc,
    '.json': _lint_json_inproc,
    '.yaml': _lint_yaml_inproc,
    '.yml': _lint_yaml_inproc,
    '.toml': _lint_toml_inproc,
}

# Extensions where write_file REFUSES on a parse failure. ``.py`` is excluded on
# purpose: test fixtures use ``*.py`` paths as a stand-in for arbitrary text, so
# Python keeps the non-blocking lint-delta report.
_FAIL_CLOSED_INPROC_EXTS = frozenset({'.json', '.yaml', '.yml', '.toml'})


class LintMixin:
    """Post-write syntax lint + LSP diagnostics. Requires ``_exec``,
    ``_has_command``, ``_escape_shell_arg``, ``_escape_native_tool_arg`` and
    ``env`` from the host class."""

    def _check_lint(self, path: str, content: Optional[str] = None) -> LintResult:
        """Syntax-check ``path``: in-process linter when one matches the
        extension (``content`` avoids a re-read), else the shell linter table."""
        ext = os.path.splitext(path)[1].lower()
        inproc = LINTERS_INPROC.get(ext)
        if inproc is not None:
            if content is None:
                read_result = self._exec(f"cat {self._escape_shell_arg(path)} 2>/dev/null")
                if read_result.exit_code != 0:
                    return LintResult(skipped=True, message=f"Failed to read {path} for lint")
                content = read_result.stdout
            ok, err = inproc(content)
            if err == "__SKIP__":
                return LintResult(skipped=True, message=f"No linter available for {ext} (missing dependency)")
            return LintResult(success=ok, output="" if ok else err)
        if ext not in LINTERS:
            return LintResult(skipped=True, message=f"No linter for {ext} files")
        # Single-file tsc can't read tsconfig.json and floods phantom TS2307/TS2339
        # errors the delta filter misreports as "pre-existing"; let the LSP tier speak.
        if ext == '.ts' and self._has_ancestor_tsconfig(path):
            return LintResult(skipped=True, message=(
                "Project tsconfig.json detected — per-file tsc skipped "
                "(single-file tsc can't resolve project aliases/globals; "
                "use the LSP tier or `tsc -p tsconfig.json` for real "
                "diagnostics)."
            ))
        if ext in _SHELL_LINTER_LSP_REDUNDANT and self._lsp_will_handle(path):
            return LintResult(skipped=True, message=f"LSP server handles {ext} — shell linter skipped")
        linter_cmd = LINTERS[ext]
        base_cmd = linter_cmd.split()[0]
        if not self._has_command(base_cmd):
            return LintResult(skipped=True, message=f"{base_cmd} not available")
        # Native Windows binaries need C:/... not MSYS /c/... (→ phantom ENOENT).
        result = self._exec(linter_cmd.replace("{file}", self._escape_native_tool_arg(path)), timeout=30)
        if result.exit_code != 0 and _looks_like_linter_unusable(base_cmd, result.stdout):
            from tools.ansi_strip import strip_ansi
            cleaned = strip_ansi(result.stdout).strip()
            # Collapse to one line — the npx banner is multi-line ASCII art.
            first_line = next((ln.strip() for ln in cleaned.splitlines() if ln.strip()), cleaned[:120])
            return LintResult(skipped=True, message=f"{base_cmd} not usable: {first_line[:200]}")
        return LintResult(success=result.exit_code == 0, output=result.stdout.strip())

    def _check_lint_delta(self, path: str, pre_content: Optional[str],
                          post_content: Optional[str] = None) -> LintResult:
        """Post-write lint; when it fails and ``pre_content`` is known, report only
        errors this edit introduced (pre-existing lines filtered out). Semantic
        (LSP) diagnostics are a separate channel — see ``_maybe_lsp_diagnostics``."""
        post = self._check_lint(path, content=post_content)
        if post.success or post.skipped or pre_content is None:
            return post
        pre = self._check_lint(path, content=pre_content)
        if pre.success or pre.skipped or not pre.output:
            return post  # pre-write was clean (or unlintable): all post errors are new
        # Single-error parsers stop at the first error, so if every post error already
        # existed we can't prove the edit is clean — say nothing new was introduced.
        pre_lines = {ln.strip() for ln in pre.output.splitlines() if ln.strip()}
        post_lines = [ln for ln in post.output.splitlines() if ln.strip() and ln.strip() not in pre_lines]
        if not post_lines:
            return LintResult(success=False, output=post.output, message=(
                "Pre-existing lint errors — this edit didn't introduce new ones but the file is still broken."))
        return LintResult(success=False, output=(
            "New lint errors introduced by this edit "
            "(pre-existing errors filtered out):\n" + "\n".join(post_lines)
        ))

    def _lsp_local_only(self) -> bool:
        """True iff wired to a local backend. LSP servers run on the host and
        can't see files inside Docker/Modal/SSH/Daytona sandboxes."""
        env = getattr(self, "env", None)  # tests may build via __new__ without __init__
        if env is None:
            return False
        try:
            from tools.environments.local import LocalEnvironment
        except Exception:  # noqa: BLE001
            return False
        return isinstance(env, LocalEnvironment)

    def _lsp_service(self):
        """The active LSPService, or None on a non-local backend / any failure.
        LSP is an enrichment layer and must never break a write."""
        if not self._lsp_local_only():
            return None
        try:
            from agent.lsp import get_service
            return get_service()
        except Exception:  # noqa: BLE001
            return None

    def _lsp_handles_extension(self, ext: str) -> bool:
        """True iff some registered LSP server claims ``ext`` (static registry
        only; safe on remote backends). Decides whether pre-write content is
        worth capturing for the line-shift map."""
        if not ext:
            return False
        try:
            from agent.lsp.servers import SERVERS
        except Exception:  # noqa: BLE001
            return False
        return any(ext.lower() in srv.extensions for srv in SERVERS)

    def _has_ancestor_tsconfig(self, path: str) -> bool:
        """True iff a tsconfig.json exists in ``path``'s directory or any ancestor.
        Host-side walk, local backend only: on a remote backend this answers False so
        the shell linter still runs — never suppress lint on a probe that couldn't answer."""
        if not self._lsp_local_only():
            return False
        try:
            d = os.path.dirname(os.path.abspath(path))
            while not os.path.isfile(os.path.join(d, "tsconfig.json")):
                parent = os.path.dirname(d)
                if parent == d:
                    return False
                d = parent
            return True
        except Exception:  # noqa: BLE001
            return False

    def _lsp_call(self, method: str, path: str, default):
        """``svc.<method>(path)`` on the active service; ``default`` when there is
        no service or the call raises (LSP never breaks a write)."""
        svc = self._lsp_service()
        if svc is None:
            return default
        try:
            return getattr(svc, method)(path)
        except Exception:  # noqa: BLE001
            return default

    def _lsp_will_handle(self, path: str) -> bool:
        """True iff the LSP service is active AND ``enabled_for(path)`` (workspace
        detection, disabled-server set, broken-pair short-circuit). Any failure →
        False so the shell linter still runs."""
        return bool(self._lsp_call("enabled_for", path, False))

    def _snapshot_lsp_baseline(self, path: str) -> None:
        """Capture pre-edit LSP diagnostics so the post-write delta is correct. Silent on failure."""
        self._lsp_call("snapshot_baseline", path, None)

    def _maybe_lsp_diagnostics(self, path: str, *, pre_content: Optional[str] = None,
                               post_content: Optional[str] = None) -> str:
        """Formatted LSP diagnostics introduced by this edit, or "" when LSP is
        unavailable/disabled/clean. With both pre and post content a line-shift map
        remaps baseline diagnostics into post-edit coordinates; otherwise every
        pre-existing diagnostic below an inserted line would look new."""
        svc = self._lsp_service()
        if svc is None or not svc.enabled_for(path):
            return ""
        line_shift = None
        if pre_content is not None and post_content is not None and pre_content != post_content:
            try:
                from agent.lsp.range_shift import build_line_shift
                line_shift = build_line_shift(pre_content, post_content)
            except Exception:  # noqa: BLE001
                line_shift = None
        try:
            diagnostics = svc.get_diagnostics_sync(path, delta=True, line_shift=line_shift)
        except Exception:  # noqa: BLE001
            return ""
        if not diagnostics:
            return ""
        try:
            from agent.lsp.reporter import report_for_file, truncate
            block = report_for_file(path, diagnostics)
            return truncate("LSP diagnostics introduced by this edit:\n" + block) if block else ""
        except Exception:  # noqa: BLE001
            return ""
