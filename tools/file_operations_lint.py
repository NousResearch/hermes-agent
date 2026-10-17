"""Syntax-lint and LSP-diagnostics tier for ``tools.file_operations``.

``ShellFileOperations`` inherits ``LintMixin``; module constants and in-process
linters are pure functions importable from this module.
"""

import ast
import json
import logging
import os
import tomllib
from typing import Callable, Dict, Optional

from tools.file_operations_common import ExecuteResult, LintResult

logger = logging.getLogger(__name__)

# Shell linters by extension (external toolchain). ``.tsx`` is deliberately absent:
# it hits the "No linter" skip and LSP covers it when enabled.
LINTERS = {
    '.py': 'python -m py_compile {file} 2>&1',
    '.js': 'node --check {file} 2>&1',
    '.ts': 'npx tsc --noEmit {file} 2>&1',
    '.go': 'go vet {file} 2>&1',
    '.rs': 'rustfmt --check {file} 2>&1',
}

# Node linters Hermes runs on the host (local backend) under its PM-managed Node,
# never the user's: the terminal PATH puts the user's dirs first, and ``npx tsc``
# re-execs ``env node`` through that PATH.
_MANAGED_NODE_LINTERS = {'.js': ('node', '--check'), '.ts': ('npx', 'tsc', '--noEmit')}

# Per-file shell linters that flood phantom errors on real projects (single-file
# ``tsc`` ignores tsconfig, ``go vet`` fails outside a module, ``rustfmt --check``
# is style-only): skipped when an LSP server claims the file. py_compile /
# node --check are file-local and correct so always run.
_SHELL_LINTER_LSP_REDUNDANT = frozenset({'.ts', '.go', '.rs'})

# Shell linters that resolve the toolchain from the checkout they run in: ``npx tsc`` runs the
# repo's ``node_modules/.bin/tsc`` (or installs from the registry its ``.npmrc`` names) and
# rustup honours its ``rust-toolchain.toml``.  Skipped on a local backend in an untrusted workspace.
_SHELL_LINTER_RUNS_PROJECT_TOOLCHAIN = frozenset({'.ts', '.rs'})

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
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def _lint_yaml_inproc(content: str) -> tuple[bool, str]:
    """In-process YAML syntax check; ``__SKIP__`` when ruamel.yaml is missing. Syntax-only
    (``YAML.parse``), NOT ``safe_load``: loading rejects valid multi-doc streams and
    app tags (``!Sub``, ``!vault``), and this is a fail-closed WRITE gate."""
    try:
        from ruamel.yaml import YAML
        from ruamel.yaml.error import YAMLError
    except ImportError:
        return True, "__SKIP__"
    try:
        for _event in YAML(typ="safe").parse(content):
            pass
        return True, ""
    except YAMLError as e:
        return False, f"YAMLError: {e}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def _lint_toml_inproc(content: str) -> tuple[bool, str]:
    """In-process TOML syntax check (stdlib tomllib)."""
    try:
        tomllib.loads(content)
        return True, ""
    except Exception as e:  # TOMLDecodeError is a ValueError subclass
        return False, f"{type(e).__name__}: {e}"


def _lint_python_inproc(content: str) -> tuple[bool, str]:
    """In-process Python syntax check via ast.parse (py_compile's scope, no subprocess). Parses the
    bytes, as py_compile does, so a legacy ``coding:`` cookie is honoured for a file the edit paths
    read with surrogateescape."""
    try:
        ast.parse(content.encode("utf-8", "surrogateescape"))
        return True, ""
    except SyntaxError as e:
        loc = f" (line {e.lineno}, column {e.offset})" if e.lineno else ""
        return False, f"{type(e).__name__}: {e.msg}{loc}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


# In-process linters, preferred over shell linters (no subprocess). Each returns
# (ok, error); error ``"__SKIP__"`` = unavailable dependency, counts as "no linter".
LINTERS_INPROC: dict[str, Callable[[str], tuple[bool, str]]] = {
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
        if ext in _SHELL_LINTER_RUNS_PROJECT_TOOLCHAIN and self._local_workspace_untrusted():
            return LintResult(skipped=True, message=(
                f"{LINTERS[ext].split()[0]} skipped: untrusted workspace (add it to lsp.trusted_workspaces to lint {ext} here)"))
        if ext in _MANAGED_NODE_LINTERS and self._lsp_local_only():
            base_cmd = _MANAGED_NODE_LINTERS[ext][0]
            result = self._run_managed_node_linter(ext, path)
            if result is None:
                return LintResult(skipped=True, message=f"{base_cmd} not available (Hermes-managed Node not installed)")
        else:
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

    def _run_managed_node_linter(self, ext: str, path: str) -> Optional[ExecuteResult]:
        """Run the ``ext`` Node linter on the host under PM's Node; None when PM has none."""
        import shutil
        import subprocess

        from hermes_cli._subprocess_compat import windows_hide_flags
        from hermes_constants import with_hermes_node_path
        from tools.environments.local import _IS_WINDOWS, _msys_to_windows_path, hermes_subprocess_env

        tool, *args = _MANAGED_NODE_LINTERS[ext]
        executable = shutil.which(tool, path=with_hermes_node_path({"PATH": ""})["PATH"])
        if executable is None:
            return None
        cwd = getattr(self.env, "cwd", None) or self.cwd
        if _IS_WINDOWS:
            path, cwd = _msys_to_windows_path(path), cwd and _msys_to_windows_path(cwd)
        try:
            proc = subprocess.run(
                [executable, *args, path], cwd=cwd or None, env=with_hermes_node_path(hermes_subprocess_env()),
                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", timeout=30, creationflags=windows_hide_flags(), check=False)
        except subprocess.TimeoutExpired:
            return ExecuteResult(stdout=f"{tool} timed out after 30s", exit_code=124)
        return ExecuteResult(stdout=proc.stdout or "", exit_code=proc.returncode)

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
        except Exception:
            return False
        return isinstance(env, LocalEnvironment)

    def _local_workspace_untrusted(self) -> bool:
        """True iff on a local backend the linter's cwd lies outside every trusted workspace
        (``workspace.is_trusted_workspace``): ``npx`` and rustup resolve the toolchain from there, not
        from the linted file's directory.  Sandboxed backends answer False: their toolchain is not the host's."""
        if not self._lsp_local_only():
            return False
        from agent.lsp.manager import parse_trusted_workspaces
        from agent.lsp.workspace import is_trusted_workspace, operator_workspace_roots
        from hermes_cli.config import load_config_readonly
        lsp_cfg = load_config_readonly().get("lsp")
        trusted = parse_trusted_workspaces(lsp_cfg.get("trusted_workspaces") if isinstance(lsp_cfg, dict) else None)
        return not is_trusted_workspace(getattr(self.env, "cwd", None) or self.cwd, trusted, operator_workspace_roots())

    def _lsp_service(self):
        """The active LSPService, or None on a non-local backend / any failure.
        LSP is an enrichment layer and must never break a write."""
        if not self._lsp_local_only():
            return None
        try:
            from agent.lsp import get_service
            return get_service()
        except Exception:
            return None

    def _lsp_handles_extension(self, ext: str) -> bool:
        """True iff the active service (config-declared servers included) or, without one,
        the static registry claims ``ext``. Decides whether pre-write content is worth
        capturing for the line-shift map."""
        if not ext:
            return False
        try:
            svc = self._lsp_service()
            if svc is not None:
                return svc.handles_extension(ext)
            from agent.lsp.servers import SERVERS
        except Exception:
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
        except Exception:
            return False

    def _lsp_call(self, method: str, path: str, default):
        """``svc.<method>(path)`` on the active service; ``default`` when there is
        no service or the call raises (LSP never breaks a write)."""
        svc = self._lsp_service()
        if svc is None:
            return default
        try:
            return getattr(svc, method)(path)
        except Exception:
            return default

    def _lsp_will_handle(self, path: str) -> bool:
        """True iff the LSP service is active AND ``enabled_for(path)`` (workspace
        detection, disabled-server set, broken-pair short-circuit). Any failure →
        False so the shell linter still runs."""
        return bool(self._lsp_call("enabled_for", path, False))

    def _maybe_lsp_diagnostics(self, path: str, *, pre_content: Optional[str] = None,
                               post_content: Optional[str] = None, started: Optional[float] = None) -> str:
        """Formatted LSP diagnostics this edit introduced, or "" when LSP is unavailable/disabled/clean
        OR the verdict is not ready yet.  Never blocks the write on the language server: the check
        (``agent.lsp.edit_check``) diagnoses ``post_content`` (the bytes this write produced) against
        ``pre_content`` (None = new file) on the LSP loop: those snapshots, not a later read of the
        file, which another session may already have rewritten.  The write waits for the verdict only until ``inline_deadline(started)`` (what is
        left of the write's latency budget; ``started`` = ``time.monotonic()`` at write entry); a slower
        verdict reaches the model with a later tool result of the same session (``agent.lsp.deferred``)."""
        svc = self._lsp_service()
        if svc is None or not self._lsp_will_handle(path):
            return ""
        try:
            from agent.lsp import deferred
            from agent.lsp.edit_check import inline_wait
            abs_path = os.path.abspath(path)
            owner = deferred.current_owner()
            # This session's undelivered verdict for the file: the new check takes it over (diffs
            # against its baseline) and hands it back in ``result.carry`` only if it gets no verdict.
            older = deferred.take(owner, abs_path)
            carry = (older,) if older is not None else ()
            future = svc.start_edit_check(path, pre_text=pre_content, post_text=post_content, owner=owner,
                                          carry=carry)
        except Exception:
            return ""
        if future is None:
            deferred.restore(owner, abs_path, carry)
            return ""
        import concurrent.futures
        try:
            result = future.result(timeout=inline_wait(started))
        except concurrent.futures.TimeoutError:
            if owner:
                token = deferred.track_pending(owner)
                future.add_done_callback(lambda f: _post_deferred(owner, abs_path, f, carry, token))
            return ""
        except Exception:  # LSP never breaks a write
            logger.debug("LSP edit check failed for %s", path, exc_info=True)
            deferred.restore(owner, abs_path, carry)
            return ""
        if result.superseded:
            return ""
        if result.diagnostics is None:
            deferred.restore(owner, abs_path, result.carry)
            return ""
        return _format_lsp_block(path, result.diagnostics, "LSP diagnostics introduced by this edit:\n")


def _format_lsp_block(path: str, diagnostics, header: str) -> str:
    if not diagnostics:
        return ""
    try:
        from agent.lsp.reporter import report_for_file, truncate
        block = report_for_file(path, diagnostics)
        return truncate(header + block) if block else ""
    except Exception:  # LSP never breaks a write
        logger.debug("formatting LSP diagnostics for %s failed", path, exc_info=True)
        return ""


def _post_deferred(owner: str, abs_path: str, future, carry, token) -> None:
    """Done-callback (LSP loop thread): queue a late verdict for the session that made the edit, then
    release a turn end waiting on it (``agent.lsp.deferred.settle_for_agent``)."""
    from agent.lsp import deferred
    try:
        _queue_late_verdict(owner, abs_path, future, carry)
    finally:
        deferred.settled(owner, token)


def _queue_late_verdict(owner: str, abs_path: str, future, carry) -> None:
    """A superseded check posts nothing: the newer check owns its verdict AND the notes it carried.
    No verdict hands back only what the check still holds (``result.carry``), never the caller's copy."""
    from agent.lsp import deferred
    try:
        result = None if future.cancelled() else future.result()
    except Exception:  # a failed check reports nothing; the older note goes back
        logger.debug("deferred LSP edit check failed for %s", abs_path, exc_info=True)
        deferred.restore(owner, abs_path, carry)
        return
    if result is None:
        deferred.restore(owner, abs_path, carry)
        return
    if result.superseded:
        return
    if result.diagnostics is None:
        deferred.restore(owner, abs_path, result.carry)
        return
    text = _format_lsp_block(abs_path, result.diagnostics, (
        f"LSP diagnostics introduced by your earlier edit of {abs_path} "
        "(the language server answered after that tool result returned):\n"))
    if text:
        deferred.post(owner, abs_path, text, result.baseline_text)
