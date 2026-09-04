"""Content/file search tier for ``tools.file_operations``.

``ShellFileOperations`` inherits ``SearchMixin``; module-level helpers are pure
(no I/O).
"""

import os
import posixpath
import re
import sys
import threading
from pathlib import Path
from typing import Any, List, Optional

from agent.search_policy import SEARCH_PRUNE_DIR_NAMES
from tools import interrupt as tool_interrupt
from tools.file_operations_common import ExecuteResult, SearchMatch, SearchResult

_MACOS_TCC_PROTECTED_HOME_DIRS = (
    "Desktop", "Documents", "Downloads", "Library", "Movies", "Music", "Pictures",
)


def _macos_protected_search_exclusions(
    path: str, *, cwd: Optional[str] = None, home: Optional[str] = None, platform: Optional[str] = None,
) -> List[str]:
    """Protected home dirs (relative to ``path``) below a broad macOS search root.

    Only an ANCESTOR search (``$HOME``, ``/Users``) gets exclusions, so recursive
    tools never trigger unattended TCC prompts; a search rooted inside a
    protected dir stays allowed.
    """
    if (platform or sys.platform) != "darwin":
        return []
    root = Path(path).expanduser()
    if not root.is_absolute():
        root = Path(cwd or os.getcwd()) / root
    root = Path(os.path.normpath(str(root)))
    home_path = Path(os.path.normpath(str(Path(home or Path.home()).expanduser())))
    exclusions: List[str] = []
    for dirname in _MACOS_TCC_PROTECTED_HOME_DIRS:
        try:
            relative = (home_path / dirname).relative_to(root)
        except ValueError:
            continue
        if relative.parts:
            exclusions.append(relative.as_posix())
    return exclusions


# --- Filename-walk admission: one walk per (backend, root) at a time --------------

_FILENAME_SEARCH_ADMISSION = threading.Condition()
_ACTIVE_FILENAME_SEARCH_ROOTS: set[tuple[str, str, str]] = set()
_FILENAME_SEARCH_WAIT_SECONDS = 0.05


def _normalized_filename_search_root(env: Any, root: str, fallback_cwd: str) -> str:
    """Normalize a filename-walk root without resolving remote paths locally."""
    from tools.environments.local import LocalEnvironment, _IS_WINDOWS, _msys_to_windows_path

    cwd = getattr(env, "cwd", None) or fallback_cwd
    if isinstance(env, LocalEnvironment):
        if _IS_WINDOWS:
            root = _msys_to_windows_path(root)
            cwd = _msys_to_windows_path(cwd)
        if not os.path.isabs(root):
            root = os.path.join(cwd, root)
        return os.path.normcase(os.path.abspath(os.path.normpath(root)))
    if not posixpath.isabs(root):
        root = posixpath.join(cwd, root)
    return posixpath.normpath(root)


def _filename_search_root_keys(env: Any, roots: List[str], fallback_cwd: str) -> tuple[tuple[str, str, str], ...]:
    """Unique backend/root admission keys in deterministic order."""
    env_type = type(env)
    return tuple(sorted({
        (env_type.__module__, env_type.__qualname__, _normalized_filename_search_root(env, root, fallback_cwd))
        for root in roots}))


def _acquire_filename_search_roots(keys: tuple[tuple[str, str, str], ...]) -> bool:
    """Atomically claim every key, polling for thread-scoped interruption."""
    with _FILENAME_SEARCH_ADMISSION:
        while any(key in _ACTIVE_FILENAME_SEARCH_ROOTS for key in keys):
            if tool_interrupt.is_interrupted():
                return False
            _FILENAME_SEARCH_ADMISSION.wait(_FILENAME_SEARCH_WAIT_SECONDS)
            if tool_interrupt.is_interrupted():
                return False
        if tool_interrupt.is_interrupted():
            return False
        return tool_interrupt.run_if_not_interrupted(lambda: _ACTIVE_FILENAME_SEARCH_ROOTS.update(keys))


def _release_filename_search_roots(keys: tuple[tuple[str, str, str], ...]) -> None:
    """Release a completed walk and leave no idle per-root state behind."""
    with _FILENAME_SEARCH_ADMISSION:
        _ACTIVE_FILENAME_SEARCH_ROOTS.difference_update(keys)
        _FILENAME_SEARCH_ADMISSION.notify_all()


_ADMISSION_INTERRUPTED_ERROR = (
    "File search was interrupted while waiting for another filename "
    "search on the same root. Retry when ready.")

_SEARCH_TIMEOUT_MARKER_RE = re.compile(r"\n?\[Command timed out after \d+s\]\s*$")


def _search_stdout_and_limit(result: ExecuteResult) -> tuple[str, Optional[str]]:
    """Return stdout cleaned for parsing and a limit reason for search timeouts."""
    if result.exit_code == 124:
        return _SEARCH_TIMEOUT_MARKER_RE.sub("", result.stdout), "search_timeout"
    return result.stdout, None


# A real rg/grep output line is a whitespace-free path token followed by ``:``
# (match/count), ``-`` (context), or nothing (files_only); tool diagnostics
# ("rg: ...", indented carets) never match.
_SEARCH_OUTPUT_RE = re.compile(r'^([A-Za-z]:)?[^\s:][^\n]*?[:\-]\d|^[^\s:][^\s]*$')


def _split_tool_diagnostics(output: str) -> tuple[str, str]:
    """Separate rg/grep diagnostic lines from real match output → ``(diagnostics, payload)``.
    ``_exec`` merges stderr into stdout; classifying by SHAPE lets the exit-2 guard
    tell a pure failure (no payload) from a partial one (one unreadable file, others
    matched) and guarantees error text is never parsed as a match."""
    diagnostics: list[str] = []
    payload: list[str] = []
    for line in output.split('\n'):
        if not line.strip():
            continue
        # Prefix check first: a match path can contain "-<digit>" (".../pytest-686/...").
        if line.lstrip().startswith(("rg: ", "grep: ")):
            diagnostics.append(line)
        elif line == "--" or _SEARCH_OUTPUT_RE.match(line):
            payload.append(line)
        else:
            diagnostics.append(line)
    return '\n'.join(diagnostics), '\n'.join(payload)


def _parse_search_context_line(line: str) -> tuple[str, int, str] | None:
    """Parse a ``path-line-content`` context line using the RIGHTMOST numeric
    separator (filenames may contain ``-<digits>-`` segments):
    ``dir/file-12-name.py-8-context`` → (``dir/file-12-name.py``, 8, ``context``)."""
    if not line or line == "--":
        return None
    match = None
    for candidate in re.finditer(r'-(\d+)-', line):
        match = candidate
    if match is None or match.start() == 0:
        return None
    return line[:match.start()], int(match.group(1)), line[match.end():]


_REGEX_NEWLINE_ESCAPE_RE = re.compile(r"(?<!\\)(?:\\\\)*\\n")


def _pattern_has_regex_newline(pattern: str) -> bool:
    """True when a content regex wants to match a newline: a literal newline or a
    ``\\n`` escape with an ODD number of backslashes (``\\\\n`` is a literal
    backslash+n and must not count)."""
    return "\n" in pattern or bool(_REGEX_NEWLINE_ESCAPE_RE.search(pattern))


def _is_line_oriented_newline_error(error: Optional[str]) -> bool:
    """Return True for rg's hard error when multiline mode is required."""
    return bool(error) and "literal \"\\n\" is not allowed" in error and "--multiline" in error


def _maybe_warn_line_oriented_newline_pattern(result: SearchResult, pattern: str) -> SearchResult:
    """Attach a newline-regex warning only when search found no usable results."""
    if result.total_count != 0 or not _pattern_has_regex_newline(pattern):
        return result
    if result.error and not _is_line_oriented_newline_error(result.error):
        return result
    result.error = None
    result.warning = (
        "0 results found. Note: search_files content search is line-oriented "
        "and does not run ripgrep with -U/--multiline, so `\\n` in the regex "
        "does not match line breaks. Use context=N to inspect neighboring "
        "lines, or escape as `\\\\n` when searching for a literal backslash+n.")
    return result


# Match lines are "file:lineno:content". Windows paths carry a drive letter
# ("C:\path"), so a naive split(":") breaks — the regex handles both.
_MATCH_LINE_RE = re.compile(r'^([A-Za-z]:)?(.*?):(\d+):(.*)$')

# Output-mode → engine flag (identical for rg and grep).
_OUTPUT_MODE_FLAGS = {"files_only": "-l", "count": "-c"}


def _parse_search_output(result, output_mode: str, limit: int, offset: int,
                         context: int, warning: Optional[str] = None) -> SearchResult:
    """Parse rg/grep ``| head`` output into a SearchResult (shared by both engines).
    Exit codes: 0=matches, 1=none, 2=error — but both tools return 2 on PARTIAL
    errors (one unreadable file), so an error is surfaced only when exit==2 AND no
    usable payload remains. ``warning`` is attached to files_only/content results."""
    stdout, limit_reason = _search_stdout_and_limit(result)
    diagnostics, payload = _split_tool_diagnostics(stdout)
    if result.exit_code == 2 and not payload.strip():
        error_msg = diagnostics.strip() or result.stdout.strip() or "Search error"
        return SearchResult(error=f"Search failed: {error_msg}", total_count=0)
    lines = [ln for ln in payload.strip().split('\n') if ln]
    if output_mode == "files_only":
        return SearchResult(
            files=lines[offset:offset + limit], total_count=len(lines),
            truncated=bool(limit_reason), limit_reason=limit_reason, warning=warning)
    if output_mode == "count":
        counts = {}
        for line in lines:
            if ':' in line:
                path, n = line.rsplit(':', 1)
                try:
                    counts[path] = int(n)
                except ValueError:
                    pass
        return SearchResult(
            counts=counts, total_count=sum(counts.values()),
            truncated=bool(limit_reason), limit_reason=limit_reason)
    matches = []
    for line in lines:
        if line == "--":
            continue
        m = _MATCH_LINE_RE.match(line)
        if m:
            matches.append(SearchMatch(
                path=(m.group(1) or '') + m.group(2), line_number=int(m.group(3)), content=m.group(4)[:500],
            ))
            continue
        # Context lines only when requested, to avoid false positives on dashy paths.
        if context > 0:
            parsed = _parse_search_context_line(line)
            if parsed:
                matches.append(SearchMatch(path=parsed[0], line_number=parsed[1], content=parsed[2][:500]))
    total = len(matches)
    return SearchResult(
        matches=matches[offset:offset + limit], total_count=total,
        truncated=total > offset + limit or bool(limit_reason), limit_reason=limit_reason, warning=warning,
    )


def _posix_roots(roots: List[str]) -> bool:
    """Darwin-only: every root is POSIX-shaped (no drive letter / backslash)."""
    return sys.platform == "darwin" and all(
        not re.match(r"^[A-Za-z]:[\\/]", root) and "\\" not in root for root in roots)


class SearchMixin:
    """File-name and content search via rg with find/grep fallbacks. Requires
    ``_exec``, ``_has_command``, ``_expand_path``, ``_escape_shell_arg``,
    ``_escape_native_tool_arg``, ``env``, ``cwd``, ``_command_cache``,
    ``_rg_resolution_cache`` and ``_rg_modified_capability`` from the host class."""

    # --- rg resolution --------------------------------------------------------

    def _resolve_command(self, cmd: str) -> Optional[str]:
        """Resolve an executable in the command host's namespace. Ordinary commands
        keep the bool hit/miss cache; rg alone caches successful resolved paths and
        re-probes misses so a mid-session install becomes visible (with off-PATH
        Windows candidates: cargo, scoop, winget)."""
        if cmd != "rg":
            return cmd if self._has_command(cmd) else None
        cached = self._rg_resolution_cache.get(cmd)
        if cached:
            return cached
        result = self._exec("command -v rg 2>/dev/null")
        if result.exit_code == 0 and result.stdout.strip():
            resolved = result.stdout.strip().splitlines()[0]
            if resolved == "yes":  # compatibility with old boolean-probe fakes
                resolved = "rg"
            self._rg_resolution_cache[cmd] = resolved
            return resolved
        from tools.environments.local import LocalEnvironment, _IS_WINDOWS

        if _IS_WINDOWS and isinstance(self.env, LocalEnvironment):
            user_profile = os.environ.get("USERPROFILE") or str(Path.home())
            local_app_data = os.environ.get("LOCALAPPDATA")
            scoop = os.environ.get("SCOOP") or os.path.join(user_profile, "scoop")
            candidates = [
                os.path.join(user_profile, ".cargo", "bin", "rg.exe"),
                os.path.join(scoop, "shims", "rg.exe"),
            ]
            if local_app_data:
                candidates.append(os.path.join(local_app_data, "Microsoft", "WinGet", "Links", "rg.exe"))
            for candidate in candidates:
                if os.path.isfile(candidate):
                    resolved = candidate.replace("\\", "/")
                    self._rg_resolution_cache[cmd] = resolved
                    return resolved
        return None

    _RG_VERSION_RE = re.compile(
        r"(?m)^ripgrep\s+((?:0|[1-9]\d*))\."
        r"(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)"
        r"(?:-(?:(?:0|[1-9]\d*)|(?:[0-9A-Za-z-]*[A-Za-z-]"
        r"[0-9A-Za-z-]*))(?:\.(?:(?:0|[1-9]\d*)|"
        r"(?:[0-9A-Za-z-]*[A-Za-z-][0-9A-Za-z-]*)))*)?"
        r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?"
        r"(?:\s+\(rev [^)]+\))?\s*$")

    def _modified_rg_capability_error(self, executable: str) -> Optional[str]:
        """Cached actionable error unless rg can sort exactly (full SemVer, >= 14)."""
        if executable in self._rg_modified_capability:
            return self._rg_modified_capability[executable]
        result = self._exec(f"{self._quote_executable(executable)} --version", timeout=10)
        match = self._RG_VERSION_RE.search(result.stdout or "")
        if result.exit_code == 0 and match and int(match.group(1)) >= 14:
            error = None
        else:
            error = ("Exact modification-time order requires ripgrep 14 or newer; "
                     "upgrade ripgrep or use order='discovery'.")
        self._rg_modified_capability[executable] = error
        return error

    def _quote_executable(self, executable: str) -> str:
        """Quote an executable without leaking controller path semantics."""
        if re.fullmatch(r"[A-Za-z0-9_.-]+", executable):
            return executable
        from tools.environments.local import LocalEnvironment

        if isinstance(self.env, LocalEnvironment):
            return self._escape_native_tool_arg(executable)
        return "'" + executable.replace("'", "'\"'\"'") + "'"

    # --- macOS protected-folder exclusions --------------------------------------

    def _macos_search_exclusions(self, path: str) -> List[str]:
        """Protected descendants to prune for this search root, if any. Gated on
        ``env.is_local``: ``sys.platform``/``_HOME`` describe the CONTROLLER, but the
        search runs on ``env``'s host. Envs without the flag default to local
        semantics; pruning is a warning-carrying skip, never data loss."""
        env = getattr(self, "env", None)
        if env is not None and getattr(env, "is_local", True) is False:
            return []
        from tools import file_operations as _fo  # lazy: _HOME is monkeypatched there
        cwd = getattr(self.env, "cwd", None) or self.cwd
        return _macos_protected_search_exclusions(path, cwd=cwd, home=_fo._HOME, platform=sys.platform)

    def _protected_prune_paths(self, path: str) -> List[str]:
        """Absolute-ish protected paths for find's ``-path ... -prune``."""
        return [os.path.normpath(os.path.join(path, item)) for item in self._macos_search_exclusions(path)]

    def _effective_macos_search_exclusions(self, roots: List[str]) -> List[tuple[str, str, str]]:
        """Unique ``(root, relative, absolute)`` exclusions across ``roots``, never
        pruning a root the caller chose explicitly."""
        cwd = getattr(self.env, "cwd", None) or self.cwd
        use_posix_paths = _posix_roots(roots)

        def normalized(root: str) -> str:
            if use_posix_paths:
                return posixpath.normpath(root if posixpath.isabs(root) else posixpath.join(cwd, root))
            return os.path.normcase(os.path.abspath(os.path.normpath(root)))

        normalized_roots = [normalized(root) for root in roots]
        explicit_roots = set(normalized_roots)
        seen = set()
        effective = []
        for root, normalized_root in zip(roots, normalized_roots):
            for relative in self._macos_search_exclusions(root):
                if use_posix_paths:
                    absolute = key = posixpath.normpath(posixpath.join(normalized_root, relative))
                else:
                    absolute = os.path.normpath(os.path.join(root, relative))
                    key = os.path.normcase(os.path.abspath(absolute))
                if key in explicit_roots or key in seen:
                    continue
                seen.add(key)
                effective.append((root, relative, absolute))
        return effective

    @staticmethod
    def _macos_protected_search_warning(paths: List[str]) -> str:
        skipped = ", ".join(os.path.basename(item) for item in paths)
        return ("Skipped macOS protected folders during broad search to avoid "
                f"an unattended privacy prompt: {skipped}. Search a protected "
                "folder directly when access is intentional.")

    def _prune_expr(self, protected_paths: List[str]) -> str:
        """find ``\\( -path A -o -path B \\) -prune`` clause for the protected dirs."""
        terms = " -o ".join(f"-path {self._escape_shell_arg(item)}" for item in protected_paths)
        return f"\\( {terms} \\) -prune"

    def _rg_exclusion_globs(self, path: str) -> List[str]:
        """``--glob '!<dir>/**'`` pairs excluding protected dirs from an rg run."""
        out: List[str] = []
        for item in self._macos_search_exclusions(path):
            out.extend(["--glob", self._escape_shell_arg(f"!{item}/**")])
        return out

    def _path_exists_probe(self, path: str) -> str:
        """Stdout of the existence probe: contains "exists" or "not_found"."""
        return self._exec(f"test -e {self._escape_shell_arg(path)} && echo exists || echo not_found").stdout

    def _dispatch_search(self, pattern: str, path: str, target: str,
                         file_glob: Optional[str], limit: int, offset: int,
                         output_mode: str, context: int, order: str = "discovery") -> SearchResult:
        if target == "files":
            return self._search_files(pattern, path, limit, offset, order)
        return self._search_content(pattern, path, file_glob, limit, offset, output_mode, context)

    def _path_not_found_result(self, path: str) -> SearchResult:
        """Error result for a missing search root, with nearby-entry suggestions."""
        parent = os.path.dirname(path) or "."
        basename_query = os.path.basename(path)
        hint_parts = [f"Path not found: {path}"]
        parent_check = self._exec(f"test -d {self._escape_shell_arg(parent)} && echo yes || echo no")
        if "yes" in parent_check.stdout and basename_query:
            ls_result = self._exec(f"ls -1 {self._escape_shell_arg(parent)} 2>/dev/null | head -20")
            if ls_result.exit_code == 0 and ls_result.stdout.strip():
                lq = basename_query.lower()
                candidates = [
                    os.path.join(parent, e) for e in ls_result.stdout.strip().split('\n')
                    if e and (lq in e.lower() or e.lower() in lq or e.lower().startswith(lq[:3]))]
                if candidates:
                    hint_parts.append("Similar paths: " + ", ".join(candidates[:5]))
        return SearchResult(error=". ".join(hint_parts), total_count=0)

    def _try_multi_path_search(self, pattern: str, path: str, target: str,
                               file_glob: Optional[str], limit: int, offset: int,
                               output_mode: str, context: int,
                               order: str = "discovery") -> Optional[SearchResult]:
        """Recover a not-found ``path`` that is really several paths in one string.
        Commas explicitly delimit paths (internal spaces preserved); without commas
        split on whitespace. Search every existing part, merge, and note skipped
        parts. None when it doesn't look like a multi-path string."""
        if "," in path:
            parts = [part.strip() for part in path.split(",") if part.strip()]
        else:
            parts = path.split()
        if len(parts) < 2:
            return None
        existing, missing = [], []
        for p in parts:
            expanded = self._expand_path(p)
            (existing if "exists" in self._path_exists_probe(expanded) else missing).append(expanded)
        if not existing:
            return None
        if target == "files":
            # One global traversal across roots so modified ordering and pagination
            # are exact; root admission wraps the actual rg/find invocation.
            merged = self._search_files(pattern, existing, limit, offset, order)
        else:
            merged = SearchResult()
            for root in existing:
                sub = self._search_content(pattern, root, file_glob, limit, offset, output_mode, context)
                if sub.error:
                    return sub
                merged.matches.extend(sub.matches)
                merged.files.extend(sub.files)
                merged.counts.update(sub.counts)
                merged.total_count += sub.total_count
                merged.truncated = merged.truncated or sub.truncated
            merged.matches = merged.matches[:limit]
            merged.files = merged.files[:limit]
        note = f"path contained {len(parts)} entries; searched {len(existing)} that exist"
        if missing:
            note += "; skipped missing: " + ", ".join(missing[:3])
            if len(missing) > 3:
                note += f" (+{len(missing) - 3} more)"
        warning_parts = [note]
        if not merged.error:
            protected_paths = [absolute for _r, _rel, absolute in self._effective_macos_search_exclusions(existing)]
            if protected_paths:
                warning_parts.append(self._macos_protected_search_warning(protected_paths))
        merged.warning = " ".join(warning_parts)
        return merged

    def _search_prune_glob_args(self) -> str:
        """rg globs pruning known heavyweight recursive subtrees. Both forms are
        needed: globs are relative to each rg root, so ``**/name/**`` alone misses an
        explicitly selected ``name/`` root. Names come from the shared scan policy —
        no second search-only list."""
        globs = []
        for dirname in sorted(SEARCH_PRUNE_DIR_NAMES):
            for prefix in ("", "**/"):
                globs.extend(("--glob", self._escape_shell_arg(f"!{prefix}{dirname}/**")))
        return " ".join(globs)

    # (rg flags, message template) probes for a 0-match content search, in order.
    # The fixed-string probe only runs when the pattern has regex metacharacters.
    _ZERO_MATCH_PROBES = (
        ("-i", "0 exact matches, but {total} case-insensitive match(es) in {n} file(s): "
               "{paths} — the pattern's casing may be wrong."),
        # rg skips dotdirs and .gitignore'd files by default; say so instead of a bare zero.
        ("--hidden --no-ignore", "0 matches in visible files, but {total} match(es) in {n} "
                                 "hidden or gitignored file(s): {paths} — these are excluded by default."),
        ("-F", "0 regex matches, but {total} literal match(es) in {n} file(s): {paths} — the "
               "pattern contains regex metacharacters that likely need escaping "
               "(or pass a simpler substring)."),
    )

    def _zero_match_probe(self, pattern: str, path: str, file_glob: Optional[str]) -> Optional[str]:
        """Steering hint for a 0-match content search, or None: a bare zero gives the
        model nothing to act on, so run cheap count-only rg probes (case-insensitive,
        hidden/ignored, fixed-string) and report the first that hits."""
        rg_executable = self._resolve_command('rg')
        if not rg_executable:
            return None
        rg = self._quote_executable(rg_executable)
        has_meta = bool(re.search(r"[.\[\](){}?*+^$\\|]", pattern))
        glob_expr = f" --glob {self._escape_shell_arg(file_glob)}" if file_glob else ""
        for flags, template in self._ZERO_MATCH_PROBES:
            if flags == "-F" and not has_meta:
                continue
            # The hidden/ignored probe keeps --no-ignore so project-local ignored
            # files stay diagnosable, but prunes heavyweight trees before rg recurses.
            if flags.startswith("--hidden"):
                glob_expr_probe = f"{glob_expr} {self._search_prune_glob_args()}"
            else:
                glob_expr_probe = glob_expr
            probe = self._exec(
                f"{rg} {flags} --count-matches{glob_expr_probe} "
                f"{self._escape_shell_arg(pattern)} {self._escape_native_tool_arg(path)} "
                f"2>/dev/null | head -50",
                timeout=30)
            total, per_file = 0, []
            for line in (probe.stdout or "").strip().splitlines():
                p, _sep, n = line.rpartition(":")
                if n.isdigit():
                    total += int(n)
                    per_file.append(p)
            if total > 0:
                extra = len(per_file) - 5
                paths = ", ".join(per_file[:5]) + (f" (+{extra} more)" if extra > 0 else "")
                return template.format(total=total, n=len(per_file), paths=paths)
        return None

    def _is_broad_local_search_root(self, path: str) -> bool:
        """Whether a no-rg LOCAL root (filesystem root, $HOME or an ancestor of it) is
        unsafe for recursive find. Controller paths never classify remotes."""
        from tools.environments.local import LocalEnvironment, _IS_WINDOWS, _msys_to_windows_path

        if not isinstance(self.env, LocalEnvironment):
            return False

        def normalized(value: str) -> str:
            if _IS_WINDOWS:
                value = _msys_to_windows_path(value).replace("\\", "/")
            if not os.path.isabs(value):
                value = os.path.join(getattr(self.env, "cwd", None) or self.cwd, value)
            return os.path.normcase(os.path.abspath(value))

        from tools import file_operations as _fo  # lazy: _HOME is monkeypatched there
        root = normalized(path)
        home = normalized(_fo._HOME)
        drive = os.path.splitdrive(root)[0]
        anchor = drive + os.sep if drive else os.path.abspath(os.sep)
        if root == os.path.normcase(anchor):
            return True
        try:
            common = os.path.commonpath([root, home])
        except ValueError:
            return False
        return root == home or common == root

    def _search_files(self, pattern: str, path: str | List[str], limit: int, offset: int,
                      order: str = "discovery") -> SearchResult:
        """Search for files by name (glob-like) across one or more roots: rg --files,
        else a bounded find. ``order``: "discovery" (fast, bounded) or "modified"
        (exact global newest-first; needs rg 14+ or GNU find)."""
        search_pattern = pattern if (not pattern.startswith('**/') and '/' not in pattern) \
            else pattern.split('/')[-1]
        roots = [path] if isinstance(path, str) else path
        if not roots:
            return SearchResult(error="File search requires at least one search root in 'path'.")

        # Prefer ripgrep: bounded parallel traversal with ignore semantics. Resolve
        # the engine and exact-order capability BEFORE admission so a queued request
        # does not occupy a root while doing command discovery.
        if self._has_command("rg"):
            rg_executable = self._resolve_command("rg") or "rg"
            if order == "modified":
                capability_error = self._modified_rg_capability_error(rg_executable)
                if capability_error:
                    return SearchResult(error=capability_error)
            keys = _filename_search_root_keys(self.env, roots, self.cwd)
            if not _acquire_filename_search_roots(keys):
                return SearchResult(error=_ADMISSION_INTERRUPTED_ERROR)
            try:
                return self._search_files_rg(search_pattern, path, limit, offset, order,
                                             rg_executable=rg_executable)
            finally:
                _release_filename_search_roots(keys)

        # A local find rooted at/above $HOME or a filesystem root can take minutes and
        # prompt on protected paths: refuse before invoking find.
        if any(self._is_broad_local_search_root(root) for root in roots):
            return SearchResult(error=(
                "Broad local file search without ripgrep is disabled because "
                "find cannot keep this traversal safely bounded. Install "
                "ripgrep or search a narrower directory."))
        if not self._has_command("find"):
            return SearchResult(
                error="File search requires 'rg' (ripgrep) or 'find'. "
                      "Install ripgrep for best results: "
                      "https://github.com/BurntSushi/ripgrep#installation")

        # Prune hidden descendant dirs (and hidden files, matching rg's default) while
        # still allowing an explicitly selected hidden root; dash-prefixed roots get
        # ``./`` so find doesn't parse them as options.
        find_roots = [f"./{root}" if root.startswith("-") else root for root in roots]
        q_roots = [self._escape_shell_arg(root) for root in find_roots]
        root_exemptions = "".join(f" ! -path {root}" for root in q_roots)
        hidden_prune = f" \\( -type d -name '.*'{root_exemptions} \\) -prune -o"
        protected_paths = [absolute for _r, _rel, absolute in self._effective_macos_search_exclusions(roots)]
        protected_prune = f" {self._prune_expr(protected_paths)} -o" if protected_paths else ""
        fetch_limit = offset + limit + 1
        base = (f"find {' '.join(q_roots)}{protected_prune}{hidden_prune} -type f "
                f"! -name '.*' -name {self._escape_shell_arg(search_pattern)}")
        if order == "modified":
            cmd = "set -o pipefail; " + base + f" -printf '%T@ %p\\n' 2>/dev/null | sort -rn | head -n {fetch_limit}"
        else:
            cmd = "set -o pipefail; " + base + f" -print 2>/dev/null | head -n {fetch_limit}"

        keys = _filename_search_root_keys(self.env, roots, self.cwd)
        if not _acquire_filename_search_roots(keys):
            return SearchResult(error=_ADMISSION_INTERRUPTED_ERROR)
        try:
            result = self._exec(cmd, timeout=60)
        finally:
            _release_filename_search_roots(keys)
        stdout, limit_reason = _search_stdout_and_limit(result)

        # Parse BEFORE classifying exit 141: under pipefail a bounded producer gets
        # SIGPIPE when head closes after fetch_limit rows — benign only when the
        # payload proves the bound was reached; a shorter payload is a hard failure.
        raw_files: List[str] = []
        for line in stdout.splitlines():
            if order == "modified":
                parts = line.split(" ", 1)
                if len(parts) != 2 or not parts[0].replace(".", "", 1).isdigit():
                    continue
                raw_files.append(parts[1])
            elif line:
                raw_files.append(line)
        bounded_sigpipe = result.exit_code == 141 and len(raw_files) >= fetch_limit
        if result.exit_code not in {0, 124} and not bounded_sigpipe:
            if order == "modified":
                return SearchResult(error=(
                    "Exact modification-time order requires GNU find with "
                    "-printf support; install ripgrep 14+ or use order='discovery'."))
            return SearchResult(error="File search failed while running bounded find traversal.")

        from tools.environments.local import LocalEnvironment, _IS_WINDOWS, _msys_to_windows_path
        if _IS_WINDOWS and isinstance(self.env, LocalEnvironment):
            raw_files = [_msys_to_windows_path(file_path) for file_path in raw_files]
        return SearchResult(
            files=raw_files[offset:offset + limit], total_count=len(raw_files),
            truncated=len(raw_files) > offset + limit or bool(limit_reason), limit_reason=limit_reason)

    def _search_files_rg(self, pattern: str, path: str | List[str], limit: int, offset: int,
                         order: str = "discovery", rg_executable: Optional[str] = None) -> SearchResult:
        """File-name search via ``rg --files`` (respects .gitignore, skips hidden dirs,
        parallel walk). Discovery order stays bounded and fast; exact modification-time
        ordering is explicit because it scans globally."""
        # Wrap bare names so -g matches at any depth (equivalent to find -name).
        glob_pattern = f"*{pattern}" if ('/' not in pattern and not pattern.startswith('*')) else pattern
        roots = [path] if isinstance(path, str) else path
        fetch_limit = limit + offset + 1
        effective_exclusions = self._effective_macos_search_exclusions(roots)
        scoped_common = None
        command_roots = roots
        if len(roots) > 1 and effective_exclusions and _posix_roots(roots):
            # Several roots: rg globs are root-relative, so cd to the common ancestor
            # and express roots + exclusions relative to it.
            cwd = getattr(self.env, "cwd", None) or self.cwd
            absolute_roots = [
                posixpath.normpath(root if posixpath.isabs(root) else posixpath.join(cwd, root))
                for root in roots]
            scoped_common = posixpath.commonpath(absolute_roots)
            command_roots = [posixpath.relpath(root, scoped_common) for root in absolute_roots]
            exclusion_terms = [
                f"--glob {self._escape_shell_arg(f'!{posixpath.relpath(absolute, scoped_common)}/**')}"
                for _r, _rel, absolute in effective_exclusions]
        else:
            exclusion_terms = [
                f"--glob {self._escape_shell_arg(f'!{relative}/**')}"
                for _r, relative, _abs in effective_exclusions]
        exclusion_globs = " ".join(dict.fromkeys(exclusion_terms))
        exclusion_args = f" {exclusion_globs}" if exclusion_globs else ""
        rg_executable = rg_executable or self._resolve_command("rg")
        if not rg_executable:
            return SearchResult(error="File search requires ripgrep (rg).")
        if order == "modified":
            capability_error = self._modified_rg_capability_error(rg_executable)
            if capability_error:
                return SearchResult(error=capability_error)
        rg = self._quote_executable(rg_executable)
        sort_arg = " --sortr=modified" if order == "modified" else ""
        root_args = " ".join(self._escape_native_tool_arg(root) for root in command_roots)
        cd_prefix = f"cd {self._escape_shell_arg(scoped_common)} && " if scoped_common else ""
        # ``--`` terminates options so a dash-prefixed root is never parsed as a flag.
        cmd = (f"set -o pipefail; {cd_prefix}{rg} --files{sort_arg} -g {self._escape_shell_arg(glob_pattern)}"
               f"{exclusion_args} -- {root_args} 2>/dev/null | head -n {fetch_limit}")
        result = self._exec(cmd, timeout=60)
        stdout, limit_reason = _search_stdout_and_limit(result)
        all_files = [f for f in stdout.splitlines() if f]
        if scoped_common:
            all_files = [
                f if posixpath.isabs(f) else posixpath.normpath(posixpath.join(scoped_common, f))
                for f in all_files]
        bounded_sigpipe = result.exit_code == 141 and len(all_files) >= fetch_limit
        if result.exit_code not in {0, 1, 124} and not bounded_sigpipe:
            if order == "modified":
                return SearchResult(error=(
                    "Exact modification-time order failed; ripgrep 14+ is "
                    "required. Upgrade ripgrep or use order='discovery'."))
            return SearchResult(error="File search failed while running ripgrep.")
        return SearchResult(
            files=all_files[offset:offset + limit], total_count=len(all_files),
            truncated=len(all_files) > offset + limit or bool(limit_reason), limit_reason=limit_reason)

    def _search_content(self, pattern: str, path: str, file_glob: Optional[str],
                        limit: int, offset: int, output_mode: str, context: int) -> SearchResult:
        """Content search: rg, else grep; attaches zero-match steering hints."""
        used_rg = self._has_command('rg')
        if used_rg:
            result = self._search_with_rg(pattern, path, file_glob, limit, offset, output_mode, context,
                                          rg_executable=self._resolve_command("rg") or "rg")
        elif self._has_command('grep'):
            result = self._search_with_grep(pattern, path, file_glob, limit, offset, output_mode, context)
        else:
            return SearchResult(
                error="Content search requires ripgrep (rg) or grep. "
                      "Install ripgrep: https://github.com/BurntSushi/ripgrep#installation")
        if (not result.error and result.total_count == 0
                and not result.matches and not result.files and not result.counts):
            try:
                hint = self._zero_match_probe(pattern, path, file_glob)
            except Exception:
                hint = None
            if hint:
                result.warning = hint if not result.warning else f"{result.warning} {hint}"
        # rg auto-enables --multiline for \n patterns, so the line-oriented
        # explanation only applies to the grep fallback.
        if used_rg:
            return result
        return _maybe_warn_line_oriented_newline_pattern(result, pattern)

    def _run_search_pipeline(self, cmd_parts: List[str], output_mode: str, limit: int,
                             offset: int, context: int, warning: Optional[str] = None,
                             line_cap: bool = False) -> SearchResult:
        """Run ``cmd_parts | head -n <fetch_limit>`` under pipefail and parse. Extra
        rows report the true total (context mode also emits "--" separators, so
        grab 200 more). pipefail keeps the engine's exit 2 alive across ``| head``
        (a truncating head makes rg exit 0 / grep 141, which the ==2 guard ignores).
        ``line_cap`` appends ``| cut -c1-2000`` for engines without --max-columns
        (grep): bounds giant single-line matches at the pipe layer; skipped for
        files_only/count where lines are paths/counts."""
        fetch_limit = limit + offset + (200 if context > 0 else 0)
        parts = cmd_parts + ["|", "head", "-n", str(fetch_limit)]
        if line_cap and output_mode not in ("files_only", "count"):
            parts += ["|", "cut", "-c1-2000"]
        result = self._exec("set -o pipefail; " + " ".join(parts), timeout=60)
        return _parse_search_output(result, output_mode, limit, offset, context, warning=warning)

    def _search_with_rg(self, pattern: str, path: str, file_glob: Optional[str],
                        limit: int, offset: int, output_mode: str, context: int,
                        rg_executable: Optional[str] = None) -> SearchResult:
        """Search using ripgrep."""
        rg_executable = rg_executable or self._resolve_command("rg")
        if not rg_executable:
            return SearchResult(error="Content search requires ripgrep (rg).")
        cmd_parts = [self._quote_executable(rg_executable), "--line-number", "--no-heading", "--with-filename"]
        # Giant-single-line containment (cline#13525): a match inside a multi-MB
        # single-line dump makes rg emit the ENTIRE line (``head -n`` counts lines).
        # --max-columns bounds each printed line at the rg layer; --max-columns-preview
        # keeps a truncated prefix so the model still sees the hit. 2000 cols exceeds
        # the 500-char content clamp, so nothing previously visible is lost.
        if output_mode not in ("files_only", "count"):
            cmd_parts.extend(["--max-columns", "2000", "--max-columns-preview"])
        # A regex \n hard-errors in line-oriented mode; enable -U up front and say so.
        multiline = _pattern_has_regex_newline(pattern)
        if multiline:
            cmd_parts.append("--multiline")
        if context > 0:
            cmd_parts.extend(["-C", str(context)])
        cmd_parts.extend(self._rg_exclusion_globs(path))
        if file_glob:
            cmd_parts.extend(["--glob", self._escape_shell_arg(file_glob)])
        if output_mode in _OUTPUT_MODE_FLAGS:
            cmd_parts.append(_OUTPUT_MODE_FLAGS[output_mode])
        cmd_parts.append(self._escape_shell_arg(pattern))
        # rg is a native Windows binary (winget/cargo/choco): needs C:/... not MSYS /c/...
        cmd_parts.append(self._escape_native_tool_arg(path))
        ml_note = (
            "Pattern contains \\n — multiline mode (-U) was enabled automatically "
            "so the regex can match across line boundaries."
        ) if multiline else None
        return self._run_search_pipeline(cmd_parts, output_mode, limit, offset, context, warning=ml_note)

    def _grep_cmd(self, head: List[str], pattern: str, output_mode: str, context: int,
                  file_glob: Optional[str] = None) -> List[str]:
        """``head`` + context/include/mode flags + quoted pattern (argument order is fixed)."""
        parts = list(head)
        if context > 0:
            parts.extend(["-C", str(context)])
        if file_glob:
            parts.extend(["--include", self._escape_shell_arg(file_glob)])
        if output_mode in _OUTPUT_MODE_FLAGS:
            parts.append(_OUTPUT_MODE_FLAGS[output_mode])
        parts.append(self._escape_shell_arg(pattern))
        return parts

    def _search_with_grep(self, pattern: str, path: str, file_glob: Optional[str],
                          limit: int, offset: int, output_mode: str, context: int) -> SearchResult:
        """Fallback search using grep."""
        # grep's --exclude-dir matches BASENAMES anywhere, so it can't express "only
        # the home-level Downloads"; route pruning through find's path-scoped -prune.
        protected_paths = self._protected_prune_paths(path)
        if protected_paths:
            return self._search_with_grep_pruned(
                pattern, path, file_glob, limit, offset, output_mode, context, protected_paths)
        # -H forces filenames; -E matches rg regex behavior; --exclude-dir='.*'
        # mirrors rg's hidden-dir default (.git/, .hub/index-cache/, ...).
        cmd_parts = self._grep_cmd(["grep", "-rnHE", "--exclude-dir='.*'"], pattern, output_mode, context, file_glob)
        # --exclude-dir applies to the root too, so "." would be excluded by '.*';
        # anchor relative paths at the shell's live $PWD.
        is_absolute = path.startswith(("/", "\\\\")) or bool(re.match(r"^[A-Za-z]:[\\/]", path))
        if is_absolute:
            search_root = self._escape_shell_arg(path)
        else:
            relative_path = path[2:] if path.startswith("./") else path
            search_root = '"$PWD"'
            if relative_path not in {"", "."}:
                search_root += f"/{self._escape_shell_arg(relative_path)}"
        cmd_parts.append(search_root)
        return self._run_search_pipeline(cmd_parts, output_mode, limit, offset, context, line_cap=True)

    def _search_with_grep_pruned(self, pattern: str, path: str, file_glob: Optional[str],
                                 limit: int, offset: int, output_mode: str, context: int,
                                 protected_paths: List[str]) -> SearchResult:
        """grep fallback with PATH-scoped protected-dir pruning: ``find ... -prune``
        enumerates files (traversal never enters protected dirs) and hands them to
        grep via ``-exec {} +``; hidden dirs pruned to mirror ``--exclude-dir='.*'``.
        Trade-off: find folds grep's exit code, so a hard grep error surfaces as an
        empty result — acceptable for this darwin-local-broad-search-only branch."""
        grep_parts = self._grep_cmd(["grep", "-nHE"], pattern, output_mode, context)
        find_parts = [
            "find", self._escape_shell_arg(path or "."),
            self._prune_expr(protected_paths), "-o",
            "\\( -type d -name '.*' \\) -prune", "-o",
            "-type f",
        ]
        if file_glob:
            find_parts.extend(["-name", self._escape_shell_arg(file_glob)])
        find_parts.extend(["-exec", *grep_parts, "{}", "+", "2>/dev/null"])
        return self._run_search_pipeline(find_parts, output_mode, limit, offset, context, line_cap=True)
