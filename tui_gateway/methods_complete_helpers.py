"""Completion helpers (@-mention / path fuzzy ranking, repo file listing) for the complete.* RPCs.

Bodies are rebound onto server.py's globals at install time (see
method_ctx.bind_module), so they reference server.py globals bare.
"""

from __future__ import annotations

import re
import threading

from .method_ctx import HandlerRegistry, bind_module

_registry = HandlerRegistry()

_FUZZY_CACHE_TTL_S = 5.0
_FUZZY_CACHE_MAX_FILES = 20000
_FUZZY_FALLBACK_EXCLUDES = frozenset(
    {".git", ".hg", ".svn", ".next", ".cache", ".venv", "venv", "node_modules", "__pycache__",
     "dist", "build", "target", ".mypy_cache", ".pytest_cache", ".ruff_cache"})
_fuzzy_cache_lock = threading.Lock()
_fuzzy_cache: dict[str, tuple[float, list[str]]] = {}


def _git_repo_files(root: str):
    """Yield ``git ls-files`` paths (tracked + untracked) relative to ``root``; empty outside a
    repo or on git failure/timeout. Entries above ``root`` are skipped (Cmd-P workspace scope)."""
    from hermes_cli._subprocess_compat import windows_hide_flags
    run_kw = dict(capture_output=True, timeout=2.0, check=False, stdin=subprocess.DEVNULL, creationflags=windows_hide_flags())
    try:
        top_result = subprocess.run(["git", "-C", root, "rev-parse", "--show-toplevel"], **run_kw)
        if top_result.returncode != 0:
            return
        top = top_result.stdout.decode("utf-8", "replace").strip()
        list_result = subprocess.run(
            ["git", "-C", top, "ls-files", "-z", "--cached", "--others", "--exclude-standard"], **run_kw)
        if list_result.returncode != 0:
            return
    except (OSError, subprocess.TimeoutExpired):
        return
    for p in list_result.stdout.decode("utf-8", "replace").split("\0"):
        if p:
            rel = os.path.relpath(os.path.join(top, p), root).replace(os.sep, "/")
            if not rel.startswith("../"):
                yield rel


def _walk_repo_files(root: str):
    """Non-git fallback: ``os.walk`` skipping vendor/build dirs + dot-dirs; dotfiles survive
    (the ranker decides based on whether the query starts with `.`)."""
    try:
        for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
            dirnames[:] = [d for d in dirnames if d not in _FUZZY_FALLBACK_EXCLUDES and not d.startswith(".")]
            rel_dir = os.path.relpath(dirpath, root)
            for f in filenames:
                yield (f if rel_dir == "." else f"{rel_dir}/{f}").replace(os.sep, "/")
    except OSError:
        return


def _list_repo_files(root: str) -> list[str]:
    """File paths relative to ``root`` (git listing, else a bounded walk), cached per-root for
    ``_FUZZY_CACHE_TTL_S`` so rapid keystrokes don't respawn git."""
    now = time.monotonic()
    with _fuzzy_cache_lock:
        cached = _fuzzy_cache.get(root)
        if cached and now - cached[0] < _FUZZY_CACHE_TTL_S:
            return cached[1]
    from itertools import islice
    files = list(islice(_git_repo_files(root), _FUZZY_CACHE_MAX_FILES))
    if not files:
        files = list(islice(_walk_repo_files(root), _FUZZY_CACHE_MAX_FILES))
    with _fuzzy_cache_lock:
        # The TTL is only consulted on read, so a root that stops being queried (one .worktrees/<id>
        # root per worktree flow, MBs each) would stay pinned for the process lifetime (#62950).
        for stale in [r for r, (ts, _) in _fuzzy_cache.items() if now - ts >= _FUZZY_CACHE_TTL_S]:
            del _fuzzy_cache[stale]
        _fuzzy_cache[root] = (now, files)
    return files


# `@symbol:` index: a regex mini-ctags over `_list_repo_files`, cached per root. Entries are
# ``(name, kind, relpath)``. Patterns are keyed by language so a C-family "type name(" shape
# never fires on Python/JS lines (where `await foo(` / `print foo(` would read as a definition).
_SYMBOL_CACHE_TTL_S = 15.0
_SYMBOL_MAX_FILES = 4000
_SYMBOL_MAX_RESULTS = 4000
_SYMBOL_MAX_FILE_BYTES = 1_500_000
_SYMBOL_MAX_LINES = 6000
_symbol_cache_lock = threading.Lock()
_symbol_cache: dict[str, tuple[float, list[tuple[str, str, str]]]] = {}

# Leading words that make a `word name(` line a statement, not a definition.
_SYMBOL_STMT_KW = (
    r"(?!(?:return|else|if|while|for|foreach|switch|case|do|new|delete|throw|goto|await|yield|"
    r"typeof|sizeof|nameof|using|lock|fixed|checked|unchecked|catch|synchronized|co_return|"
    r"co_await|co_yield|not|and|or|in|is|as|echo|print)\b)")
_JVM_MODS = (r"(?:public|private|protected|internal|static|final|abstract|synchronized|native|virtual|"
             r"override|async|sealed|extern|unsafe|partial|new|default|strictfp|readonly)")
_C_FUNC = re.compile(
    # [template<..>] <return type words/ptrs/refs> [Qual::]name( with no `;` after it (or a C# `=>` body):
    # a definition, not a call or a prototype
    r"^\s*(?:template\s*<[^>]*>\s*)?" + _SYMBOL_STMT_KW
    + r"(?:[A-Za-z_][\w:<>,]*[\s*&]+)+(?:[A-Za-z_]\w*::)*(~?[A-Za-z_]\w*)\s*\([^;]*(?:$|=>)")
_C_QUALIFIED = re.compile(r"^\s*(?:[A-Za-z_]\w*::)+(~?[A-Za-z_]\w*)\s*\([^;]*$")  # Foo::Foo(...) ctor
_C_TYPE = re.compile(r"^\s*(?:typedef\s+)?(?:class|struct|union|enum(?:\s+class)?)\s+([A-Za-z_]\w*)\s*(?:[:{]|$)")
_JVM_TYPE = re.compile(
    r"^\s*(?:@\w+(?:\([^)]*\))?\s+)*(?:" + _JVM_MODS + r"\s+)*"
    r"(?:class|interface|enum|record|struct|@interface)\s+([A-Za-z_]\w*)")
_JVM_CTOR = re.compile(  # `public Foo(int x) {` / `public static <T> List<T> of(` — modifiers required
    r"^\s*(?:" + _JVM_MODS + r"\s+)+(?:<[^>]*>\s*)?(?:[\w.<>\[\],?]+\s+)?([A-Za-z_]\w*)\s*\([^;]*(?:$|=>)")
_MOD_FUNC = re.compile(  # Kotlin `fun` / Swift `func` / PHP `function`, with modifiers and receivers
    r"^\s*(?:@\w+\s+)*(?:(?:public|private|protected|internal|open|override|static|final|abstract|"
    r"suspend|inline|operator|infix|tailrec|fileprivate|mutating|class)\s+)*"
    r"(?:fun|func|function)\s+(?:<[^>]*>\s*)?&?(?:[\w.]+\.)?([A-Za-z_]\w*)")
_MOD_TYPE = re.compile(
    r"^\s*(?:@\w+\s+)*(?:(?:public|private|protected|internal|open|final|abstract|sealed|data|"
    r"enum|inner|value|annotation|fileprivate|case|implicit|readonly)\s+)*"
    r"(?:class|interface|object|protocol|struct|enum|trait|extension)\s+([A-Za-z_]\w*)")

_SYMBOL_PATTERNS: dict[str, tuple[tuple[str, "re.Pattern[str]"], ...]] = {
    "py": (
        ("def", re.compile(r"^\s*(?:async\s+)?def\s+([A-Za-z_]\w*)")),
        ("class", re.compile(r"^\s*class\s+([A-Za-z_]\w*)")),
    ),
    "js": (
        ("func", re.compile(r"^\s*(?:export\s+(?:default\s+)?)?(?:async\s+)?function\s*\*?\s*([A-Za-z_$][\w$]*)")),
        ("class", re.compile(r"^\s*(?:export\s+(?:default\s+)?)?(?:abstract\s+)?class\s+([A-Za-z_$][\w$]*)")),
        ("const", re.compile(
            r"^\s*(?:export\s+)?const\s+([A-Za-z_$][\w$]*)\s*(?::[^=]+)?=\s*(?:async\s*)?"
            r"(?:\([^)]*\)|[A-Za-z_$][\w$]*)\s*(?::[^=]+)?=>")),
        ("type", re.compile(r"^\s*(?:export\s+)?(?:declare\s+)?(?:type|interface|enum)\s+([A-Za-z_$][\w$]*)")),
    ),
    "go": (
        ("func", re.compile(r"^\s*func\s+(?:\([^)]*\)\s*)?([A-Za-z_]\w*)")),
        ("type", re.compile(r"^\s*type\s+([A-Za-z_]\w*)\s+(?:struct|interface)")),
    ),
    "rust": (
        ("fn", re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:const\s+)?(?:async\s+)?(?:unsafe\s+)?fn\s+([A-Za-z_]\w*)")),
        ("type", re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:struct|enum|trait|union)\s+([A-Za-z_]\w*)")),
    ),
    "ruby": (
        ("def", re.compile(r"^\s*def\s+(?:self\.)?([A-Za-z_]\w*[!?=]?)")),
        ("class", re.compile(r"^\s*(?:class|module)\s+([A-Z]\w*)")),
    ),
    "c": (("type", _C_TYPE), ("func", _C_FUNC), ("func", _C_QUALIFIED)),
    "jvm": (("type", _JVM_TYPE), ("method", _JVM_CTOR), ("method", _C_FUNC)),
    "kotlin": (("func", _MOD_FUNC), ("type", _MOD_TYPE)),
    "swift": (("func", _MOD_FUNC), ("type", _MOD_TYPE)),
    "php": (("func", _MOD_FUNC), ("type", _MOD_TYPE)),
    "scala": (
        ("def", re.compile(r"^\s*(?:(?:override|private|protected|final|implicit|inline)\s+)*def\s+([A-Za-z_]\w*)")),
        ("type", _MOD_TYPE),
    ),
}

# Only extensions with a pattern set above are scanned; each maps to its language.
_SYMBOL_EXTS: dict[str, str] = {
    **dict.fromkeys((".py", ".pyi"), "py"),
    **dict.fromkeys((".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".mts", ".cts"), "js"),
    ".go": "go", ".rs": "rust", ".rb": "ruby",
    **dict.fromkeys((".c", ".h", ".cc", ".cpp", ".cxx", ".hh", ".hpp", ".hxx"), "c"),
    **dict.fromkeys((".java", ".cs"), "jvm"),
    **dict.fromkeys((".kt", ".kts"), "kotlin"),
    ".swift": "swift", ".php": "php", ".scala": "scala",
}


def _scan_symbols(root: str) -> list[tuple[str, str, str]]:
    """``(name, kind, relpath)`` definitions across ``root`` for `@symbol:`: regex per language over
    ``_list_repo_files`` (git-ignore + workspace scope), bounded per file and overall, cached per root."""
    now = time.monotonic()
    with _symbol_cache_lock:
        cached = _symbol_cache.get(root)
        if cached and now - cached[0] < _SYMBOL_CACHE_TTL_S:
            return cached[1]
    out: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str]] = set()
    scanned = 0
    for rel in _list_repo_files(root):
        lang = _SYMBOL_EXTS.get(os.path.splitext(rel)[1].lower())
        if lang is None:
            continue
        scanned += 1
        if scanned > _SYMBOL_MAX_FILES or len(out) >= _SYMBOL_MAX_RESULTS:
            break
        patterns = _SYMBOL_PATTERNS[lang]
        abs_path = os.path.join(root, rel)
        try:
            if os.path.getsize(abs_path) > _SYMBOL_MAX_FILE_BYTES:
                continue
            with open(abs_path, encoding="utf-8", errors="ignore") as fh:
                for i, line in enumerate(fh):
                    if i >= _SYMBOL_MAX_LINES or len(out) >= _SYMBOL_MAX_RESULTS:
                        break
                    if len(line) > 400:
                        continue
                    for kind, pattern in patterns:
                        if m := pattern.match(line):
                            if (m.group(1), rel) not in seen:
                                seen.add((m.group(1), rel))
                                out.append((m.group(1), kind, rel))
                            break
        except (OSError, ValueError):
            continue
    with _symbol_cache_lock:
        for stale in [r for r, (ts, _) in _symbol_cache.items() if now - ts >= _SYMBOL_CACHE_TTL_S]:
            del _symbol_cache[stale]
        _symbol_cache[root] = (now, out)
    return out


def _fuzzy_basename_rank(name: str, query: str) -> tuple[int, int] | None:
    """Rank ``name`` against ``query`` as (tier, len(name)); lower wins, None rejects.
    Tiers: 0 exact · 1 prefix · 2 word-boundary/camelCase hit (`chrome` → `appChrome.tsx`)
    · 3 substring · 4 subsequence (query chars appear in order)."""
    if not query:
        return (3, len(name))
    nl, ql = name.lower(), query.lower()
    if nl == ql:
        return (0, len(name))
    if nl.startswith(ql):
        return (1, len(name))
    # Word boundaries: split on -_. and camelCase (`appChrome` → ["app","Chrome"]); cheap
    # approximation, falls through to substring/subsequence if it misses.
    parts: list[str] = []
    buf = ""
    for ch in name:
        if ch in "-_." or (ch.isupper() and buf and not buf[-1].isupper()):
            parts += [buf] if buf else []
            buf = ch if ch not in "-_." else ""
        else:
            buf += ch
    if any(p.lower().startswith(ql) for p in parts + ([buf] if buf else [])):
        return (2, len(name))
    if ql in nl:
        return (3, len(name))
    it = iter(nl)
    return (4, len(name)) if all(any(c == q for c in it) for q in ql) else None


def _abs_completion_prefix_exists(path_part: str) -> bool:
    """True when ``path_part`` reads sensibly as an absolute path (parent exists and a
    partially-typed final segment matches an entry): decides `@/foo` = `/foo` vs cwd `foo`."""
    expanded = _normalize_completion_path(path_part)
    parent = os.path.dirname(expanded.rstrip("/")) or "/"
    tail = os.path.basename(expanded.rstrip("/"))
    if not os.path.isdir(parent):
        return False
    if not tail or expanded.endswith("/"):
        return os.path.isdir(expanded) or expanded == "/"
    try:
        tail_lower = tail.lower()
        return any(e.lower().startswith(tail_lower) for e in os.listdir(parent))
    except OSError:
        return False


_DETAILS_SECTIONS = ("thinking", "tools", "subagents", "activity")
_DETAILS_MODES = ("hidden", "collapsed", "expanded")


def _details_root_meta(candidate: str) -> str:
    if candidate in _DETAILS_SECTIONS:
        return "section override"
    return "cycle global mode" if candidate == "cycle" else "global mode"


def _details_completions(text: str) -> list[dict] | None:
    """Argument completions for ``/details [section] [mode]``; None when ``text`` is not that command."""
    if not text.lower().startswith("/details"):
        return None
    stripped = text.strip()
    if stripped and not "/details".startswith(stripped.lower().split()[0]):
        return None
    body = text[len("/details") :].removeprefix(" ")
    parts = body.split()
    trailing = text.endswith(" ")
    root_candidates = (*_DETAILS_MODES, "cycle", *_DETAILS_SECTIONS)
    if not body or (not parts and trailing):
        lead = "" if trailing else " "
        return [_item(f"{lead}{c}", _details_root_meta(c)) for c in root_candidates]
    if len(parts) == 1 and not trailing:
        prefix = parts[0].lower()
        return [_item(c, _details_root_meta(c)) for c in root_candidates if c.startswith(prefix) and c != prefix]
    section = parts[0].lower() if parts else ""
    if section not in _DETAILS_SECTIONS:
        return []

    def section_meta(candidate: str) -> str:
        return f"clear {section} override" if candidate == "reset" else f"set {section}"
    mode_candidates = (*_DETAILS_MODES, "reset")
    if len(parts) == 1:  # trailing space after the section
        return [_item(c, section_meta(c)) for c in mode_candidates]
    if len(parts) == 2 and not trailing:
        prefix = parts[1].lower()
        return [_item(c, section_meta(c)) for c in mode_candidates if c.startswith(prefix) and c != prefix]
    return []


def _model_picker_context(agent):
    """Layer live session state onto config without losing custom identity."""
    from hermes_cli.inventory import load_picker_context
    ctx = load_picker_context()
    provider, base_url, model = (getattr(agent, k, "") if agent else "" for k in ("provider", "base_url", "model"))
    if str(provider or "").strip().lower() == "custom":
        try:
            from hermes_cli.runtime_provider import canonical_custom_identity
            provider = canonical_custom_identity(
                base_url=base_url or None, config_provider=ctx.current_provider, model=model or None) or provider
        except Exception:
            logger.debug("custom provider identity recovery failed (model picker)", exc_info=True)
    return ctx.with_overrides(
        current_provider=provider, current_model=model or _resolve_model(), current_base_url=base_url)


def register(server) -> None:
    """Publish this module's helpers + handlers onto ``server``, rebound to its globals."""
    bind_module(globals(), server, skip=("_",))
