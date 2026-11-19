"""Non-executing structural sed mutation classification."""
import os
import re
import unicodedata

_SED_PROJECT_BASENAME_RE = re.compile(r'\.env(?:\.[^/]+)?|\.envrc', re.IGNORECASE)
# GNU sed's complete long-option namespace: --f is ambiguous with
# --follow-symlinks, whereas --fil uniquely owns a program filename.
_SED_LONG_OPTIONS = (
    "--binary", "--debug", "--expression", "--file", "--follow-symlinks",
    "--help", "--in-place", "--line-length", "--null-data", "--posix",
    "--quiet", "--regexp-extended", "--sandbox", "--separate", "--silent",
    "--unbuffered", "--version",
)


def _sed_detection_sources(command: str):
    """Project env -S through its existing non-executing argv parser."""
    from tools.approval_detection import (_scan_shell, _splice, _iter_shell_command_word_spans,
        _deobfuscate_shell_word_for_detection, _shell_segment_tokens, _shell_command_segment, _env_split_payload,
        _nested_payloads, _MAX_EVAL_NESTING)
    # Unlike full normalization, this projection preserves raw control quotes.
    # Only unquoted IFS expansion can split a shell word into sed and its flags.
    if "$IFS" in command or "${IFS" in command:
        edits = []
        ifs_re = re.compile(r'\$\{IFS\b[^}]*\}|\$IFS\b')
        pending_spans = [(0, len(command))]
        while pending_spans:
            begin, end = pending_spans.pop()
            for kind, start, stop, quote in _scan_shell(command, begin, end, subst="uq", brace=True,
                                                       comments=True):
                if kind == "subst" and stop is not None and (
                    command.startswith("$(", start) or command[start] == "`"
                ):
                    # Executable bodies have their own quote state; do not carry
                    # the surrounding argument's quotes into the nested scan.
                    opener = 2 if command.startswith("$(", start) else 1
                    pending_spans.append((start + opener, stop - 1))
                elif quote is None and kind in {"char", "subst"} and (match := ifs_re.match(command, start)):
                    edits.append((start, match.end(), " "))
        command = _splice(command, sorted(edits))
    pending, seen = [(command, 0)], set()
    while pending:
        source, depth = pending.pop()
        if source in seen:
            continue
        seen.add(source)
        yield source
        if depth >= _MAX_EVAL_NESTING:
            continue
        for payload, payload_depth in _nested_payloads(source, depth, None):
            pending.append((payload, max(depth + 1, payload_depth)))
        for start, _, word in _iter_shell_command_word_spans(source):
            name = os.path.basename(_deobfuscate_shell_word_for_detection(word))
            tokens = _shell_segment_tokens(_shell_command_segment(source, start), 0)
            if name == "env" and tokens and (payload := _env_split_payload(tokens)):
                pending.append((payload, depth + 1))
            elif name == "ssh" and tokens:
                # The first positional is the host; remaining words are a remote shell command.
                index = 1
                while index < len(tokens) and tokens[index].startswith('-'):
                    option = tokens[index]
                    index += 1
                    if option in {'-B', '-b', '-c', '-D', '-E', '-e', '-F', '-I', '-i', '-J', '-L', '-l', '-m', '-O', '-o', '-p', '-Q', '-R', '-S', '-W', '-w'}:
                        index += 1
                if index + 1 < len(tokens):
                    pending.append((' '.join(tokens[index + 1:]), depth + 1))


def _sed_command_tokens(segment: str) -> list[tuple[str, str]] | None:
    """Remove redirections and retain each argv word's source provenance."""
    from tools.approval_detection import (_scan_shell, _SHELL_REDIRECTION_RE, _read_shell_word,
        _splice, _shell_tokens_with_spans)
    edits, skip = [], -1
    for kind, start, _, quote in _scan_shell(segment, subst="uq", brace=True):
        if start < skip or quote is not None or kind != "char":
            continue
        # An IO-number prefix must begin a word. In `.env2>out`, 2 belongs
        # to the filename; only the `>` is a redirection operator.
        if segment[start].isdigit() and start and not segment[start - 1].isspace():
            continue
        if redirect := _SHELL_REDIRECTION_RE.match(segment, start):
            _, skip, _ = _read_shell_word(segment, redirect.end())
            edits.append((start, skip, " "))
    source = _splice(segment, edits)
    tokens = _shell_tokens_with_spans(source, 0)
    if tokens is None:
        return None
    return [(value, source[start:end]) for value, start, end, _ in tokens]


def _sed_in_place_findings(command: str):
    """Inspect only actual sed commands, using the existing quote-aware scanner.

    Option flags are case-sensitive. GNU -i owns the remainder of its short
    bundle as an optional backup suffix (unlike -n); only file operands can
    identify a protected target. No shell or editor is executed here.
    """
    from tools.approval_detection import (_iter_shell_command_word_spans, _deobfuscate_shell_word_for_detection,
        _shell_command_segment, _rewrite_resolved_user_home, _rewrite_resolved_hermes_home,
        _strip_shell_word_syntax, _USER_SENSITIVE_WRITE_TARGET,
        _HERMES_CONFIG_PATH, _HERMES_ENV_PATH, _SYSTEM_CONFIG_PATH, _SSH_SENSITIVE_PATH, _HERMES_STORE_ROOT)
    _SED_USER_TARGET_RE = re.compile(_USER_SENSITIVE_WRITE_TARGET, re.IGNORECASE)
    _SED_HERMES_TARGET_RE = re.compile(rf'(?:{_HERMES_CONFIG_PATH}|{_HERMES_ENV_PATH}|{_HERMES_STORE_ROOT})', re.IGNORECASE)
    _SED_SYSTEM_TARGET_RE = re.compile(_SYSTEM_CONFIG_PATH, re.IGNORECASE)
    _SED_SSH_TARGET_RE = re.compile(_SSH_SENSITIVE_PATH, re.IGNORECASE)
    for start, _, word in _iter_shell_command_word_spans(command):
        if os.path.basename(_deobfuscate_shell_word_for_detection(word)) != "sed":
            continue
        # Fold source words before argv decoding dissolves native separators or
        # removes the quotes that keep spaced HOME prefixes in one operand.
        segment = _rewrite_resolved_user_home(
            _rewrite_resolved_hermes_home(_shell_command_segment(command, start))
        )
        args = _sed_command_tokens(segment)
        if not args:
            continue
        parsed = _sed_operands(args)
        if parsed is None:
            continue
        operands, explicit_program = parsed
        # Without -e/-f the first positional is sed's program, not an input file.
        for operand in operands if explicit_program else operands[1:]:
            # Normalize the original word, not a re-quoted decoded value: invented
            # single quotes would turn unknown expansions into literal filenames.
            from tools.ansi_strip import strip_ansi
            from tools.approval_detection import expand_ansi_c_quotes
            target = _strip_shell_word_syntax(unicodedata.normalize(
                "NFKC", strip_ansi(expand_ansi_c_quotes(operand)).replace("\x00", "")))
            user_target = _SED_USER_TARGET_RE.match(target)
            # A slash-terminated inventory match marks a protected subtree;
            # bare filename entries must still consume the entire operand.
            sensitive_user_target = user_target is not None and (
                user_target.end() == len(target) or user_target.group().endswith("/")
            )
            if _SED_SYSTEM_TARGET_RE.match(target):
                yield "in-place edit of system config"
            elif _SED_HERMES_TARGET_RE.fullmatch(target):
                yield "in-place edit of Hermes config/env"
            elif _SED_SSH_TARGET_RE.match(target) or sensitive_user_target:
                yield "in-place edit of sensitive credential/SSH/shell-rc path"
            elif _SED_PROJECT_BASENAME_RE.fullmatch(os.path.basename(target)):
                yield "in-place edit of project env file"




def _sed_operands(args):
    in_place, options, explicit_program = False, True, False
    operands = []
    index = 1
    while index < len(args):
        token, raw = args[index]
        if options and token == "--":
            options = False
        elif options and token.startswith("--"):
            option, equals, _ = token.partition("=")
            matches = [name for name in _SED_LONG_OPTIONS if name.startswith(option)]
            if len(matches) != 1:
                in_place = False  # GNU sed rejects unknown/ambiguous options.
                break
            option = matches[0]
            if equals and option not in {"--in-place", "--expression", "--file", "--line-length"}:
                in_place = False  # No-argument options cannot own an attached value.
                break
            if option == "--in-place":
                in_place = True
            elif option in {"--expression", "--file", "--line-length"}:
                explicit_program = explicit_program or option in {"--expression", "--file"}
                if not equals:
                    index += 1
        elif options and token.startswith("-") and token != "-":
            for offset, flag in enumerate(token[1:], 2):
                if flag == "i":
                    in_place = True
                    break  # the rest is GNU sed's optional backup suffix
                if flag in "efl":
                    explicit_program = explicit_program or flag in "ef"
                    if offset == len(token):
                        index += 1
                    break  # the rest (or next token) is this option's value
        else:
            operands.append(raw)
        index += 1
    if not in_place:
        return None
    return operands, explicit_program



def sed_findings(command, only):
    """Classify sed in both full and content-only scans."""
    if only is None or only.intersection({
        "in-place edit of system config", "in-place edit of Hermes config/env",
        "in-place edit of sensitive credential/SSH/shell-rc path", "in-place edit of project env file",
    }):
        # Literal quoting/escaping can splice "sed"; expansions and controls need the full scanner.
        plain = unicodedata.normalize("NFKC", command)
        if not any(marker in plain for marker in ("$", "`", "\x1b", "\x00")):
            if "sed" not in re.sub(r"[\\'\"]", "", plain):
                return
        for source in _sed_detection_sources(command):
            for finding in _sed_in_place_findings(source):
                if only is None or finding in only:
                    yield finding
