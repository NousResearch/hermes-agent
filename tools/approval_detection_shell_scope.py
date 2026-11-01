"""Control-flow shape of one shell command, for same-command variable resolution. Detection only.

``approval_detection_assignments`` may treat an assignment as PROOF of a variable's value at a later
use only if the assignment certainly ran, in the same shell, before that use. Text order alone is
not proof:

* ``false && X=echo; $X``: the assignment is conditional;
* ``(X=echo); $X``, ``X=echo | cat; $X``, ``X=echo & $X``: it ran in a subshell;
* ``f() { X=echo; }``: a function body runs when called, not where it is written;
* ``if c; then X=echo; fi; $X``: only one branch runs.

This module walks the command once and answers ``dominates(binding_offset, use_offset)``: is the
binding at a position that always executes, in the use's own shell, before the use? Any doubt,
including a parse this walker does not understand, means "no". The binding then only ADDS a
possible value and never replaces the earlier ones. It also reports which heredoc bodies a shell
executes (``bash <<EOF``), because those are commands rather than stdin data.
"""

import bisect
import os
import re
from dataclasses import dataclass, field

_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

# Words that open / continue / close compound commands at a command position.
_LOOP_OPENERS = frozenset({"while", "until", "for", "select"})
_OPENERS = _LOOP_OPENERS | {"if", "case"}
_CONTINUERS = frozenset({"then", "else", "elif", "do"})
_CLOSERS = {"fi": "if", "done": "loop", "esac": "case"}
# Consumers that execute a heredoc body as shell commands (``ssh`` runs it in the remote shell).
_HEREDOC_SHELL_CONSUMERS = frozenset({"bash", "sh", "zsh", "ksh", "dash", "ssh", "source", "."})


@dataclass(frozen=True)
class Ctx:
    """Where an offset sits: group (a brace/paren/compound body), list (``;``/newline/``&``
    separated) and element (``&&``/``||``/``|`` separated) inside that list."""
    group: int
    lst: int
    elem: int


@dataclass(frozen=True)
class HeredocBody:
    start: int
    end: int
    executed: bool   # the consumer is a shell, so the body is commands, not data
    # The consumer's command word is an expansion (`$S <<EOF`): which program reads the body is not
    # in the text, so it counts as executed here. The resolver may prove it a non-shell
    # (``approval_detection_assignments._executed_heredocs``).
    owner: str = ""
    owner_start: int = -1
    owner_dynamic: bool = False
    # The consumer's OUTPUT goes to a shell (``cat <<EOF | bash``), or to a pipe reader the text
    # does not name. Resolving the consumer to a non-shell never makes such a body data.
    piped: bool = False


class ShellScope:
    def __init__(self, command: str):
        self.command = command
        self.heredocs = _heredoc_bodies(command)
        self.reliable = True
        self._parent: list[Ctx | None] = [None]
        self._floating: list[bool] = [False]    # function body: runs when called
        self._loop: list[bool] = [False]        # loop body: runs repeatedly
        self._ops: dict[tuple[int, int], dict[int, str]] = {}
        self._isolated: set[Ctx] = set()        # pipeline element: a subshell
        self._background: set[tuple[int, int]] = set()
        self._offsets: list[int] = []
        self._ctxs: list[Ctx] = []
        text = _blank_spans(command, [(h.start, h.end) for h in self.heredocs])
        try:
            self._walk(text, 0, len(text), Ctx(0, 0, 0))
        except _Unparsed:
            self.reliable = False

    # ---- queries --------------------------------------------------------------------------------

    def ctx_at(self, offset: int) -> Ctx:
        index = bisect.bisect_right(self._offsets, offset) - 1
        return self._ctxs[index] if index >= 0 else Ctx(0, 0, 0)

    def floating(self, offset: int) -> bool:
        return self._floating[self.ctx_at(offset).group]

    def in_loop_with(self, first: int, second: int) -> bool:
        """Both offsets sit inside one loop body (or *second* is in a function body), so code
        written after *second* may still run before it."""
        if self.floating(second):
            return True
        return bool(self._loop_groups(first) & self._loop_groups(second))

    def dominates(self, binding: int, use: int) -> bool:
        """True only if the command at *binding* always runs, in the same shell, before *use*."""
        if not self.reliable or binding >= use:
            return False
        if any(h.start <= binding < h.end for h in self.heredocs):
            return False
        b, cur = self.ctx_at(binding), self.ctx_at(use)
        if self._floating[b.group]:
            return False
        while cur.group != b.group:
            parent = self._parent[cur.group]
            if parent is None:
                return False
            cur = parent
        same_elem = (cur.lst, cur.elem) == (b.lst, b.elem)
        if not same_elem and (b in self._isolated or (b.group, b.lst) in self._background):
            return False
        if cur.lst != b.lst:
            return cur.lst > b.lst and b.elem == 0
        if cur.elem <= b.elem:
            return cur.elem == b.elem
        ops = self._ops.get((b.group, b.lst), {})
        return b.elem == 0 or all(ops.get(k) == "&&" for k in range(b.elem + 1, cur.elem + 1))

    def _loop_groups(self, offset: int) -> set[int]:
        found, ctx = set(), self.ctx_at(offset)
        group: int | None = ctx.group
        while group is not None:
            if self._loop[group]:
                found.add(group)
            parent = self._parent[group]
            group = parent.group if parent else None
        return found

    # ---- walker ---------------------------------------------------------------------------------

    def _mark(self, offset: int, ctx: Ctx) -> None:
        self._offsets.append(offset)
        self._ctxs.append(ctx)

    def _new_group(self, parent: Ctx, *, floating: bool = False, loop: bool = False) -> Ctx:
        self._parent.append(parent)
        self._floating.append(floating or self._floating[parent.group])
        self._loop.append(loop)
        return Ctx(len(self._parent) - 1, 0, 0)

    def _walk(self, text: str, pos: int, end: int, ctx: Ctx) -> None:
        st = _WalkState(text, pos, end, ctx)
        self._mark(pos, ctx)
        handlers = (self._skip_trivia, self._case_arm_end, self._separator, self._redirection,
                    self._operator, self._open_paren, self._close_paren)
        while st.pos < st.end:
            if any(handler(st) for handler in handlers):
                continue
            self._word(st)
        if st.stack:
            raise _Unparsed

    def _advance(self, st: "_WalkState", width: int, ctx: Ctx | None = None) -> None:
        """Move past an operator, open a new command slot at the new position."""
        if ctx is not None:
            st.ctx = ctx
        st.pos += width
        self._mark(st.pos, st.ctx)
        st.at_cmd = True

    def _skip_trivia(self, st: "_WalkState") -> bool:
        from tools.approval_detection import _is_shell_comment_start
        ch, text = st.text[st.pos], st.text
        if ch in " \t" or text.startswith("\\\n", st.pos):
            st.pos += 1 if ch in " \t" else 2
            return True
        if ch == "#" and _is_shell_comment_start(text, st.pos):
            newline = text.find("\n", st.pos, st.end)
            st.pos = st.end if newline < 0 else newline
            return True
        return False

    def _case_arm_end(self, st: "_WalkState") -> bool:
        if not st.text.startswith((";;", ";&"), st.pos):
            return False
        if not st.stack or st.stack[-1][0] != "case":
            raise _Unparsed
        width = 3 if st.text.startswith(";;&", st.pos) else 2
        self._advance(st, width, self._new_group(st.stack[-1][1]))
        return True

    def _separator(self, st: "_WalkState") -> bool:
        if st.text[st.pos] not in ";\n":
            return False
        self._advance(st, 1, Ctx(st.ctx.group, st.ctx.lst + 1, 0))
        return True

    def _redirection(self, st: "_WalkState") -> bool:
        from tools.approval_detection import _REDIRECTION_OPERATOR_RE, _read_shell_word
        redirect = _REDIRECTION_OPERATOR_RE.match(st.text, st.pos)
        if not redirect or st.text.startswith(("<(", ">("), st.pos):
            return False
        st.pos = _read_shell_word(st.text, redirect.end())[1]
        return True

    def _operator(self, st: "_WalkState") -> bool:
        text, pos, ctx = st.text, st.pos, st.ctx
        if text.startswith(("&&", "||"), pos):
            self._advance(st, 2, self._next_elem(ctx, text[pos:pos + 2]))
        elif text[pos] == "|":
            self._isolated.add(ctx)
            nxt = self._next_elem(ctx, "|")
            self._isolated.add(nxt)
            self._advance(st, 2 if text.startswith("|&", pos) else 1, nxt)
        elif text[pos] == "&":
            self._background.add((ctx.group, ctx.lst))
            self._advance(st, 1, Ctx(ctx.group, ctx.lst + 1, 0))
        else:
            return False
        return True

    def _open_paren(self, st: "_WalkState") -> bool:
        text, pos = st.text, st.pos
        ch = text[pos]
        if not (ch == "(" or text.startswith(("<(", ">("), pos)):
            return False
        if ch == "(" and st.last_name and text[pos + 1:st.end].lstrip(" \t").startswith(")"):
            st.pos = text.index(")", pos) + 1      # `name()`: a function definition follows
            st.function_next, st.last_name, st.at_cmd = True, None, True
            return True
        st.stack.append(("(", st.ctx, False))
        group = self._new_group(st.ctx, floating=st.function_next)
        st.function_next = False
        self._advance(st, 1 if ch == "(" else 2, group)
        return True

    def _close_paren(self, st: "_WalkState") -> bool:
        if st.text[st.pos] != ")":
            return False
        st.pos += 1
        if st.stack and st.stack[-1][0] == "(":
            st.ctx, st.at_cmd = st.stack.pop()[1], False
        elif st.stack and st.stack[-1][0] == "case":
            st.ctx, st.at_cmd = self._new_group(st.stack[-1][1]), True   # a case arm body
        else:
            raise _Unparsed
        self._mark(st.pos, st.ctx)
        return True

    def _word(self, st: "_WalkState") -> None:
        from tools.approval_detection import _read_shell_word, _scan_shell
        start, stop, word = _read_shell_word(st.text, st.pos)
        if start == stop:
            st.pos += 1
            return
        st.last_name = None
        if st.at_cmd and word == "function":
            _, stop, name = _read_shell_word(st.text, stop)    # the function name
            st.function_next, st.last_name, st.at_cmd = True, name, True
        elif not self._keyword(st, word):
            if st.at_cmd and _NAME_RE.fullmatch(word):
                st.last_name = word
            st.at_cmd = st.at_cmd and "=" in word and not word.startswith("=")
            self._walk_substitutions(st.text, start, stop, st.ctx, _scan_shell)
        self._mark(stop, st.ctx)
        st.pos = stop

    def _keyword(self, st: "_WalkState", word: str) -> bool:
        """Handle a compound-command keyword; False if *word* is an ordinary word."""
        stack, ctx = st.stack, st.ctx
        if st.at_cmd and word in _OPENERS:
            kind = "loop" if word in _LOOP_OPENERS else word
            stack.append((kind, ctx, word in ("for", "select", "case")))
            st.ctx = self._new_group(ctx, loop=kind == "loop", floating=st.function_next)
            st.function_next = False
            st.at_cmd = word in ("if", "while", "until")
        elif st.at_cmd and word in _CONTINUERS:
            if not stack or stack[-1][0] not in ("if", "loop"):
                raise _Unparsed
            kind, outer, _ = stack[-1]
            stack[-1] = (kind, outer, False)
            # then/do: the body runs after its condition, so it is the condition's child.
            # else/elif: a sibling of the previous body, a child of that body's condition.
            parent = ctx if word in ("then", "do") else self._parent[ctx.group]
            st.ctx = self._new_group(parent, loop=kind == "loop")
            st.at_cmd = True
        elif st.at_cmd and word in _CLOSERS:
            if not stack or stack[-1][0] != _CLOSERS[word]:
                raise _Unparsed
            st.ctx = stack.pop()[1]
            st.at_cmd = False
        elif stack and stack[-1][0] == "case" and stack[-1][2] and word == "in":
            stack[-1] = ("case", stack[-1][1], False)
            st.ctx = self._new_group(stack[-1][1])
            st.at_cmd = True           # a pattern (or `esac`) comes next
        elif st.at_cmd and word == "{":
            stack.append(("{", ctx, False))
            st.ctx = self._new_group(ctx, floating=st.function_next)
            st.function_next = False
        elif st.at_cmd and word == "}":
            if not stack or stack[-1][0] != "{":
                raise _Unparsed
            st.ctx = stack.pop()[1]
            st.at_cmd = False
        else:
            return st.at_cmd and word in ("!", "time")
        return True

    def _next_elem(self, ctx: Ctx, op: str) -> Ctx:
        nxt = Ctx(ctx.group, ctx.lst, ctx.elem + 1)
        self._ops.setdefault((ctx.group, ctx.lst), {})[nxt.elem] = op
        return nxt

    def _walk_substitutions(self, text, start, stop, ctx, scan) -> None:
        # A $(...) / backtick body runs in a subshell, before the command that contains it.
        for kind, i, j, _ in scan(text, start, stop, subst="uq"):
            if kind == "subst" and j is not None:
                inner = i + (1 if text[i] == "`" else 2)
                self._walk(text, inner, j - 1, self._new_group(ctx))
                self._mark(j, ctx)


@dataclass
class _WalkState:
    text: str
    pos: int
    end: int
    ctx: Ctx
    stack: list = field(default_factory=list)   # (kind, ctx to restore, header pending)
    at_cmd: bool = True
    function_next: bool = False
    last_name: str | None = None


class _Unparsed(Exception):
    """The walker met syntax it does not model: every binding is then treated as unproven."""


def _blank_spans(text: str, spans) -> str:
    if not spans:
        return text
    chars = list(text)
    for lo, hi in spans:
        for k in range(lo, min(hi, len(chars))):
            if chars[k] != "\n":
                chars[k] = " "
    return "".join(chars)


_PIPE_TO_SHELL_RE = re.compile(r"\|&?\s*(?:sudo\s+(?:-\S+\s+)*)?(?:\S*/)?(?:bash|sh|zsh|ksh|dash)\b")


def _pipes_to_shell(rest: str) -> bool:
    """Does the rest of a heredoc's operator line send the consumer's output to a shell? Every
    pipeline reader after a ``|`` is checked with wrappers skipped (``| sudo -u x bash``,
    ``| env -i sh``); a reader that is a shell, an expansion (``| $S``) or a wrapper whose program
    cannot be read counts, and so does a pipe the line leaves open (``cat <<EOF |`` continues after
    the body). Then resolving the consumer to a non-shell never makes the body data."""
    from tools.approval_detection import (
        _COMMAND_WRAPPER_WORDS, _deobfuscate_shell_word_for_detection, _read_shell_word, _scan_shell,
    )
    from tools.approval_detection_clobbers import is_dynamic
    if _PIPE_TO_SHELL_RE.search(rest):
        return True
    if rest.rstrip().endswith(("|", "\\")):
        return True
    for kind, i, _, quote in _scan_shell(rest, subst="uq", brace=True):
        if kind != "char" or quote is not None or rest[i] != "|" or rest.startswith("||", i):
            continue
        if i > 0 and rest[i - 1] == "|":
            continue
        pos = i + 1 + rest.startswith("&", i + 1)
        wrapped = False     # after a wrapper, any word may be the program (`sudo -u root bash`)
        while True:
            start, end, raw = _read_shell_word(rest, pos)
            if start == end:
                break
            if is_dynamic(raw):
                return True
            word = _deobfuscate_shell_word_for_detection(raw)
            base = os.path.basename(word).lower()
            if base in _HEREDOC_SHELL_CONSUMERS:
                return True
            wrapped = wrapped or base in _COMMAND_WRAPPER_WORDS
            if wrapped or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", word, re.DOTALL):
                pos = end
                continue
            break
    return False


def _heredoc_bodies(command: str) -> list[HeredocBody]:
    """Every heredoc body and whether a shell executes it (``bash <<EOF``, ``ssh h <<EOF``,
    ``cat <<EOF | sh``). Any doubt about which body belongs to which operator means executed."""
    if "<<" not in command:
        return []
    from tools.approval_detection import (
        _deobfuscate_shell_word_for_detection, _iter_shell_command_word_spans, _quoted_heredoc_body_spans,
        _scan_shell, _shell_command_segment,
    )
    spans = _quoted_heredoc_body_spans(command, quoted_only=False)
    if not spans:
        return []
    blanked = _blank_spans(command, spans)
    operators = [i for kind, i, _, quote in _scan_shell(blanked, subst="uq", comments=True)
                 if kind == "char" and quote is None and blanked.startswith("<<", i)
                 and not blanked.startswith("<<<", i) and (i == 0 or blanked[i - 1] != "<")]
    words = sorted(_iter_shell_command_word_spans(blanked))
    from tools.approval_detection_clobbers import is_dynamic
    consumers = []
    for op in operators:
        owner, owner_start = None, -1
        for start, _, word in words:
            if start < op and start + len(_shell_command_segment(blanked, start)) >= op:
                owner, owner_start = word, start
        dynamic = bool(owner) and is_dynamic(owner)
        name = os.path.basename(_deobfuscate_shell_word_for_detection(owner)).lower() if owner else ""
        newline = blanked.find("\n", op)
        rest = blanked[op:len(blanked) if newline < 0 else newline]
        piped = _pipes_to_shell(rest)
        executed = dynamic or name in _HEREDOC_SHELL_CONSUMERS or piped
        consumers.append((executed, owner or "", owner_start, dynamic, piped))
    if len(consumers) != len(spans):
        consumers = [(True, "", -1, False, True)] * len(spans)
    return [HeredocBody(lo, hi, *consumer) for (lo, hi), consumer in zip(spans, consumers)]
