"""Git global-option grammar for the destructive-git approval rules (#150744, #155469).

The rules in :mod:`tools.approval_detection` match the SUBCOMMAND past git's global options
(`git -C <dir> reset --hard`); `_join_escaped_word_chars` feeds its escaped-separator variant.
"""

from bisect import bisect_left, bisect_right
import hermes_re as re

# `git` plus the global options that may sit before the subcommand (`git -C <dir> reset --hard`).
# Value-taking options consume their separate value word; any other option is a single dash token.
# A word may contain quoted runs (`-C "/a b"`, `--git-dir='/a b/.git'`) or a command substitution
# (`-C $(git rev-parse --show-toplevel)`), which the shell passes as one argument. Each option is an
# atomic group with the value-taking form tried first: the two alternatives overlap (`-c x` is also
# two dash-or-word tokens), so a non-atomic run of option-like tokens (`git -c -c -c ... x`) is
# re-split exponentially on a failed match; words are possessive for the same reason.
# An option missing from that list still takes a separate value when another option follows it
# (`--opt val --flag`), unless that value is a gated verb or the option is a known value-less one.
# Every word ending in a `git` word (`-c git`, `alias.x=git`) starts its own search over the same rest
# of the command, so no option word may end in one: the run stops there instead of rescanning what
# that search covers, which keeps an uncapped run linear (`git -c git -c ... x`).
# A `$(...)` may nest (`-C $(realpath $(pwd))`), up to _GIT_SUBST_DEPTH levels; each level is an atomic
# run of non-paren characters or one deeper group, so an unclosed `$(` fails in a single pass.
_GIT_SUBST_DEPTH = 4
_GIT_PARENS = r'\([^()\n]*\)'
for _ in range(_GIT_SUBST_DEPTH - 1):
    _GIT_PARENS = r'\((?>[^()\n]|' + _GIT_PARENS + r')*+\)'
_GIT_WORD = r'''(?:"[^"]*"|'[^']*'|\$''' + _GIT_PARENS + r'''|`[^`\n]*`|[^\s;|&"'])++(?<!\bgit)'''
_GIT_NO_VALUE = (r'(?:--no-pager|--paginate|--bare|--no-replace-objects|--no-lazy-fetch|--no-optional-locks'
                 r'|--no-advice|--(?:literal|glob|noglob|icase)-pathspecs|--(?:html|man|info)-path|--version|--help|-[pPvh])')
_GIT = (r'\bgit(?>\s+(?:(?:-c|--git-dir|--work-tree|--namespace|--exec-path|--super-prefix|--config-env'
        rf'|--attr-source)\s+{_GIT_WORD}|(?!{_GIT_NO_VALUE}\s)(?:--[^\s=;|&"\'`$]++|-[^\s;|&"\'`$-])\s+'
        rf'(?!-|(?:push|reset|clean|branch)\s){_GIT_WORD}(?=\s+-)|-(?:{_GIT_WORD}|(?=\s))))*\s+')
_GIT_PUSH = rf'{_GIT}push\b'
# The push's own arguments, bounded to ONE command segment: a later command's flag is never the
# push's (`git push -q origin main && grep -f patterns.txt log` is not a force push). A quoted
# string is one atom, so a quoted `;`/`&`/`|` (`-o 'ci;skip'`) does not end the segment, and a
# flag inside quotes is not seen; a redirection's `&`/`|` (`2>&1`, `&>`, `>|`) stays inside too.
# Every alternative starts on a different character (or lookaround), so a failed match scans a
# run once instead of re-splitting it exponentially (`>&>&>&...`, `'a' 'a' 'a' ...`). The scan
# stops at a nested push, whose own search covers the rest, so `git push git push ...` is linear.
_GIT_PUSH_ARGS = (r'''(?:(?>'[^']*'|"(?:[^"\\]|\\[\s\S])*")|'(?![^']*')|"(?![^"]*")'''
                  r'''|\\[\s\S]|(?!''' + _GIT_PUSH + r''')[^;&|\n'"\\]|(?<=[<>])&|(?<![<>])&(?=>)|(?<=>)\|)*''')


def _join_escaped_word_chars(command: str) -> str:
    """Turn each escaped `;`/`&`/`|`, space, tab or quote into `_`, so it stays literal text inside its
    word after normalization strips the backslash. Inside single quotes a backslash is literal, so only
    the separators are joined there, as before."""
    def join(span: re.Match) -> str:
        chars = ';&|' if span.group(0)[0] == "'" else ';&| \t"\'`'
        return re.sub(r'\\(.)', lambda m: '_' if m.group(1) in chars else m.group(0), span.group(0))
    return re.sub(r''''[^']*'|"(?:[^"\\]|\\[\s\S])*"|\\[\s\S]''', join, command)


# RE2 has no atomic group, possessive quantifier or mid-pattern lookaround, so it cannot compile `_GIT`.
# GitCommandMatcher walks the option run `_GIT` describes instead: one option per step, the three forms
# tried in `_GIT`'s order (value-taking option, unlisted option plus value, single dash token), the
# first that fits kept and each word read whole. A word is one anchored match of the plain spelling
# below: RE2 does not backtrack, and from a fixed start it ends where the possessive spelling ends.
_GIT_RE = re.compile(r'\bgit', re.IGNORECASE)
_GIT_PLAIN_PARENS = r'\([^()\n]*\)'
for _ in range(_GIT_SUBST_DEPTH - 1):
    _GIT_PLAIN_PARENS = r'\((?:[^()\n]|' + _GIT_PLAIN_PARENS + r')*\)'
_GIT_WORD_RE = re.compile(r'''(?:"[^"]*"|'[^']*'|\$''' + _GIT_PLAIN_PARENS + r'''|`[^`\n]*`|[^\s;|&"'])+''')
_GIT_VALUE_OPTION_RE = re.compile(r'\s+(?:-c|--git-dir|--work-tree|--namespace|--exec-path|--super-prefix'
                                  r'|--config-env|--attr-source)\s+', re.IGNORECASE)
_GIT_NO_VALUE_RE = re.compile(rf'\s+{_GIT_NO_VALUE}\s', re.IGNORECASE)
_GIT_OTHER_OPTION_RE = re.compile(r'''\s+(?:--[^\s=;|&"'`$]+|-[^\s;|&"'`$-])\s+''')
_GIT_GATED_VALUE_RE = re.compile(r'-|(?:push|reset|clean|branch)\s', re.IGNORECASE)
_GIT_DASH_RE = re.compile(r'\s+-')


def _git_word_end(text: str, pos: int) -> int:
    """End of the shell word at *pos*; -1 when there is none or it ends in a `git` word."""
    word = _GIT_WORD_RE.match(text, pos)
    return -1 if word is None or _GIT_RE.match(text, word.end() - 3) else word.end()


def _git_option_end(text: str, pos: int) -> int:
    """End of the ONE global option after the blanks at *pos*; -1 when none starts there."""
    if (option := _GIT_VALUE_OPTION_RE.match(text, pos)) and (end := _git_word_end(text, option.end())) >= 0:
        return end
    if not _GIT_NO_VALUE_RE.match(text, pos) and (option := _GIT_OTHER_OPTION_RE.match(text, pos)) \
            and not _GIT_GATED_VALUE_RE.match(text, option.end()):
        end = _git_word_end(text, option.end())
        if end >= 0 and _GIT_DASH_RE.match(text, end):
            return end
    if dash := _GIT_DASH_RE.match(text, pos):
        if (end := _git_word_end(text, dash.end())) >= 0:
            return end
        if text[dash.end():dash.end() + 1].isspace():
            return dash.end()
    return -1


def _git_option_ends(text: str, pos: int):
    """For *pos* right after a `git` word: *pos*, then the end of each global option in turn. The
    regex could stop its option run after any of them, so a rule tries its subcommand at each."""
    while pos >= 0:
        yield pos
        pos = _git_option_end(text, pos)


class GitCommandMatcher:
    """Stand-in for a `{_GIT}<tail>` rule; *tail* (the subcommand and what follows it) stays a regex.
    Every `git` word is tried, as the unanchored regex did, once the tail is known to occur at all.
    The options after a position do not depend on the `git` word the walk began at, so a walk stops
    where an earlier one passed: a `git` inside an option's value (`-C $(git -C x)`) rescans nothing.
    The match returned is the tail's, blanks included: callers only test it for truth."""

    def __init__(self, tail: str, flags: int):
        self._tail = re.compile(r'\s+(?:' + tail + ')', flags)

    def search(self, text: str):
        if self._tail.search(text) is None:
            return None
        tried: set[int] = set()
        for git in _GIT_RE.finditer(text):
            for pos in _git_option_ends(text, git.end()):
                if pos in tried:
                    break
                tried.add(pos)
                if match := self._tail.match(text, pos):
                    return match
        return None


# The force-push rules go on past `push` with `_GIT_PUSH_ARGS`, whose lookarounds RE2 lacks too.
# GitPushMatcher reads the push's arguments the way that fragment does: a closed quoted string or a
# backslash pair is one step, any other character is one step, and the scan ends at a `;`, a newline,
# an `&`/`|` that is not a redirection's, a `"` whose later quotes are all escaped, or a nested push.
_GIT_PUSH_RE = re.compile(r'\s+push\b', re.IGNORECASE)
_GIT_PUSH_STOP_RE = re.compile(r'''[;&|\n'"\\]|\bgit''', re.IGNORECASE)
_GIT_PUSH_DQUOTE_RE = re.compile(r'"(?:[^"\\]|\\[\s\S])*"')
# A push option's value (`-o +foo`) is not a refspec. The `+refspec` rule opens with this text, which
# GitPushMatcher takes off and checks itself: RE2 has no lookbehind wider than one character.
_GIT_PUSH_NOT_OPTION_VALUE = r'(?<!\s-o)(?<!\s--push-option)'
_GIT_PUSH_OPTION_RE = re.compile(r'\s(?:-o|--push-option)', re.IGNORECASE)


def _git_push_end(text: str, pos: int, known: dict[int, int]) -> int:
    """End of the `push` word that the option run after a `git` word ending at *pos* leads to, else -1.
    Runs that meet go on alike, so *known* keeps the answer of every position already walked."""
    walked = []
    end = -1
    for at in _git_option_ends(text, pos):
        if at in known:
            end = known[at]
            break
        walked.append(at)
        if push := _GIT_PUSH_RE.match(text, at):
            end = push.end()
            break
    known.update(dict.fromkeys(walked, end))
    return end


def _git_push_step_end(text: str, stop: re.Match, known: dict[int, int]) -> int:
    """End of the one scan step that starts at *stop* (a _GIT_PUSH_STOP_RE match past the `push`
    word); -1 when the push's arguments end there."""
    at = stop.start()
    char = text[at]
    if char == "'":  # a closed string is one step; a quote that never closes is literal
        return text.find("'", at + 1) + 1 or at + 1
    if char == '"':
        if quoted := _GIT_PUSH_DQUOTE_RE.match(text, at):
            return quoted.end()
        return at + 1 if text.find('"', at + 1) < 0 else -1
    if char == '\\':
        return at + 2 if at + 1 < len(text) else -1
    if char == '&':  # a redirection's only: `2>&1`, `<&0`, `&>file`
        return at + 1 if text[at - 1] in '<>' or text.startswith('>', at + 1) else -1
    if char == '|':  # `>|` only
        return at + 1 if text[at - 1] == '>' else -1
    if char in ';\n':
        return -1
    # A `git` word: a nested push ends the scan (its own search covers the rest), any other is text.
    return -1 if _git_push_end(text, stop.end(), known) >= 0 else at + 1


def _git_push_args(text: str, pos: int, known: dict[int, int], stood: set[int]) -> tuple[int, list[tuple[int, int]]]:
    """Where the argument scan from *pos* (the end of a `push` word) ends, and the spans it takes in one
    step (quoted strings, backslash pairs). An argument may start anywhere from *pos* to that end,
    except inside one of the spans. Two scans that stand on the same character go on alike, so a scan
    ends where an earlier push's stood (*stood*): past it there is nothing that scan did not offer.
    That keeps pushes hidden from each other (`git push \\git push \\git push ...`) linear."""
    spans = []
    while stop := _GIT_PUSH_STOP_RE.search(text, pos):
        at = stop.start()
        if at in stood:
            return at, spans
        stood.add(at)
        pos = _git_push_step_end(text, stop, known)
        if pos < 0:
            return at, spans
        if pos > at + 1:
            spans.append((at, pos))
    return len(text), spans


class GitPushMatcher:
    """Stand-in for a `{_GIT_PUSH}{_GIT_PUSH_ARGS}<tail>` rule; *tail* (the flag or `+refspec` and the
    blank before it) stays a regex. Every place the tail matches is found once, then every `git` word
    that leads to a `push` is tried, as the unanchored regex did: the rule fires when one of those
    places is where an argument of that push may start. The match returned is the tail's: callers only
    test it for truth."""

    def __init__(self, tail: str, flags: int):
        self._skips_option_value = tail.startswith(_GIT_PUSH_NOT_OPTION_VALUE)
        self._tail = re.compile(tail.removeprefix(_GIT_PUSH_NOT_OPTION_VALUE), flags)

    def _tail_matches(self, text: str) -> list[re.Match]:
        found = []
        pos = 0
        while match := self._tail.search(text, pos):
            pos = match.start() + 1
            if not self._skips_option_value or not any(
                    _GIT_PUSH_OPTION_RE.fullmatch(text, pos - width, pos - 1) for width in (4, 15)):
                found.append(match)
        return found

    def search(self, text: str):
        found = self._tail_matches(text)
        if not found:
            return None
        starts = [match.start() for match in found]
        known: dict[int, int] = {}
        scanned: set[int] = set()
        stood: set[int] = set()
        for git in _GIT_RE.finditer(text):
            pos = _git_push_end(text, git.end(), known)
            if pos < 0 or pos in scanned:
                continue
            scanned.add(pos)
            stop, spans = _git_push_args(text, pos, known, stood)
            for index in range(bisect_left(starts, pos), bisect_right(starts, stop)):
                span = bisect_left(spans, (starts[index],)) - 1
                if span < 0 or starts[index] >= spans[span][1]:
                    return found[index]
        return None
