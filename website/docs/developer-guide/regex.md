---
sidebar_position: 31
title: "Regular Expressions"
description: "How Hermes code writes regexes: hermes_re, the stdlib re API on Google RE2 — what it refuses, what differs, caching, threads and fork"
---

# Regular Expressions

Product code matches regular expressions with **`hermes_re`**, a first-party package that offers the
stdlib `re` API on top of Google RE2. One RE2 search takes time linear in the input, so no pattern, bundled
or user-supplied, can hang a turn through backtracking (a scan for many matches with a lookahead edge guard
is the one exception, below).

```python
import hermes_re as re

_KEY = re.compile(r"^(?P<key>\w+)\s*=\s*(?P<value>.*)$", re.MULTILINE)

def keys(text: str) -> list[str]:
    return [m["key"] for m in _KEY.finditer(text)]
```

Everything you know from `re` is there: `compile`, `search`, `match`, `fullmatch`, `findall`, `finditer`,
`sub`, `subn`, `split`, `escape`, `purge`, `error` / `PatternError`, the flags `IGNORECASE`, `MULTILINE`,
`DOTALL`, `ASCII`, `UNICODE` (and their one-letter names), `Pattern` and `Match` with stdlib's methods and
attributes, `str` and `bytes` patterns, `pos` / `endpos`, named groups, replacement templates.

There is **no fallback to stdlib `re`**. A pattern RE2 cannot run is refused; it is never matched by
another engine.

## What is refused, and how to respell it

`re.compile()` raises `hermes_re.error` (a subclass of stdlib `re.error`, so existing
`except re.error:` handlers catch it) for constructs RE2 has no linear-time answer for.

| Construct | Example | Respell as |
|---|---|---|
| Lookahead / lookbehind that is not an *edge guard* | `a(?=b)c`, `(?<=ab)c` | Match the context and capture what you need: `a(b)c`; or check the neighbour in Python after the match. |
| Backreference | `(["'])x\1`, `(?P=name)` | One alternative per case: `"x"\|'x'`; or capture both and compare in Python. |
| Possessive quantifier, atomic group | `a++`, `(?>a+)` | No drop-in replacement: dropping the `+` or the `(?>` can change which strings match (`fullmatch(r"a++a", "aa")` and `fullmatch(r"(?>a+)a", "aa")` are `None`, `fullmatch(r"a+a", "aa")` matches). The plain form is equivalent only where nothing after the repeat can start with a character it consumes (`[^"]*+"` → `[^"]*"`); otherwise rewrite for the intended match. Either way, check the rewrite against the original on the inputs that matter. |
| Conditional | `(a)?(?(1)b\|c)` | Two alternatives: `ab\|c`. |
| Two capture groups in one repeat that can both match empty, one at the start and one at the end of an iteration | `(?:(?P<y>)b\|a(?P<x>))+` | Give one of them text to match, or move it out of the repeat: which one closed last (`lastindex`, `lastgroup`) cannot be told from RE2's spans. |
| `VERBOSE`, `LOCALE`, `DEBUG`; inline `(?x)`, `(?a)`, `(?u)`, `(?L)` | `re.compile(p, re.X)` | Build the pattern from commented string pieces; pass `re.ASCII` as a flag. These raise `ValueError`. |
| Named character | `\N{EM DASH}` | `\u2014`. |
| More than 495 nested groups | `"(" * 496 + "a" + ")" * 496` | Flatten the nesting. Stdlib's parser recurses once per group and raises `RecursionError` past this depth (from a fresh interpreter at the default recursion limit); the facade refuses at `compile()` instead and handles every depth up to it without recursing. |
| A non-multiline `$` with something to match after it | `x$y`, `(x$)+` | `\Z`, or `(?m)` if "end of line" is meant. |
| Repeat count over 1000 (also as a product of nested counts) | `a{1001}`, `(?:a{501}){2}` | A smaller bound, or `*` / `+` and a length check in Python. |
| A lone surrogate outside a bracket class | `"\ud800"` | See *Lone surrogates* below. |

Syntax the running stdlib refuses raises `hermes_re.error` as well, even where RE2 could run it:
`\pL`, `(?<name>a)`, `(?U)`, a global inline flag after the start (`a(?i)b`), and `\z` before Python 3.14.

**Edge guards are supported.** One guard per edge of a top-level alternative: a one-character lookbehind
at its start (`(?<!\w)`, `(?<=[.!])`), a lookahead at its end (`(?!\w)`, `(?=\s|$)`), and a tail `$`.
That covers the usual "not part of a longer word" patterns unchanged.

A lookahead guard is evaluated once per match, as in stdlib. With an unbounded body (`a(?=.*z)`) every
match of `findall` / `finditer` / `sub` re-reads the rest of the line, so a scan for many matches is
quadratic in both engines, and slower here: `findall(r"a(?=.*z)", "a" * 16000 + "z")` takes 0.03 s in
stdlib and 0.75 s through the facade. Keep a lookahead's body short (`(?=\s|$)`), or test the condition once
in Python.

### Bounded repeats of Unicode classes

`\w`, `\d` and `\s` have stdlib's Unicode meaning on `str`. The Unicode `\w` class is large, and RE2 copies
a repeated class once per count: `\w{0,512}` compiles to about a hundred thousand instructions, and
`(?:\w{1,64}\s{1,64}){0,8}` to millions (hundreds of milliseconds to build, tens of megabytes held).
Each RE2 program has a 64 MiB budget; one that exceeds it fails **on first use** with
`error: pattern too large`. Where the text is ASCII by construction, write the ASCII class
(`[A-Za-z0-9_]{0,64}`) or pass `re.ASCII`; otherwise prefer `+` / `*` with a length check in Python.

### `\b` next to non-ASCII text

RE2's own `\b` knows ASCII word characters only. The facade makes a `\b` exact (Unicode, as stdlib)
wherever one side of it is known from the pattern: at the edges of the pattern (`\bfoo\b`,
`\b` + `re.escape(name)` + `\b`), and next to a literal on both sides. A `\b` in the **interior** next to a
class that can be a word or a non-word character (`\b[\w./-]+`), a `\b` inside a repeated group or a
lookaround, and every `\B` keep the ASCII rule: on ASCII text they give stdlib's result, next to a
non-ASCII letter or digit they can differ in either direction.

`re.inexact_boundaries(pattern, flags=0)` returns the offsets of such boundaries (an empty list means all
are exact). To make one exact, say which side you mean: `(?<!\w)` before a word, `(?!\w)` after it.

### Lone surrogates

RE2 scans UTF-8, which cannot carry a lone surrogate (`"\ud83d"` without its pair). In a subject it is
scanned as U+FFFD; in a bracket class it never matches. A sanitiser like
`re.sub(r"[\ud800-\udfff]", "", text)` therefore removes nothing: use
`text.encode("utf-16", "surrogatepass")`-style handling or `str.translate` instead of a regex.

### Matcher objects

`isinstance(p, re.Pattern)` is true for facade patterns only when `re` is `hermes_re`; a compiled facade
pattern is not a stdlib `re.Pattern`, and a stdlib function does not accept one. Do not hand a facade
pattern to third-party code that type-checks for stdlib's class (such call sites keep stdlib `re` under a
named lint exemption). The other direction works: `hermes_re.compile(stdlib_pattern)` converts from its
`.pattern` and `.flags`. There is no `Pattern.scanner` and no `re.Scanner`.

## Silent differences

These compile and can give a result that differs from stdlib:

| # | Difference | Example (stdlib → facade) |
|---|---|---|
| B1 | A `\b` listed by `inexact_boundaries()` and every `\B` use ASCII word characters | `search(r"x\b.", "xé")`: `None` → match |
| B2 | After an empty match, stdlib retries a non-empty match at the same position; the facade moves on | `findall("\|a", "a")`: `['', 'a', '']` → `['', '']` |
| B3 | Lone surrogates (above) | `findall(r"[\ud800-\udfff]", "a\ud83db")`: `['\ud83d']` → `[]` |
| B4 | A repeated body that can match empty: group contents can differ | `fullmatch("((b??)+)?", "bbb").groups()` |
| B5 | `pos > endpos` | `compile("").match("abc", 3, 0)`: match → `None` |
| B6 | (none: syntax stdlib refuses, such as `\pL`, `(?<name>a)` or `(?U)`, is refused at `compile()`) | |
| B7 | IGNORECASE classes with members above U+FFFF follow Python 3.14's rule on every interpreter | |
| B8 | An interpreter whose Unicode version, or an RE2 build whose Unicode generation, has no entry in `hermes_re/_tables.py` gets RE2's own `\w` / `\d` where they differ | caught by `tests/hermes_re/test_tables_current.py` |

Also: `error.pos` / `lineno` / `colno` are `None`.

Avoid patterns that can match the empty string in `sub` / `split` / `findall` (B2); write `\s+`, not `\s*`.

## `build()`, `build_all()`, `inexact_boundaries()`

`re.compile()` checks and translates the whole pattern at once, without loading the engine, so syntax
errors and refused constructs surface at import. The RE2 program itself is built on **first use**
(programs cannot be persisted and a large one takes hundreds of milliseconds). Two things can therefore
still fail at first use: `error: pattern too large` (the 64 MiB budget) and `ModuleNotFoundError` when
google-re2 is not installed. That error always has `name == "re2"` (it is an `ImportError` subclass, and there
is no fallback to stdlib `re`): the update step of releases that shipped before google-re2 was a dependency
imports the new checkout's core modules without it, and treats exactly that shape as "dependencies not
installed yet" rather than as a broken checkout. For the same reason no product module may run a match while
it is being imported: compile at module level, match inside functions.

- `re.build(pattern, flags=0)` is `compile()` with the program built now. It raises what first use would
  raise, including for the exact form of an edge guard that matching might need later.
- `re.build_all(wide=False)` builds every pattern alive in the process and returns the failures as
  `(Pattern, error)` pairs. The test for a module with patterns is:

  ```python
  import hermes_re as re
  import my_module  # noqa: F401

  def test_every_pattern_builds():
      assert re.build_all() == []
  ```

  `wide=True` also builds the Unicode `\b` forms that matching builds on the first non-ASCII subject.
- `re.inexact_boundaries(pattern, flags=0)`: see above. Security patterns (redaction, approval, scanners)
  should assert that it returns `[]`.

A pattern compiled lazily inside `except re.error:` that skips the entry would hide a refused bundled
rule: every table of bundled patterns gets a test that all its entries build.

## User-supplied patterns

For a pattern from configuration, a plugin manifest or a remote party:

```python
try:
    compiled = re.build(text, flags)   # not compile(): the budget is checked now
except re.error as exc:
    raise ConfigError(f"unsupported regular expression {text!r}: {exc}") from exc
if re.inexact_boundaries(text, flags):
    logger.warning("a \\b in this pattern follows ASCII word rules")
```

Tell users what is refused (the table above) in the feature's documentation. Lookarounds at the edges of
the pattern, the common case in wake words and mention patterns, keep working.

## Memory budgets

| Budget | Value | Meaning |
|---|---|---|
| Per program | 64 MiB | RE2's limit for one program and its lazily built DFA. Over it: `error` on first use. |
| Compile cache, entries | 4,096 | Distinct `(type, pattern, flags)` keys kept; least recently used leaves first. |
| Compile cache, size | 256 MiB | Estimated size of the programs built for cached patterns; least recently used leaves first. |

The size bound counts every program built for a pattern (8 KiB plus 48 bytes per RE2 instruction, which
is measured resident memory after a few searches). It is an estimate: a program whose DFA keeps growing
can hold more, up to its own 64 MiB. A pattern that leaves the cache is freed unless a module still holds
it; a held pattern stays built and usable.

A pattern built at run time (`rf"\b{re.escape(name)}\b"`) costs about 0.1 ms on first use and about
20 KiB while cached. A run-time pattern with a Unicode class repeated inside (`\w{0,32}`) costs
milliseconds and megabytes each; that is the pattern's own program, and only the size bound limits it.
Use the ASCII class (`[A-Za-z0-9_]{0,32}`); keep a compiled one only for a small, fixed set of names (each holds
about 8 MiB).

## Caching and retention

- **Compile cache**: as above. `re.search(pattern_text, s)` with a cached pattern costs one lookup.
- **Subject cache**: RE2 scans bytes, so a `str` subject is encoded first. The encoded form of the text in
  use is kept so that a loop of `pattern.search(text, pos)`, or many patterns over one text, encodes once:
  - slot 1 keeps the **last subject of 1,024 to 4,194,304 characters together with its encoded copy until
    another subject of that size arrives**, or `re.purge()`. In a process that redacts secrets this is the
    text *before* redaction. Call `re.purge()` after handling text that must not stay in memory;
  - slot 2 keeps a larger subject only while it is being used repeatedly, and lets it go once other
    matching of the same volume has happened.
- **Windows**: a search confined by `pos` / `endpos` to at most 32,768 characters of a subject that is not
  cached encodes only that range. A stream buffer, which is a new `str` on every tick, can therefore be
  searched in its tail (`pattern.search(buf, len(buf) - 512)`) at a cost independent of its length. A
  search over the whole of a new non-ASCII buffer costs time proportional to the buffer on every call.
- **Replacement templates**: parsed once per `(template, pattern)`.

`re.purge()` clears all of them.

## Threads and fork

Patterns and the caches are safe to share between threads; a pattern used by several threads before it
is built may be built twice, never used half-built.

**Fork.** RE2 guards each program's DFA cache with a mutex, and a mutex held by another thread at the
moment of `fork()` stays locked in the child. `hermes_re` therefore drops every inherited program in a
forked child and builds its own on first use, so matching in a forked child does not hang on a lock the
parent's threads held. The child pays the build again for each pattern it uses. What the facade cannot
reset is state inside the RE2 / Abseil library itself: **a child forked from a multi-threaded process
should `exec`, or use the `spawn` start method, rather than keep matching and start threads of its own.**

## Where stdlib `re` remains

Lint bans `import re` (code-health rule HX018, later ruff `TID251`) outside a named exemption list:

- code that runs before third-party packages can exist (process bootstrap, the package manager, update
  takeover and recovery) and files that still-supported old updater processes call after a checkout swap;
- `scripts/`, skill helper scripts, and program text embedded in string literals: they run in another
  interpreter;
- `tests/`: stdlib `re` is also the reference the facade is checked against (`tests/hermes_re/`);
- a call site whose third-party API type-checks for stdlib's `re.Pattern` (per-line waiver with the reason).

`hermes_re` itself imports stdlib `re` in one place, for the base class of `error`.

The Unicode tables in `hermes_re/_tables.py` are generated: after a Python or google-re2 upgrade run
`python scripts/gen_hermes_re_tables.py` and commit the result (`--check` verifies it, and so does
`tests/hermes_re/test_tables_current.py`). RE2's Unicode data depends on the BUILD, not only the
google-re2 version: the wheels use RE2's bundled tables (Unicode 15.1), a libre2 built with
`RE2_USE_ICU` (Alpine, hence the musl lane; Nix) takes `\p{..}` from ICU. Entries are keyed by Python's
Unicode version and the build's generation (`hermes_re._engine.unicode_generation()`), so run the
generator once per build kind (e.g. a wheel venv and the musl container).
