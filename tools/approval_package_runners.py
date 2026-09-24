"""Package-runner projection for ``approvals.deny`` matching.

``npx``/``bunx``/``pnpm dlx``/``yarn dlx``/``uvx``/``pipx run`` fetch a package and run it,
so the thing a deny rule names is the PACKAGE, not the runner binary. A user who wrote
``npx sketchy-tool *`` meant every invocation of that package — but the runner accepts a
version pin on the operand (``sketchy-tool@1.2.3``, ``evil-pkg==1.0``) and boolean flags in
front of it (``npx -y``), and the deny floor matched the literal text, so the pinned or flagged
spelling ran under ``--yolo`` while the bare one was blocked. This module folds those
spellings back to the canonical one; :func:`tools.approval_detection._deny_command_variants`
yields the result as one more projection, so existing rules keep matching what they matched.

Only the FIRST positional after the runner (the package operand) and the values of the
runner's own package-bearing options are folded. Later arguments are the package's argv and
may legitimately contain ``@`` or ``==`` (``--email a@b.c``); they are left byte-for-byte.
"""

import os
import re

# runner basename -> number of leading words that name the runner (``pnpm dlx`` is two).
_RUNNER_WORDS = {"npx": 1, "pnpx": 1, "bunx": 1, "uvx": 1}
_TWO_WORD_RUNNERS = {("pnpm", "dlx"), ("yarn", "dlx"), ("pipx", "run")}

# Boolean runner options that change nothing about WHICH package runs.
_RUNNER_BOOL_OPTIONS = frozenset({
    "-y", "--yes", "--no", "-q", "--quiet", "-v", "--verbose", "--no-install", "--ignore-existing",
    "--prefer-offline", "--prefer-online", "--isolated", "--no-cache", "--refresh", "--offline",
    "--no-progress", "--silent", "-s", "--",
})
# Runner options whose VALUE is a package spec (``npx -p tool@1``, ``uvx --from pkg==1``).
_RUNNER_PACKAGE_OPTIONS = frozenset({"-p", "--package", "--from", "--with", "--spec"})

# ``name@spec`` / ``@scope/name@spec`` (npm) and ``name[extras]==spec`` (PEP 508 operators).
_NPM_PIN_RE = re.compile(r"^(@[^/@\s]+/)?([^@\s/]+)@\S+$")
_PEP508_PIN_RE = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)(\[[^\]]*\])?\s*(===|==|~=|!=|>=|<=|<|>)\S*$")


def strip_package_pin(spec: str) -> str:
    """``tool@1.2.3`` -> ``tool``, ``@scope/tool@latest`` -> ``@scope/tool``, ``pkg[x]==1`` -> ``pkg``."""
    m = _NPM_PIN_RE.match(spec)
    if m:
        return (m.group(1) or "") + m.group(2)
    m = _PEP508_PIN_RE.match(spec)
    if m:
        return m.group(1)
    return spec


def canonical_package_runner_argv(tokens: list[str]) -> list[str] | None:
    """Return the canonical argv for a package-runner invocation, or None if ``tokens`` is not one.

    Canonical = runner word(s), package-bearing options with their pins stripped, the package
    operand with its pin stripped, then the package's own argv untouched. Boolean runner options
    are dropped. Returns None when nothing changed, so callers add no duplicate variant.
    """
    if not tokens:
        return None
    head = os.path.basename(tokens[0]).lower()
    runner_len = _RUNNER_WORDS.get(head)
    if runner_len is None:
        if len(tokens) < 2 or (head, tokens[1].lower()) not in _TWO_WORD_RUNNERS:
            return None
        runner_len = 2
    out = [head, *tokens[1:runner_len]]
    index, changed = runner_len, False
    while index < len(tokens):
        token = tokens[index]
        if not token.startswith("-"):
            break
        option, equals, value = token.partition("=")
        if option in _RUNNER_BOOL_OPTIONS and not equals:
            changed = True
        elif option in _RUNNER_PACKAGE_OPTIONS:
            if equals:
                folded = strip_package_pin(value)
                changed |= folded != value
                out.append(f"{option}={folded}")
            else:
                out.append(option)
                if index + 1 < len(tokens):
                    index += 1
                    folded = strip_package_pin(tokens[index])
                    changed |= folded != tokens[index]
                    out.append(folded)
        else:
            out.append(token)
        index += 1
    if index < len(tokens):
        folded = strip_package_pin(tokens[index])
        changed |= folded != tokens[index]
        out.append(folded)
        out.extend(tokens[index + 1:])
    return out if changed else None
