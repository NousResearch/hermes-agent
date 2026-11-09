"""Narrow literal refinements shared by the Skills Guard pattern scan."""

import re


# A complete assignment to a plain variable cannot also be an environment write
# target continued on another line. Keep all other forms at their original severity.
_CONFIG_PATH_READ = re.compile(
    r"""\s*[A-Za-z_]\w*\s*=\s*Path\s*\(\s*os\.environ\s*\[
        \s*(?P<quote>["'])(?P<name>[A-Za-z_]\w*)(?P=quote)\s*\]\s*\)\s*""",
    re.VERBOSE,
)
_SECRET_NAME = re.compile(r"KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL", re.IGNORECASE)
# Complete, deliberately low-entropy instructional values only. Provider token-shape
# rules still run independently; adding a real-looking suffix does not qualify.
_PLACEHOLDER = re.compile(
    r"(?:ghp_)?your_(?:new_)?(?:github_)?(?:token|api_key|secret|password)"
    r"|unused-for-(?:script-only|tests?|testing|examples?)"
)


def literal_context(
    pid: str,
    line: str,
    severity: str,
    description: str,
    secret_matches: tuple[str, ...] = (),
) -> tuple[str, str]:
    """Retain notes for explicit placeholders and standalone non-secret path reads.

    General and secret-bearing environment reads retain the original warning;
    ambiguous statement boundaries never authorize a demotion.
    """
    if pid == "python_os_environ":
        match = _CONFIG_PATH_READ.fullmatch(line)
        if match and not _SECRET_NAME.search(match["name"]):
            return (
                "low",
                description + " (standalone non-secret path lookup; informational)",
            )
    if pid == "hardcoded_secret":
        # The scanner regex owns credential boundaries, including unterminated
        # literals; reparsing with a closing-quote requirement can hide a second key.
        values = [re.split(r"[\"']", span, maxsplit=1)[1] for span in secret_matches]
        if values and all(_PLACEHOLDER.fullmatch(value) for value in values):
            return (
                "medium",
                description + " (explicit placeholder literal; informational)",
            )
    return severity, description
