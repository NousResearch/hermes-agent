"""Docstring prose in a plugin's Python files is documentation, not runtime behaviour (#155823).

``plugin_guard`` already caps findings that sit in a whole-line ``#`` comment or a CHANGELOG entry
(``_is_defensive_documentation`` → ``_comment_severity``) so prose cannot make a plugin
un-overridably ``dangerous``. A Python docstring is prose too, but it is a *string literal*, so it
never matched a comment prefix: a plugin documenting its own defenses with realistic examples
(``load("../etc/passwd") -> None``) was blocked from the catalog with no override.

These tests pin the cap for genuinely non-executable docstring text, and — just as important — pin
that executable code and non-docstring string constants keep their full severity.
"""

from pathlib import Path

from tools.plugin_guard import scan_plugin, should_allow_plugin_install

PASSPHRASE = "/etc/passwd"


def _mk_plugin(tmp_path: Path, files: dict[str, str]) -> Path:
    plugin = tmp_path / "test-plugin"
    plugin.mkdir()
    for rel, content in files.items():
        p = plugin / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
    return plugin


def _findings(plugin: Path, rel: str = "core/store.py"):
    result = scan_plugin(plugin, source="owner/repo")
    return result, [f for f in result.findings if f.file == rel]


# A defensive module: every entry point validates the id, and the docstrings say so with a
# realistic traversal example -- the shape that blocked hermes-genui (catalog pin 43de4cdb).
DEFENSIVE_MODULE = '''"""Vault store: every id is validated before it reaches the filesystem."""

import re

ID_PATTERN = re.compile(r"^[a-z0-9-]{1,64}$")


def size(block_id):
    """Return the stored byte size, or None.

    size("../etc/passwd") -> None (never a path outside the store)
    """
    if not ID_PATTERN.fullmatch(block_id):
        return None
    return 0
'''


class TestDocstringProseIsDemoted:
    def test_function_docstring_example_does_not_block_install(self, tmp_path):
        plugin = _mk_plugin(tmp_path, {"core/store.py": DEFENSIVE_MODULE})
        result, findings = _findings(plugin)
        assert findings, "the example is still reported; it is demoted, not hidden"
        assert all(f.severity != "critical" for f in findings)
        assert result.verdict != "dangerous"
        allowed, _reason = should_allow_plugin_install(result)
        assert allowed is not False

    def test_module_and_class_docstrings_are_covered(self, tmp_path):
        plugin = _mk_plugin(tmp_path, {"core/ids.py": f'''"""Ids are validated; load("{PASSPHRASE}") -> None."""


class Store:
    """Never opens "{PASSPHRASE}" (see load())."""

    def load(self, block_id):
        """Reject anything that is not an id: load("{PASSPHRASE}") -> None."""
        return None
'''})
        result, findings = _findings(plugin, "core/ids.py")
        assert len(findings) == 3
        assert all(f.severity == "medium" for f in findings), [f.severity for f in findings]
        assert result.verdict == "safe"


class TestExecutableContentKeepsFullSeverity:
    def test_runtime_code_line_stays_critical(self, tmp_path):
        plugin = _mk_plugin(tmp_path, {"core/store.py": f'''"""Vault store."""

import pathlib


def read(block_id):
    return pathlib.Path("{PASSPHRASE}").read_text()
'''})
        result, findings = _findings(plugin)
        assert [f.severity for f in findings] == ["critical"]
        assert result.verdict == "dangerous"

    def test_triple_quoted_constant_is_not_a_docstring(self, tmp_path):
        # A payload held in a string constant is *not* documentation: `ast` only reports a
        # docstring for the first statement of a module/class/function, so this keeps severity.
        plugin = _mk_plugin(tmp_path, {"core/store.py": f'''PAYLOAD = """
cat {PASSPHRASE}
"""


def run():
    return PAYLOAD
'''})
        result, findings = _findings(plugin)
        assert [f.severity for f in findings] == ["critical"]
        assert result.verdict == "dangerous"

    def test_string_after_a_statement_is_not_a_docstring(self, tmp_path):
        plugin = _mk_plugin(tmp_path, {"core/store.py": f'''def run():
    print("reading")
    """open("{PASSPHRASE}")"""
    return None
'''})
        _result, findings = _findings(plugin)
        assert [f.severity for f in findings] == ["critical"]
