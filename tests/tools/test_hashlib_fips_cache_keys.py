"""FIPS guard for hashlib cache-key digests in the skills-hub modules.

On FIPS-enforced interpreters (RHEL 8/9 in FIPS mode) ``hashlib.md5`` and
``hashlib.sha1`` raise ``ValueError: EVP_DigestInit_ex disabled for FIPS``
unless the call passes ``usedforsecurity=False``.  Every digest in these
modules is a cache key or a display-name slug, never a security primitive,
so the flag is always safe to set and always required.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

# Modules whose digests are pure cache keys / name slugs.
TARGET_MODULES = (
    "tools/skills_hub_clawhub.py",
    "tools/skills_hub_skillssh.py",
    "tools/skills_hub_sources.py",
    "hermes_cli/web_server_profiles.py",
)


def _hashlib_calls(path: Path) -> list[ast.Call]:
    tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
    calls: list[ast.Call] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            if func.value.id == "hashlib" and func.attr in {"md5", "sha1"}:
                calls.append(node)
    return calls


@pytest.mark.parametrize("module", TARGET_MODULES)
def test_hashlib_digest_calls_set_usedforsecurity(module: str) -> None:
    path = REPO_ROOT / module
    assert path.is_file(), f"missing module: {module}"
    calls = _hashlib_calls(path)
    assert calls, f"no hashlib.md5/sha1 calls found in {module}"
    for call in calls:
        keywords = {kw.arg for kw in call.keywords}
        assert "usedforsecurity" in keywords, (
            f"{module}:{call.lineno} calls hashlib without usedforsecurity=False "
            "(crashes on FIPS-enforced interpreters)"
        )
