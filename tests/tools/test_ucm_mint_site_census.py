"""Mint-site census for ``ucm_structured_process`` authorization (Phase 1).

Phase 1 must introduce the factory only. No production dispatch site may call
``mint_ucm_auth_context`` yet. Later phases will allow exactly:

- agent/tool_executor.py (sequential quiet + non-quiet) — M1/M2
- agent/agent_runtime_helpers.py (invoke_tool) — M3

Until those phases are authorized, the production allowlist is empty.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

# Production mint sites authorized in future phases — empty in Phase 1.
PHASE1_ALLOWED_PRODUCTION_MINT_SITES: frozenset[str] = frozenset()

MINT_FUNCTION_NAMES = frozenset(
    {
        "mint_ucm_auth_context",
        "_mint_auth_context",
    }
)

# Paths relative to repo root that are allowed to *mention* the factory for
# definition/export (the module itself) or tests.
FACTORY_DEFINITION_FILE = "tools/ucm_auth_context.py"


def _iter_python_files(root: Path) -> list[Path]:
    skip_dirs = {
        ".git",
        ".venv",
        "venv",
        "__pycache__",
        "node_modules",
        "dist",
        "build",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "website",
    }
    files: list[Path] = []
    for path in root.rglob("*.py"):
        if any(part in skip_dirs for part in path.parts):
            continue
        files.append(path)
    return files


def _call_name(func: ast.AST) -> str | None:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _mint_call_sites(path: Path) -> list[tuple[int, str]]:
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
    except (OSError, SyntaxError) as exc:
        pytest.fail(f"cannot parse {path}: {exc}")

    sites: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node.func)
        if name in MINT_FUNCTION_NAMES:
            sites.append((node.lineno, name))
    return sites


class TestPhase1MintSiteCensus:
    def test_factory_module_exists(self):
        assert (REPO_ROOT / FACTORY_DEFINITION_FILE).is_file()

    def test_no_production_mint_call_sites_in_phase_1(self):
        """Production tree must not call the mint factory yet."""
        production_hits: list[str] = []
        for path in _iter_python_files(REPO_ROOT):
            rel = path.relative_to(REPO_ROOT).as_posix()
            if rel.startswith("tests/"):
                continue
            if rel == FACTORY_DEFINITION_FILE:
                # Definition + internal references only; filter actual Call nodes
                # that are not the function definition itself.
                for lineno, name in _mint_call_sites(path):
                    # mint_ucm_auth_context may appear only as the def name, not
                    # as a Call inside the factory module (factory constructs
                    # via _UcmAuthCapability(..., _seal=...)).
                    production_hits.append(f"{rel}:{lineno}:{name}")
                continue
            for lineno, name in _mint_call_sites(path):
                production_hits.append(f"{rel}:{lineno}:{name}")

        # Phase 1 allowlist is empty.
        unexpected = [
            hit
            for hit in production_hits
            if hit.split(":", 1)[0] not in PHASE1_ALLOWED_PRODUCTION_MINT_SITES
        ]
        # Factory module must not call mint_ucm_auth_context (would be recursive
        # / self-mint). Construction uses the private class + seal only.
        factory_self_calls = [
            hit for hit in production_hits if hit.startswith(FACTORY_DEFINITION_FILE)
        ]
        assert factory_self_calls == [], (
            "tools/ucm_auth_context.py must not call mint_ucm_auth_context; "
            f"found {factory_self_calls}"
        )
        assert unexpected == [], (
            "Phase 1 forbids production mint call sites; found: "
            + ", ".join(unexpected)
        )

    def test_phase1_allowlist_is_empty(self):
        assert PHASE1_ALLOWED_PRODUCTION_MINT_SITES == frozenset()

    def test_forbidden_dispatch_files_do_not_import_mint(self):
        """Guarded files must not yet import the mint factory."""
        forbidden_import_files = [
            "agent/tool_executor.py",
            "agent/agent_runtime_helpers.py",
            "model_tools.py",
            "toolsets.py",
            "tools/registry.py",
        ]
        for rel in forbidden_import_files:
            path = REPO_ROOT / rel
            assert path.is_file(), f"missing expected file {rel}"
            source = path.read_text(encoding="utf-8")
            assert "mint_ucm_auth_context" not in source, (
                f"{rel} must not reference mint_ucm_auth_context in Phase 1"
            )
            assert "ucm_auth_context" not in source, (
                f"{rel} must not import ucm_auth_context in Phase 1"
            )

    def test_no_model_visible_tool_registration_in_capability_module(self):
        path = REPO_ROOT / FACTORY_DEFINITION_FILE
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in tree.body:
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                func = node.value.func
                if (
                    isinstance(func, ast.Attribute)
                    and func.attr == "register"
                    and isinstance(func.value, ast.Name)
                    and func.value.id == "registry"
                ):
                    pytest.fail("capability module must not call registry.register")
