"""Tests for path resolution and config validation (paths.py).

Covers acceptance test #8 (`../`, absolute outside-root, symlink escapes
refused), the spec's "Reject invalid or unsafe configuration at load"
contract, and the hard ceiling compilation cap of 24,000.
"""
from __future__ import annotations

from pathlib import Path

import pytest


from vault_retrieval.paths import (
    resolve_vault_root,
    resolve_safe_path,
    VaultConfigError,
    VaultPathError,
    validate_config,
    HARD_CEILING_COMPILED_CAP,
)


class TestResolveVaultRoot:
    def test_accepts_existing_directory(self, vault_root):
        resolved = resolve_vault_root(str(vault_root))
        assert resolved == vault_root.resolve()

    def test_rejects_nonexistent_path(self, tmp_path):
        with pytest.raises(VaultConfigError):
            resolve_vault_root(str(tmp_path / "does-not-exist"))

    def test_rejects_file_path(self, tmp_path):
        f = tmp_path / "file.md"
        f.write_text("hi")
        with pytest.raises(VaultConfigError):
            resolve_vault_root(str(f))

    def test_rejects_empty_string(self):
        with pytest.raises(VaultConfigError):
            resolve_vault_root("")


class TestResolveSafePath:
    def test_relative_path_resolved_under_root(self, vault_root, small_note):
        target = resolve_safe_path(vault_root.resolve(), "99 System/note.md")
        assert target == small_note.resolve()

    def test_dotdot_rejected(self, vault_root):
        with pytest.raises(VaultPathError):
            resolve_safe_path(vault_root.resolve(), "../etc/passwd")

    def test_absolute_outside_root_rejected(self, vault_root, tmp_path):
        with pytest.raises(VaultPathError):
            resolve_safe_path(vault_root.resolve(), str(tmp_path / "elsewhere.md"))

    def test_symlink_escape_rejected(self, vault_root, tmp_path):
        outside = tmp_path / "outside.md"
        outside.write_text("secret")
        link = vault_root / "link.md"
        link.symlink_to(outside)
        with pytest.raises(VaultPathError):
            resolve_safe_path(vault_root.resolve(), "link.md")

    def test_binary_file_rejected(self, vault_root):
        bin_path = vault_root / "binary.bin"
        bin_path.write_bytes(b"\x00\x01\x02")
        with pytest.raises(VaultPathError):
            resolve_safe_path(vault_root.resolve(), "binary.bin")

    def test_path_traversal_double_slash_rejected(self, vault_root):
        with pytest.raises(VaultPathError):
            resolve_safe_path(vault_root.resolve(), "..//foo")


class TestValidateConfig:
    def test_default_config_passes(self, vault_root):
        cfg = {
            "vault_root": str(vault_root),
            "mode": "enforce",
            "default_budget_chars": 12000,
            "hard_ceiling_chars": 24000,
            "large_file_chars": 20000,
            "candidate_limit": 20,
            "max_primary_extracts": 3,
            "max_expansion_extracts": 2,
            "max_range_lines": 120,
            "max_range_chars": 8000,
            "query_log_enabled": True,
            "log_raw_query_terms": False,
            "snapshots_enabled": False,
            "fts5_enabled": False,
            "block_direct_file_reads": True,
        }
        out = validate_config(cfg)
        assert out["hard_ceiling_chars"] == 24000
        assert out["default_budget_chars"] == 12000

    def test_hard_ceiling_above_compiled_cap_rejected(self, vault_root):
        cfg = {
            "vault_root": str(vault_root),
            "default_budget_chars": 12000,
            "hard_ceiling_chars": HARD_CEILING_COMPILED_CAP + 1,
            "large_file_chars": 20000,
            "candidate_limit": 20,
            "max_primary_extracts": 3,
            "max_expansion_extracts": 2,
            "max_range_lines": 120,
            "max_range_chars": 8000,
            "query_log_enabled": True,
            "log_raw_query_terms": False,
            "snapshots_enabled": False,
            "fts5_enabled": False,
            "block_direct_file_reads": True,
            "mode": "enforce",
        }
        with pytest.raises(VaultConfigError):
            validate_config(cfg)

    def test_default_strictly_greater_than_hard_ceiling_rejected(self, vault_root):
        cfg = {
            "vault_root": str(vault_root),
            "default_budget_chars": 24001,  # strictly greater
            "hard_ceiling_chars": 24000,
            "large_file_chars": 20000,
            "candidate_limit": 20,
            "max_primary_extracts": 3,
            "max_expansion_extracts": 2,
            "max_range_lines": 120,
            "max_range_chars": 8000,
            "query_log_enabled": True,
            "log_raw_query_terms": False,
            "snapshots_enabled": False,
            "fts5_enabled": False,
            "block_direct_file_reads": True,
            "mode": "enforce",
        }
        with pytest.raises(VaultConfigError):
            validate_config(cfg)

    def test_negative_limit_rejected(self, vault_root):
        cfg = {
            "vault_root": str(vault_root),
            "default_budget_chars": 12000,
            "hard_ceiling_chars": 24000,
            "large_file_chars": -1,
            "candidate_limit": 20,
            "max_primary_extracts": 3,
            "max_expansion_extracts": 2,
            "max_range_lines": 120,
            "max_range_chars": 8000,
            "query_log_enabled": True,
            "log_raw_query_terms": False,
            "snapshots_enabled": False,
            "fts5_enabled": False,
            "block_direct_file_reads": True,
            "mode": "enforce",
        }
        with pytest.raises(VaultConfigError):
            validate_config(cfg)

    def test_log_raw_query_terms_true_rejected(self, vault_root):
        cfg = {
            "vault_root": str(vault_root),
            "default_budget_chars": 12000,
            "hard_ceiling_chars": 24000,
            "large_file_chars": 20000,
            "candidate_limit": 20,
            "max_primary_extracts": 3,
            "max_expansion_extracts": 2,
            "max_range_lines": 120,
            "max_range_chars": 8000,
            "query_log_enabled": True,
            "log_raw_query_terms": True,  # FORBIDDEN
            "snapshots_enabled": False,
            "fts5_enabled": False,
            "block_direct_file_reads": True,
            "mode": "enforce",
        }
        with pytest.raises(VaultConfigError):
            validate_config(cfg)

    def test_invalid_mode_rejected(self, vault_root):
        cfg = {
            "vault_root": str(vault_root),
            "default_budget_chars": 12000,
            "hard_ceiling_chars": 24000,
            "large_file_chars": 20000,
            "candidate_limit": 20,
            "max_primary_extracts": 3,
            "max_expansion_extracts": 2,
            "max_range_lines": 120,
            "max_range_chars": 8000,
            "query_log_enabled": True,
            "log_raw_query_terms": False,
            "snapshots_enabled": False,
            "fts5_enabled": False,
            "block_direct_file_reads": True,
            "mode": "panic",
        }
        with pytest.raises(VaultConfigError):
            validate_config(cfg)
