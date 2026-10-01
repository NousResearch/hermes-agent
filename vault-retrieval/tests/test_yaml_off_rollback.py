"""Regression tests for the ``mode: off`` operator rollback contract.

The 2026-10-02 00:15 +07 reconsideration gate REJECTED commit
``14258c37556df42032faeed10c5921a138c1d814`` for LIMITED PILOT because
the documented ``mode: off`` rollback failed through the real YAML
loader: PyYAML's ``safe_load`` defaults to YAML 1.1 and turns unquoted
``mode: off`` into Python ``bool(False)``; the validator rejected
``False`` as a non-string and ``register()`` failed closed, which is the
OPPOSITE of an operator rollback (built-in file tools stay blocked AND
the ``vault_context`` surface disappears).

These tests pin BOTH the YAML-loader defense (unit) AND the end-to-end
runtime path through ``discover_plugins()`` that the operator rollback
will exercise.

Acceptance contract:

  1. ``mode: off`` (unquoted) loaded through the real PyYAML loader is
     normalized to the documented string ``"off"`` BEFORE the allow-list
     check, so the operator's literal rollback line works.
  2. ``mode: "off"`` (quoted) is the canonical form; behaves identically.
  3. ``mode: True`` (boolean) is explicitly refused — we do NOT silently
     promote it to any string; the architecture memo enumerates exactly
     three modes.
  4. End-to-end through ``discover_plugins()``: a profile whose config
     contains ``mode: off`` (unquoted) still registers the tool and the
     hook is a no-op so the built-in ``read_file``/``search_files``
     tools can read Vault paths without interference.
  5. End-to-end through ``discover_plugins()``: a profile whose config
     contains ``mode: "off"`` (quoted) is identical to (4).
  6. End-to-end through ``discover_plugins()``: ``mode: enforce``
     (which YAML 1.1 leaves as the string ``"off"``? — no, as ``"enforce"``)
     still registers the tool AND the hook blocks Vault reads. This is the
     regression guard: the rollback must not silently weaken ``enforce``.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Dict, Tuple

import pytest
import yaml


# Make the plugin importable when tests are run by pytest from the repo
# root. conftest.py already does this, but importing here keeps the file
# runnable in isolation when pytest collection starts mid-tree.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "vault-retrieval") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "vault-retrieval"))


from vault_retrieval.paths import (
    VaultConfigError,
    validate_config,
)
from vault_retrieval.plugin import register as plugin_register


# ---------------------------------------------------------------------------
# Pure YAML-loader defense
# ---------------------------------------------------------------------------


def _settings_block_yaml(mode_value: str) -> str:
    """Build the YAML snippet the plugin reads from
    ``plugins.entries.vault-retrieval.settings``.

    We only vary the ``mode:`` value; everything else is a known-good
    baseline.
    """
    return f"""
plugins:
  enabled:
    - vault-retrieval
  entries:
    vault-retrieval:
      enabled: true
      settings:
        vault_root: /root/Documents/Obsidian Vault
        mode: {mode_value}
        default_budget_chars: 12000
        hard_ceiling_chars: 24000
        large_file_chars: 20000
        candidate_limit: 20
        max_primary_extracts: 3
        max_expansion_extracts: 2
        max_range_lines: 120
        max_range_chars: 8000
        query_log_enabled: true
        log_raw_query_terms: false
        snapshots_enabled: false
        fts5_enabled: false
        block_direct_file_reads: true
"""


def _loaded_settings(mode_value: str) -> Dict:
    """Run the real PyYAML safe_load the operator's config.yaml exercises."""
    doc = yaml.safe_load(_settings_block_yaml(mode_value))
    return doc["plugins"]["entries"]["vault-retrieval"]["settings"]


class TestYAMLLoaderDefense:
    """PyYAML YAML 1.1 silently coerces ``off``/``on``/``yes``/``no`` to
    bool. The validator must normalize the boolean ``False`` to the
    documented ``"off"`` string for this single field, while still
    refusing boolean ``True`` (no silent promotion to ``"enforce"`` or
    similar)."""

    def test_unquoted_off_parses_to_bool_false_under_yaml_1_1(self):
        """First pin the bug: YAML 1.1 turns unquoted ``mode: off`` into
        ``False``. This is the regression we are documenting.
        """
        settings = _loaded_settings("off")
        assert settings["mode"] is False, (
            "PyYAML YAML 1.1 parses unquoted `mode: off` to bool False; "
            "this is the bug the normalization fixes."
        )

    def test_quoted_off_parses_to_string_off(self):
        """The canonical/quote-safe form parses to a real string."""
        settings = _loaded_settings('"off"')
        assert settings["mode"] == "off"

    def test_validator_normalizes_bool_false_to_off(self, vault_root):
        """After YAML loading, the validator must accept the boolean
        ``False`` and normalize it to ``"off"``.
        """
        cfg = dict(
            _loaded_settings("off"),
            vault_root=str(vault_root),
        )
        out = validate_config(cfg)
        assert out["mode"] == "off"

    def test_validator_normalizes_unquoted_through_documented_path(self, vault_root):
        """Drive the EXACT path the operator hits when they set
        ``mode: off`` in config.yaml and Hermes loads it:

          raw YAML -> PyYAML safe_load -> dict -> validate_config

        The end-to-end loop is the regression. We do not trust the unit
        piece in isolation because the failure mode (YAML 1.1 boolean
        coercion) lives in the loader, not the validator.
        """
        cfg = _loaded_settings("off")
        cfg["vault_root"] = str(vault_root)
        out = validate_config(cfg)
        assert out["mode"] == "off"

    def test_validator_quoted_off_also_accepted(self, vault_root):
        """The canonical/quote-safe form is also accepted."""
        cfg = _loaded_settings('"off"')
        cfg["vault_root"] = str(vault_root)
        out = validate_config(cfg)
        assert out["mode"] == "off"

    def test_validator_refuses_bool_true(self, vault_root):
        """``mode: True`` is NOT silently promoted to ``"enforce"`` (or
        any other value) — the architecture memo enumerates exactly three
        valid modes, all strings.
        """
        cfg = _loaded_settings("True")
        cfg["vault_root"] = str(vault_root)
        with pytest.raises(VaultConfigError) as ei:
            validate_config(cfg)
        # The error mentions boolean True specifically so the operator
        # sees the cause; the new error is distinguishable from the
        # generic "mode must be one of" path.
        assert "True" in str(ei.value) or "boolean" in str(ei.value).lower()

    def test_validator_refuses_bool_true_unquoted(self, vault_root):
        """Belt-and-suspenders: a profile that sets ``mode: true``
        (unquoted) must also be refused.
        """
        cfg = _loaded_settings("true")
        cfg["vault_root"] = str(vault_root)
        with pytest.raises(VaultConfigError):
            validate_config(cfg)

    def test_validator_accepts_enforce_string(self, vault_root):
        """``mode: enforce`` parses as a string under YAML 1.1 and stays
        in the allow-list; this proves the normalization is scoped to
        ``mode`` only.
        """
        cfg = _loaded_settings("enforce")
        cfg["vault_root"] = str(vault_root)
        out = validate_config(cfg)
        assert out["mode"] == "enforce"


# ---------------------------------------------------------------------------
# End-to-end through discover_plugins()
# ---------------------------------------------------------------------------


class _FakeRegistration:
    def __init__(self, kind, key, payload):
        self.kind = kind
        self.key = key
        self.payload = payload


class FakePluginContext:
    """In-memory stand-in for Hermes's PluginContext. Mirrors the one in
    ``tests/test_plugin.py`` so the plugin code under test sees the same
    surface; this is intentionally a duplicate (not an import) so this
    file is runnable in isolation."""

    def __init__(self, config=None):
        self._config = dict(config or {})
        self.tools = {}
        self.hooks = {}
        self.prompt_sections = {}

    def get_config(self, key, default=None):
        return self._config.get(key, default)

    def register_tool(
        self, name, toolset, schema, handler,
        description="", emoji="", override=False, **kwargs,
    ):
        reg = _FakeRegistration("tool", name, {
            "name": name, "toolset": toolset, "schema": schema,
            "handler": handler, "description": description,
        })
        self.tools[name] = reg
        return reg

    def register_hook(self, hook_name, callback):
        self.hooks.setdefault(hook_name, []).append(callback)
        return _FakeRegistration("hook", hook_name, callback)

    def register_system_prompt_section(
        self, id, content, *, position, max_chars, **kwargs,
    ):
        reg = _FakeRegistration("system_prompt_section", id, {
            "content": content, "position": position, "max_chars": max_chars,
        })
        self.prompt_sections[id] = reg
        return reg


def _make_plugin_context(tmp_path: Path, mode_yaml: str) -> Tuple[FakePluginContext, Path]:
    """Build a plugin context whose ``settings.mode`` came from the real
    YAML loader path. Returns (ctx, vault_root).
    """
    vault = tmp_path / "vault"
    vault.mkdir()
    # Seed with a real Vault-shaped file so the tool has something to
    # read; it does not matter for the off-mode check (the hook is a
    # no-op) but the hook tests still need a path under vault_root.
    (vault / "note.md").write_text("alpha\nbeta\n", encoding="utf-8")
    settings = _loaded_settings(mode_yaml)
    settings["vault_root"] = str(vault)
    ctx = FakePluginContext(settings)
    return ctx, vault


class TestDiscoverPluginsOffRollback:
    """End-to-end: ``mode: off`` (unquoted) keeps built-in file tools at
    baseline. The plugin still registers the tool (so an operator who
    rolls back does not lose ``vault_context``), but the pre_tool_call
    hook is a no-op so direct ``read_file`` / ``search_files`` calls
    resolve normally inside the Vault."""

    def test_off_unquoted_registers_tool_and_blocks_read_passes_through(self, tmp_path):
        """The operator's literal ``mode: off`` line must:

          1. register the ``vault_context`` tool (so the agent still
             has the documented retrieval surface),
          2. register the bounded prompt section (idempotent re-deploy),
          3. register the pre_tool_call hook,
          4. the hook's actual behavior on a Vault ``read_file`` must be
             a no-op (result is ``None``) — i.e. baseline file-tool
             behavior is restored.
        """
        from vault_retrieval.plugin import (
            make_pre_tool_call_hook,
            TOOL_NAME as _TOOL_NAME,
            _PROMPT_SECTION_ID,
        )
        ctx, vault = _make_plugin_context(tmp_path, "off")
        plugin_register(ctx)

        assert _TOOL_NAME in ctx.tools, (
            "mode: off must STILL register vault_context — losing the "
            "tool surface is not a rollback."
        )
        assert _PROMPT_SECTION_ID in ctx.prompt_sections
        assert "pre_tool_call" in ctx.hooks
        # The hook on a Vault path must be a no-op.
        hook = ctx.hooks["pre_tool_call"][-1]
        note = vault / "note.md"
        result = hook(tool_name="read_file", args={"path": str(note)})
        assert result is None, (
            "mode: off hook MUST pass through Vault reads — got "
            f"{result!r}. This is the rollback contract."
        )

    def test_off_quoted_registers_tool_and_blocks_read_passes_through(self, tmp_path):
        """Same contract for the canonical quoted form."""
        from vault_retrieval.plugin import (
            make_pre_tool_call_hook,
            TOOL_NAME as _TOOL_NAME,
        )
        ctx, vault = _make_plugin_context(tmp_path, '"off"')
        plugin_register(ctx)
        assert _TOOL_NAME in ctx.tools
        hook = ctx.hooks["pre_tool_call"][-1]
        note = vault / "note.md"
        assert hook(tool_name="read_file", args={"path": str(note)}) is None

    def test_enforce_still_blocks_unrelated_to_rollback_fix(self, tmp_path):
        """Regression guard for the fix: ``mode: enforce`` (a string under
        YAML 1.1) MUST still block Vault reads after the
        ``False → "off"`` normalization was added. The fix is scoped to
        the YAML 1.1 boolean coercion only and cannot have weakened
        enforce behavior.
        """
        from vault_retrieval.plugin import make_pre_tool_call_hook
        ctx, vault = _make_plugin_context(tmp_path, "enforce")
        plugin_register(ctx)
        hook = ctx.hooks["pre_tool_call"][-1]
        note = vault / "note.md"
        result = hook(tool_name="read_file", args={"path": str(note)})
        assert result is not None, "enforce must still block Vault reads"
        assert result["action"] == "block"

    def test_off_real_plugin_loader_path_discoverable(self, tmp_path, monkeypatch):
        """The plugin must also load through the real ``discover_plugins``
        loader when the canary config has ``mode: off`` (unquoted). This
        is the path the operator's deployed profile will actually
        exercise.

        We mirror ``tests/test_runtime_integration._isolated_hermes_home``
        but with ``mode: off`` written verbatim (no quote). If the
        normalization regresses, ``discover_plugins`` either raises or
        fails to register the tool.
        """
        home = tmp_path / "home"
        home.mkdir()
        plugins_dir = home / "plugins"
        plugins_dir.mkdir()
        plugin_src = Path(__file__).resolve().parents[1]  # vault-retrieval/
        shutil.copytree(plugin_src, plugins_dir / "vault-retrieval")

        vault = tmp_path / "vault"
        vault.mkdir()
        (vault / "note.md").write_text("alpha\n", encoding="utf-8")

        home.joinpath("config.yaml").write_text(
            "plugins:\n"
            "  enabled:\n"
            "    - vault-retrieval\n"
            "  entries:\n"
            "    vault-retrieval:\n"
            "      enabled: true\n"
            "      settings:\n"
            "        vault_root: " + str(vault) + "\n"
            "        mode: off\n"  # UNQUOTED — the documented operator line
            "        default_budget_chars: 12000\n"
            "        hard_ceiling_chars: 24000\n"
            "        large_file_chars: 20000\n"
            "        candidate_limit: 20\n"
            "        max_primary_extracts: 3\n"
            "        max_expansion_extracts: 2\n"
            "        max_range_lines: 120\n"
            "        max_range_chars: 8000\n"
            "        query_log_enabled: true\n"
            "        log_raw_query_terms: false\n"
            "        snapshots_enabled: false\n"
            "        fts5_enabled: false\n"
            "        block_direct_file_reads: true\n"
        )
        monkeypatch.setenv("HERMES_HOME", str(home))

        # Fresh subprocess-style module re-imports.
        for k in list(sys.modules):
            if k.startswith(("vault_retrieval", "hermes_cli.plugins")):
                del sys.modules[k]

        from hermes_cli.plugins import discover_plugins
        discover_plugins()
        from tools.registry import registry
        from vault_retrieval.tool import TOOL_NAME as _TOOL_NAME

        assert _TOOL_NAME in registry.get_all_tool_names(), (
            "discover_plugins() must still register vault_context for an "
            "operator who sets `mode: off` literally — losing the tool "
            "is not a rollback."
        )

        # Drive the tool with a real Vault selector to confirm the
        # normalization did not silently weaken the data surface.
        from model_tools import handle_function_call
        out = handle_function_call(
            function_name=_TOOL_NAME,
            function_args={
                "query": "rollback smoke",
                "selectors": [{
                    "path": "note.md",
                    "line_start": 1, "line_end": 2,
                }],
            },
            task_id="rollback-task",
            session_id="rollback-session",
            turn_id="rollback-turn",
            user_task="yaml-off-rollback-test",
        )
        env = json.loads(out)
        assert env["status"] == "ok", env
        assert env["extracts"], env

        # The pre_tool_call hook must pass through Vault reads in ``off``.
        from hermes_cli.plugins import invoke_hook
        block = invoke_hook(
            "pre_tool_call",
            tool_name="read_file",
            args={"path": str(vault / "note.md")},
        )
        # invoke_hook returns a list of non-None hook results; in off
        # mode every result is None, so the list is empty.
        assert not block, (
            f"mode: off hook must be a no-op; got {block!r} — the rollback "
            "did not restore baseline file-tool behavior."
        )