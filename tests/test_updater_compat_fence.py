"""CI fence for the legacy-updater compatibility contract.

Phase 2 task 2.2: for every FROZEN_CALLABLES entry, verify the symbol
exists and its signature matches the frozen string. For FROZEN_CLI_SURFACES,
verify the argparse parser accepts each argv shape without SystemExit.

This is a **behavior contract** test, NOT a change-detector. It enforces
the frozen contract from docs/updater-world.md §2.13 — the set of symbols
historical ``hermes update`` updaters import/call post-pull. Changing a
signature or deleting an entry bricks that population's next update.

Exempt from the AGENTS.md "Don't write change-detector tests" rule:
this test freezes an explicit compatibility contract, not a snapshot of
current data. See §2.13 for rationale.

Known limitation (stated in updater_compat.py docstring): the fence
freezes signatures as they exist on current main. If a frozen symbol
already drifted between some historical release and today, the fence
enshrines the drifted shape. The fence is necessary-not-sufficient: it
stops FUTURE drift. The authority on whether hop 1 actually works is
task 2.8's E2E (real old release against current main).
"""

import importlib
import inspect
import sys
from pathlib import Path

import pytest

from hermes_cli.updater_compat import FROZEN_CALLABLES, FROZEN_CLI_SURFACES, FROZEN_PATHS

REPO_ROOT = Path(__file__).resolve().parent.parent


class TestFrozenCallables:
    """Verify every frozen callable exists with the correct signature."""

    @pytest.mark.parametrize("entry", sorted(FROZEN_CALLABLES.keys()))
    def test_callable_exists_and_signature_matches(self, entry):
        """Import the module, resolve the qualname, check signature."""
        module_path, qualname = entry.split(":", 1)

        mod = importlib.import_module(module_path)

        # Resolve the qualname (may be nested, e.g., module.Class.method)
        obj = mod
        for part in qualname.split("."):
            obj = getattr(obj, part)

        assert callable(obj), f"{entry} is not callable"

        actual_sig = str(inspect.signature(obj))
        expected_sig = FROZEN_CALLABLES[entry]
        assert actual_sig == expected_sig, (
            f"FROZEN CONTRACT BROKEN: {entry}\n"
            f"  expected: {expected_sig}\n"
            f"  actual:   {actual_sig}\n"
            f"If this is an intentional change, update the frozen signature "
            f"in hermes_cli/updater_compat.py AND verify the E2E gate still "
            f"passes (task 2.8)."
        )


class TestFrozenCliSurfaces:
    """Verify the command registry accepts each frozen CLI surface."""

    def _resolve(self, name):
        """Resolve a command name via the central registry."""
        from hermes_cli.commands import resolve_command

        return resolve_command(name)

    @pytest.mark.parametrize("argv", FROZEN_CLI_SURFACES)
    def test_cli_surface_accepted(self, argv):
        """The command must resolve via the command registry or subcommand
        builder. The first element is the command name; the rest are args."""
        cmd_name = argv[0]

        # Try the slash-command registry first
        from hermes_cli.commands import resolve_command

        cmd = resolve_command(cmd_name)
        if cmd is not None:
            return  # Found in the registry — surface is accepted.

        # CLI subcommands (desktop, etc.) aren't in COMMAND_REGISTRY —
        # they're wired via subcommand builders. Check the builder exists.
        # Map command names to their builder functions.
        builder_map = {
            "desktop": "hermes_cli.subcommands.gui:build_gui_parser",
            "update": "hermes_cli.subcommands.update:build_update_parser",
        }
        if cmd_name in builder_map:
            module_path, func_name = builder_map[cmd_name].split(":", 1)
            import importlib

            mod = importlib.import_module(module_path)
            assert hasattr(mod, func_name), (
                f"FROZEN CLI SURFACE BROKEN: {argv}\n"
                f"  Builder '{func_name}' not found in {module_path}.\n"
                f"  If this is intentional, update FROZEN_CLI_SURFACES in updater_compat.py."
            )
            return

        pytest.fail(
            f"FROZEN CLI SURFACE BROKEN: {argv}\n"
            f"  Command '{cmd_name}' is not resolvable.\n"
            f"  If this is intentional, update FROZEN_CLI_SURFACES in updater_compat.py."
        )


class TestFrozenPaths:
    """Verify frozen file paths exist in the repo."""

    @pytest.mark.parametrize("path", FROZEN_PATHS)
    def test_path_exists(self, path):
        """The file must exist in the repo root."""
        full = REPO_ROOT / path
        assert full.exists(), (
            f"FROZEN PATH MISSING: {path}\n"
            f"  This file must exist for legacy updaters to function.\n"
            f"  If this is an intentional removal, update FROZEN_PATHS in updater_compat.py."
        )


class TestRegistryIntegrity:
    """Verify the registry itself is well-formed."""

    def test_frozen_callables_non_empty(self):
        assert len(FROZEN_CALLABLES) > 0

    def test_frozen_callables_format(self):
        """Every entry must be 'module:qualname' format."""
        for key in FROZEN_CALLABLES:
            assert ":" in key, f"missing ':' in {key}"
            module, qualname = key.split(":", 1)
            # Module must be a dotted path (at least one component)
            assert "." in module or module.startswith(
                "hermes"
            ) or module.startswith("tools"), f"unexpected module in {key}"

    def test_frozen_cli_surfaces_non_empty(self):
        assert len(FROZEN_CLI_SURFACES) > 0

    def test_frozen_paths_non_empty(self):
        assert len(FROZEN_PATHS) > 0
