"""bubblewrap is a host-path backend: the approval gate
treats it like local, and it belongs to none of the container or remote
backend classification sets, so file tools, prompt builder, env probe,
skills tool and credential files treat it as local. The code execution
tool has no set of its own: it asks terminal_tool._is_container_backend,
which reads the terminal_tool_config._CONTAINER_BACKENDS set checked below.
The built-in name sets are checked too: bubblewrap is in-tree, so a plugin
cannot register a backend under its name.

No test here spawns bwrap.
"""

import inspect
import re

import pytest

from tools import approval

# A collection literal holding "docker": every backend classification set is
# one of these, named or inline.
_DOCKER_LITERAL = re.compile(r'[{(\[][^{}()\[\]]*"docker"[^{}()\[\]]*[})\]]')


def _docker_literals(module) -> list[str]:
    return _DOCKER_LITERAL.findall(inspect.getsource(module))


class TestClassificationSets:
    def test_named_sets_exclude_bubblewrap(self):
        import agent.prompt_builder as prompt_builder
        import tools.env_probe as env_probe
        import tools.file_tools_paths as file_tools_paths
        import tools.skills_tool_setup as skills_tool_setup
        import tools.terminal_tool_config as terminal_tool_config

        named = {
            "terminal_tool_config._CONTAINER_BACKENDS": terminal_tool_config._CONTAINER_BACKENDS,
            "file_tools_paths._CONTAINER_PATH_BACKENDS_FALLBACK": file_tools_paths._CONTAINER_PATH_BACKENDS_FALLBACK,
            "prompt_builder._REMOTE_TERMINAL_BACKENDS": prompt_builder._REMOTE_TERMINAL_BACKENDS,
            "env_probe._REMOTE_BACKENDS": env_probe._REMOTE_BACKENDS,
            "skills_tool_setup._REMOTE_ENV_BACKENDS": skills_tool_setup._REMOTE_ENV_BACKENDS,
        }
        for name, members in named.items():
            assert "docker" in members, name  # the set is the one we mean
            assert "bubblewrap" not in members, name

    def test_builtin_name_sets_reserve_bubblewrap(self):
        from agent.terminal_env_registry import BUILTIN_BACKEND_NAMES
        from hermes_cli.doctor_tools import _BUILTIN_TERMINAL_BACKENDS
        from tools.terminal_tool_backends import _BUILTIN_BACKENDS, _ENV_BUILDERS

        assert "bubblewrap" in BUILTIN_BACKEND_NAMES
        assert "bubblewrap" in _BUILTIN_TERMINAL_BACKENDS
        assert "bubblewrap" in _BUILTIN_BACKENDS.split(", ")
        assert "bubblewrap" in _ENV_BUILDERS

    # terminal_tool_backends is left out on purpose: its builder table and
    # built-in name list are meant to name bubblewrap.
    @pytest.mark.parametrize("module_name", [
        "tools.terminal_tool",
        "tools.terminal_tool_config",
        "tools.file_tools_paths",
        "agent.prompt_builder",
        "tools.env_probe",
        "tools.skills_tool_setup",
        "tools.credential_files",
    ])
    def test_inline_backend_literals_exclude_bubblewrap(self, module_name):
        module = __import__(module_name, fromlist=["_"])
        literals = _docker_literals(module)
        assert literals, f"{module_name} has no backend literal naming docker"
        assert [lit for lit in literals if "bubblewrap" in lit] == []


class TestApprovalGate:
    @pytest.mark.parametrize("has_host_access", [True, False])
    def test_container_guards_are_not_skipped(self, has_host_access):
        assert approval._should_skip_container_guards("bubblewrap", has_host_access) is False
        assert approval._should_skip_container_guards("local", has_host_access) is False

    def test_hardline_command_gets_the_local_verdict(self):
        command = "rm -rf /"
        assert approval.detect_hardline_command(command)[0] is True
        local = approval.check_dangerous_command(command, "local")
        bubblewrap = approval.check_dangerous_command(command, "bubblewrap")
        assert bubblewrap == local
        assert local["approved"] is False

    def test_dangerous_command_reaches_the_approval_gate_as_for_local(self, monkeypatch):
        command = "rm -rf ./build"
        assert approval.detect_dangerous_command(command)[0] is True
        reached = []

        def fake_gate(**kwargs):
            reached.append(kwargs["pattern_key"])
            return {"approved": False, "message": "gate"}

        monkeypatch.setattr(approval, "_run_approval_gate", fake_gate)
        local = approval.check_dangerous_command(command, "local")
        bubblewrap = approval.check_dangerous_command(command, "bubblewrap")
        assert bubblewrap == local == {"approved": False, "message": "gate"}
        assert len(reached) == 2 and reached[0] == reached[1]
        # A sandboxed container backend skips the gate entirely.
        assert approval.check_dangerous_command(command, "docker")["approved"] is True
        assert len(reached) == 2


class TestHostPathGates:
    """Gates outside the named sets that pick host-path behavior for local
    must pick it for bubblewrap too."""

    def test_image_source_treats_bubblewrap_as_host_side(self, monkeypatch):
        from tools import image_source

        monkeypatch.setenv("TERMINAL_ENV", "bubblewrap")
        assert image_source._is_local_terminal_backend() is True
        monkeypatch.setenv("TERMINAL_ENV", "docker")
        assert image_source._is_local_terminal_backend() is False

    def test_vision_tools_reads_video_sources_host_side_for_bubblewrap(self):
        # vision_tools has no gate of its own: it asks image_source, checked above.
        from tools import vision_tools

        source = inspect.getsource(vision_tools._materialize_video)
        assert "_is_local_terminal_backend" in source
        assert "not _is_local_terminal_backend() and path_like" in source

    def test_image_generation_source_gate_names_bubblewrap_beside_local(self):
        from tools import image_generation_tool

        source = inspect.getsource(image_generation_tool._confine_source_images)
        assert 'in ("", "local", "bubblewrap")' in source

    def test_cwd_placeholder_resolves_messaging_cwd_for_bubblewrap(self):
        from gateway import cwd_placeholder

        source = inspect.getsource(cwd_placeholder.resolve_placeholder_terminal_cwd)
        assert 'in ("local", "bubblewrap")' in source

    @pytest.mark.parametrize("path, needle", [
        ("cli.py", 'effective_backend in ("local", "bubblewrap")'),
        ("run_agent.py", 'not in ("", "local", "bubblewrap")'),
        ("tui_gateway/session_workdir.py", 'backend not in ("local", "bubblewrap")'),
        ("tui_gateway/session_workdir.py", 'backend in ("local", "bubblewrap")'),
    ])
    def test_source_gates_name_bubblewrap_beside_local(self, path, needle):
        from pathlib import Path

        root = Path(__file__).resolve().parents[2]
        assert needle in (root / path).read_text(), path

    def test_tui_gateway_has_both_local_gates(self):
        from tui_gateway import session_workdir

        assert 'not in ("local", "bubblewrap")' in inspect.getsource(session_workdir._terminal_task_cwd_with_source)
        assert 'in ("local", "bubblewrap")' in inspect.getsource(session_workdir._is_local_terminal_backend)
