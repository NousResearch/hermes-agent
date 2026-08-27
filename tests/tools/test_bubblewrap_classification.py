"""bubblewrap is a host-path backend: the approval gate
treats it like local, and it belongs to none of the container or remote
backend classification sets, so file tools, prompt builder, env probe,
skills tool and credential files treat it as local. The code execution
tool has no set of its own: it asks terminal_tool._is_container_backend,
which reads the _CONTAINER_BACKENDS set checked below.

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
        import tools.file_tools as file_tools
        import tools.skills_tool as skills_tool
        import tools.terminal_tool as terminal_tool

        named = {
            "terminal_tool._CONTAINER_BACKENDS": terminal_tool._CONTAINER_BACKENDS,
            "file_tools._CONTAINER_PATH_BACKENDS_FALLBACK": file_tools._CONTAINER_PATH_BACKENDS_FALLBACK,
            "prompt_builder._REMOTE_TERMINAL_BACKENDS": prompt_builder._REMOTE_TERMINAL_BACKENDS,
            "env_probe._REMOTE_BACKENDS": env_probe._REMOTE_BACKENDS,
            "skills_tool._REMOTE_ENV_BACKENDS": skills_tool._REMOTE_ENV_BACKENDS,
        }
        for name, members in named.items():
            assert "docker" in members, name  # the set is the one we mean
            assert "bubblewrap" not in members, name

    @pytest.mark.parametrize("module_name", [
        "tools.terminal_tool",
        "tools.file_tools",
        "agent.prompt_builder",
        "tools.env_probe",
        "tools.skills_tool",
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
