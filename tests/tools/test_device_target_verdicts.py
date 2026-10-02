"""Execution-faithful target-floor regressions for G1–G7."""

import json
import os
import posixpath
import shlex
import subprocess
import tempfile
from pathlib import Path

import pytest

from tools import terminal_tool as terminal
from tools.environments.base import BaseEnvironment
from tools.registry import registry
from tools.terminal_tool_lifecycle import cleanup_vm


class IdentityBackend:
    def __init__(self, outcomes=None):
        self.outcomes = outcomes or {}
        self.queries = []

    def fetch_device_identity(self, path):
        self.queries.append(path)
        return self.outcomes.get(path, ("not_device", path))


@pytest.mark.parametrize("command", [
    "wipefs -a /tmp/rawnode", "cp source /tmp/rawnode", "printf x > /tmp/rawnode",
    "wipefs -a /tmp/alias",
])
@pytest.mark.parametrize("force", [False, True])
def test_proven_device_verdict_survives_identity_rewrite(command, force, monkeypatch):
    env = IdentityBackend({"/tmp/rawnode": ("device", "/tmp/rawnode"),
                           "/tmp/alias": ("device", "/tmp/rawnode")})
    targets = terminal._resolved_guard_targets(command, env, "/work")
    assert [target.status for target in targets] == ["prohibited_device"]
    monkeypatch.setattr(terminal, "_check_all_guards", lambda *a, **kw: {"approved": True})
    with pytest.raises(terminal._Rejected, match="mutation target"):
        terminal._run_approval_guards(command, "local", {}, force=force, env=env, cwd="/work")


@pytest.mark.parametrize("path", ["~", "~/alias", "~user/alias", "~+/alias", "~-/alias"])
@pytest.mark.parametrize("template", ["wipefs -a {}", "printf x > {}", "dd of={}"])
def test_active_tilde_is_indeterminate_without_backend_context(path, template):
    env = IdentityBackend()
    with pytest.raises(terminal._GuardTargetIndeterminate) as raised:
        terminal._resolved_guard_targets(template.format(path), env, "/work")
    assert raised.value.verdict.status == "indeterminate"
    assert not env.queries


@pytest.mark.parametrize("path", ["'~/alias'", '"~+/alias"', r"\~user/alias"])
@pytest.mark.parametrize("template", ["wipefs -a {}", "printf x > {}", "dd of={}"])
def test_literal_tilde_keeps_quote_and_escape_provenance(path, template):
    env = IdentityBackend()
    targets = terminal._resolved_guard_targets(template.format(path), env, "/work")
    assert [target.status for target in targets] == ["safe"]
    assert env.queries == [posixpath.join("/work", shlex.split(path)[0])]


@pytest.mark.parametrize("command", [
    "cp source /work/dest", "cp -t /work/dest source", "cp -t/work/dest source",
    "cp --target-directory=/work/dest source", "cp --target-directory /work/dest source",
])
def test_cp_probes_effective_child_not_just_destination_directory(command):
    env = IdentityBackend({"/work/dest": ("directory", "/work/dest"),
                           "/work/dest/source": ("device", "/tmp/rawnode")})
    targets = terminal._resolved_guard_targets(command, env, "/work")
    assert [(target.path, target.status) for target in targets] == [
        ("/work/dest/source", "prohibited_device")]
    with pytest.raises(terminal._Rejected):
        terminal._run_approval_guards(command, "local", {}, force=True, env=env, cwd="/work")


def test_cp_T_does_not_project_destination_children():
    env = IdentityBackend({"/work/dest": ("directory", "/work/dest"),
                           "/work/dest/source": ("device", "/tmp/rawnode")})
    targets = terminal._resolved_guard_targets("cp -T source /work/dest", env, "/work")
    assert [(target.path, target.status) for target in targets] == [("/work/dest", "safe")]
    assert env.queries == ["/work/dest"]


def test_cp_multiple_source_children_each_receive_a_verdict():
    env = IdentityBackend({"/work/dest": ("directory", "/work/dest"),
                           "/work/dest/b": ("device", "/tmp/rawnode")})
    targets = terminal._resolved_guard_targets("cp a b /work/dest", env, "/work")
    assert [(t.path, t.status) for t in targets] == [
        ("/work/dest/a", "safe"), ("/work/dest/b", "prohibited_device")]


@pytest.mark.parametrize("command", ["cp -r source dest", "cp -t dest -T source", "cp a b dest"])
def test_unsupported_cp_layout_is_indeterminate(command):
    with pytest.raises(terminal._GuardTargetIndeterminate):
        terminal._resolved_guard_targets(command, IdentityBackend(), "/work")


@pytest.mark.parametrize("command", [
    "mkswap alias", "mkswap alias 1024", "mkswap -L swap alias 1024",
    "mkswap --pagesize 4096 --uuid=clear alias 1024", "mkswap -p4096 alias 1024",
])
def test_mkswap_size_and_option_values_do_not_replace_device(command):
    env = IdentityBackend({"/work/alias": ("device", "/tmp/rawnode")})
    targets = terminal._resolved_guard_targets(command, env, "/work")
    assert [t.status for t in targets] == ["prohibited_device"]
    assert env.queries == ["/work/alias"]


def test_regular_swapfile_receives_safe_verdict_with_size():
    env = IdentityBackend()
    targets = terminal._resolved_guard_targets("mkswap swapfile 1024", env, "/work")
    assert [(t.path, t.status) for t in targets] == [("/work/swapfile", "safe")]


@pytest.mark.parametrize("wrapper", ["sudo", "sudo -u root", "doas", "su root -c"])
def test_privilege_context_does_not_accept_unprivileged_missing_proof(wrapper):
    env = IdentityBackend({"/sealed/alias": ("missing", None)})
    command = (f"{wrapper} 'wipefs -a /sealed/alias'" if wrapper.startswith("su ")
               else f"{wrapper} wipefs -a /sealed/alias")
    with pytest.raises(terminal._GuardTargetIndeterminate):
        terminal._resolved_guard_targets(command, env, "/work")


@pytest.mark.parametrize("command", ["newfs alias", "newfs -O 2 -L volume alias", "newfs -j alias"])
def test_plain_newfs_projects_special_file(command):
    env = IdentityBackend({"/work/alias": ("device", "/tmp/rawnode")})
    assert [t.status for t in terminal._resolved_guard_targets(command, env, "/work")] == [
        "prohibited_device"]
    assert env.queries == ["/work/alias"]


def test_newfs_no_write_control_does_not_probe():
    env = IdentityBackend({"/work/alias": ("device", "/tmp/rawnode")})
    assert terminal._resolved_guard_targets("newfs -N alias", env, "/work") == []
    assert env.queries == []


def test_newfs_N_option_value_is_not_a_no_write_flag():
    env = IdentityBackend({"/work/alias": ("device", "/tmp/rawnode")})
    assert [t.status for t in terminal._resolved_guard_targets("newfs -L -N alias", env, "/work")] == [
        "prohibited_device"]


@pytest.mark.parametrize("command", ["newfs -NU alias", "newfs -N -O 2 alias"])
def test_newfs_no_write_grammar_handles_combined_options(command):
    assert terminal._resolved_guard_targets(command, IdentityBackend(), "/work") == []


@pytest.mark.parametrize("spacing", ["", " "])
def test_combined_append_operator_checks_actual_target(spacing):
    env = IdentityBackend({"/work/alias": ("device", "/tmp/rawnode")})
    with pytest.raises(terminal._Rejected):
        terminal._run_approval_guards(f"echo msg &>>{spacing}alias", "local", {},
                                     force=True, env=env, cwd="/work")
    assert env.queries == ["/work/alias"]


@pytest.mark.parametrize("operator", ["&>>", "&>", ">>", ">"])
def test_truncated_redirections_stay_indeterminate(operator):
    with pytest.raises(terminal._GuardTargetIndeterminate):
        terminal._resolved_guard_targets(f"echo msg {operator}", IdentityBackend(), "/work")


def dispatch(command, tmp_path, monkeypatch):
    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    monkeypatch.setattr(terminal, "_check_all_guards", lambda *a, **kw: {"approved": True})
    task = "typed-device-regression"
    try:
        result = registry.dispatch("terminal", {"command": command, "workdir": str(tmp_path)}, task_id=task)
        return json.loads(result) if isinstance(result, str) else result
    finally:
        cleanup_vm(task)


@pytest.mark.parametrize("command", ["cp source dest", "cp -t dest source",
                                    "cp --target-directory=dest source", "cp -T source output"])
def test_real_copy_layouts_reach_registered_terminal(command, tmp_path, monkeypatch):
    (tmp_path / "source").write_text("payload")
    (tmp_path / "dest").mkdir()
    result = dispatch(command, tmp_path, monkeypatch)
    assert result["exit_code"] == 0, result
    output = tmp_path / "output" if "-T" in command else tmp_path / "dest" / "source"
    assert output.read_text() == "payload"


@pytest.mark.parametrize("spacing", ["", " "])
def test_real_combined_append_reaches_registered_terminal(spacing, tmp_path, monkeypatch):
    result = dispatch(f"sh -c 'printf out; printf err >&2' &>>{spacing}notes", tmp_path, monkeypatch)
    assert result["exit_code"] == 0, result
    assert (tmp_path / "notes").read_text() == "outerr"


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX PTYs")
@pytest.mark.parametrize("layout", ["cp source dest", "cp -t dest source",
                                   "cp --target-directory=dest source", "printf x > {device}"])
def test_real_device_identity_blocks_registered_mutation(layout, tmp_path, monkeypatch):
    master, slave = os.openpty()
    try:
        device = os.ttyname(slave)
        (tmp_path / "source").write_text("payload")
        (tmp_path / "dest").mkdir()
        (tmp_path / "dest" / "source").symlink_to(device)
        result = dispatch(layout.format(device=shlex.quote(device)), tmp_path, monkeypatch)
        assert result["status"] == "blocked", result
    finally:
        os.close(master)
        os.close(slave)


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX traversal permissions")
def test_backend_probe_preserves_permission_failure_and_true_absence():
    class UnprivilegedProbe:
        fetch_device_identity = BaseEnvironment.fetch_device_identity

        def execute(self, command, **kwargs):
            result = subprocess.run(["bash", "-c", command], capture_output=True, text=True,
                                    check=False)
            return {"returncode": result.returncode, "output": result.stdout + result.stderr}

    with tempfile.TemporaryDirectory() as root:
        root = Path(root)
        root.chmod(0o755)
        sealed = root / "sealed"
        sealed.mkdir(mode=0o700)
        (sealed / "alias").symlink_to("/dev/null")
        sealed.chmod(0)
        if os.access(sealed, os.X_OK):
            sealed.chmod(0o700)
            pytest.skip("probe process has traversal override capabilities")
        env = UnprivilegedProbe()
        assert env.fetch_device_identity(str(sealed / "alias")) == ("indeterminate", None)
        assert env.fetch_device_identity(str(root / "absent")) == ("missing", None)
        sealed.chmod(0o755)
        assert env.fetch_device_identity(str(sealed / "alias")) == ("device", "/dev/null")
