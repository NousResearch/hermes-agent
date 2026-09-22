"""``SSHEnvironment._sync_target_is_local`` must answer for the tree the file sync maps.

The manager syncs ``{remote_home}/.hermes`` — the same absolute path it uses as the *host* source
(`iter_sync_files(f"{self._remote_home}/.hermes")`) — while a profile runs with its own home
(``~/.hermes/profiles/<name>``). Anchoring the probe on that profile home made a same-disk target
undetectable and its self-sync stayed on (2026-09-19: profile ``trader`` →
``cameron@rei.taila6a102.ts.net``, its own host: 263 failed sync_backs, 7.2 GB tars in /tmp).

Invariants:
  * a target that reads the very file the sync maps disables sync, whatever the profile-home
    layout relative to that mapped tree;
  * a target that merely spells the same absolute paths on another machine keeps its sync — the
    probe answers a question about file identity, never about path spelling.
"""

import shlex
import subprocess
from pathlib import Path

import pytest

from hermes_constants import get_hermes_home
from tools.environments import ssh as ssh_env
from tools.environments.ssh import SSHEnvironment

_PROBE_GLOB = ".sync-probe-*"


def _probed_path(remote_cmd: str) -> Path:
    """The absolute path the probe asked the remote shell to read."""
    tokens = shlex.split(remote_cmd)
    assert tokens and tokens[0] == "cat", f"unexpected probe command: {remote_cmd!r}"
    return Path(tokens[1])


def _remote_shell(remote_root: Path | None):
    """Stand-in remote shell: ``None`` is this machine — the path *is* the local file, so the
    probe's own bytes come back — while *remote_root* is another machine holding its own copy
    of the same absolute paths."""

    def run_ssh(_self, remote_cmd, timeout):
        path = _probed_path(remote_cmd)
        if remote_root is not None:
            path = remote_root / path.relative_to("/")
        if path.is_file():
            return subprocess.CompletedProcess([], 0, stdout=path.read_text(encoding="utf-8"),
                                               stderr="")
        return subprocess.CompletedProcess([], 1, stdout="",
                                           stderr=f"cat: {path}: No such file or directory")

    return run_ssh


def _host_tree(tmp_path: Path, monkeypatch, profile_home_kind: str) -> tuple[Path, Path, Path]:
    """This host's layout: the hermes tree the sync maps, plus the launching profile's home."""
    host_home = tmp_path / "home"
    sync_root = host_home / ".hermes"
    (sync_root / "skills").mkdir(parents=True)
    (sync_root / "skills" / "SKILL.md").write_text("host skills", encoding="utf-8")
    profile_home = (sync_root / "profiles" / "trader" if profile_home_kind == "nested"
                    else tmp_path / "profile-homes" / "trader")
    profile_home.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(profile_home))
    assert get_hermes_home() == profile_home
    return host_home, sync_root, profile_home


def _build_env(monkeypatch, host_home: Path, run_ssh):
    """An SSHEnvironment reaching *host_home* through the fake remote shell; returns it and the
    kwargs every FileSyncManager construction was called with."""
    monkeypatch.setattr(ssh_env.shutil, "which", lambda _name: "/usr/bin/ssh")
    monkeypatch.setattr(ssh_env.SSHEnvironment, "_establish_connection", lambda self: None)
    monkeypatch.setattr(ssh_env.SSHEnvironment, "_detect_remote_home", lambda self: str(host_home))
    monkeypatch.setattr(ssh_env.SSHEnvironment, "_ensure_remote_dirs", lambda self: None)
    monkeypatch.setattr(ssh_env.SSHEnvironment, "init_session", lambda self: None)
    monkeypatch.setattr(ssh_env.SSHEnvironment, "_run_ssh", run_ssh)
    made = []
    monkeypatch.setattr(
        ssh_env, "FileSyncManager",
        lambda **kw: made.append(kw) or type("M", (), {
            "sync": lambda self, **k: None,
            "sync_back": lambda self: None,
        })(),
    )
    return SSHEnvironment(host="localhost", user="cameron"), made


@pytest.mark.parametrize("profile_home_kind", ["nested", "beside"])
def test_target_reading_the_synced_tree_disables_sync(tmp_path, monkeypatch, profile_home_kind):
    """The mapped tree decides, not the launch home: a same-disk target is recognised from
    either profile-home layout. Regression: the probe wrote the nonce into the profile home and
    read it back from the mapped root, so neither layout ever matched and sync stayed enabled."""
    host_home, sync_root, profile_home = _host_tree(tmp_path, monkeypatch, profile_home_kind)

    env, made = _build_env(monkeypatch, host_home, _remote_shell(None))

    assert env._sync_manager is None, "a target reading this host's own tree must not sync"
    assert made == []  # no FileSyncManager was constructed at all
    env.cleanup()  # the guarded path: no manager, nothing to sync back
    assert list(sync_root.glob(_PROBE_GLOB)) == []
    assert list(profile_home.glob(_PROBE_GLOB)) == []


def test_probe_answers_for_file_identity_not_path_spelling(tmp_path, monkeypatch):
    """Same absolute paths on another machine are not the same file: the remote's own tree keeps
    its sync manager. The two halves differ only in which filesystem the remote shell reads."""
    host_home, sync_root, _profile_home = _host_tree(tmp_path, monkeypatch, "nested")

    other_machine = tmp_path / "other-machine"
    remote_copy = other_machine / sync_root.relative_to("/")
    (remote_copy / "skills").mkdir(parents=True)
    (remote_copy / "skills" / "SKILL.md").write_text("host skills", encoding="utf-8")

    remote_env, remote_made = _build_env(monkeypatch, host_home, _remote_shell(other_machine))
    assert remote_env._sync_manager is not None
    assert len(remote_made) == 1
    remote_env.cleanup()
    assert list(sync_root.glob(_PROBE_GLOB)) == []

    local_env, local_made = _build_env(monkeypatch, host_home, _remote_shell(None))
    assert local_env._sync_manager is None
    assert local_made == []
