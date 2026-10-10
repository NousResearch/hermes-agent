"""One identity per sandbox container (t_44fdd37a): the reuse fingerprint names the container, so
reuse, the cold-start race, recovery and config drift all resolve through ``docker run --name`` and
duplicates cannot arise; the one refusal left is another identity holding a real host path RW."""
import json
import os
import subprocess
import sys
import threading
import uuid
from pathlib import Path

import pytest

from tools.environments import docker as docker_env

_FINISHED_LONG_AGO = "2000-01-01T00:00:00.000000000Z"


@pytest.fixture(autouse=True)
def _stable_tempdir(monkeypatch, tmp_path):
    """pytest's tmp_path sits under the system tempdir, whose mounts count as volatile: that would
    hide every jail path and sandbox dir from the fingerprint and the refusal probe."""
    (tmp_path / "proc-tmp").mkdir()
    monkeypatch.setattr(docker_env.tempfile, "gettempdir", lambda: str(tmp_path / "proc-tmp"))


class _FakeDaemon:
    """In-memory Docker daemon: container names are unique, ``ps`` honours label/status filters,
    and run/rename/start/stop/rm/exec behave as the real CLI does for the calls hermes makes."""

    def __init__(self, monkeypatch, tmp_path):
        self.containers: dict[str, dict] = {}
        self.create_window: float = 0.0  # name reserved, container not yet inspectable (daemon create)
        self.reserved: dict[str, str] = {}
        self.timers: list[threading.Timer] = []
        self.calls: list[list[str]] = []
        self.run_barrier: threading.Barrier | None = None
        self.run_hook = None  # cmd -> (rc, out, err) answering a ``docker run`` in place of the daemon
        self.start_error = ""
        self._lock = threading.Lock()
        monkeypatch.setenv("TERMINAL_SANDBOX_DIR", str(tmp_path / "sandboxes"))
        monkeypatch.setattr(docker_env.DockerEnvironment, "init_session", lambda self: None)
        monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
        monkeypatch.setattr(docker_env, "_get_active_profile_name", lambda: "bot_1")
        monkeypatch.setattr(docker_env, "_readonly_skill_mount_args", lambda: [])
        monkeypatch.setattr(docker_env, "_cgroup_limits_ok", True)
        monkeypatch.setattr(docker_env.subprocess, "run", self)
        monkeypatch.setattr(docker_env, "_popen_bash", self._exec)

    def add(self, name, labels, *, binds=(), image="python:3.11", state="running", network="bridge") -> str:
        cid = uuid.uuid4().hex + uuid.uuid4().hex
        self.containers[cid] = {
            "name": name, "labels": dict(labels), "state": state, "image": image, "network": network,
            "finished": "0001-01-01T00:00:00Z",
            "mounts": [{"Type": "bind" if src.startswith("/") else "volume", "Source": src,
                        "Destination": dst, "RW": rw} for src, dst, rw in binds]}
        return cid

    def created_by(self, cmd, **labels) -> str:
        """A container the ``docker run`` *cmd* created but never started, *labels* overriding its own."""
        return self.add(_flag_value(cmd, "--name"), {**_run_labels(cmd), **labels}, state="created")

    def find(self, ref):
        return next((cid for cid, c in self.containers.items() if ref in (c["name"], cid[:len(ref)])), None)

    def stop(self, cid):
        self.containers[cid].update(state="exited", finished=_FINISHED_LONG_AGO)

    def _create(self, name, cid, container):
        with self._lock:
            if self.reserved.pop(name, None) is not None:  # not already created by its timer
                self.containers[cid] = container

    def settle(self):
        """End every pending create window now."""
        for timer in self.timers:
            timer.cancel()
            self._create(*timer.args)

    def subcommands(self, *subs):
        return [c for c in self.calls if c[1] in subs]

    def __call__(self, cmd, **kwargs):
        cmd = list(cmd)
        if cmd[1] == "run" and self.run_barrier is not None:
            self.run_barrier.wait(timeout=10)  # every racer has probed before anyone runs
        with self._lock:
            self.calls.append(cmd)
            rc, out, err = self._answer(cmd)
        if kwargs.get("check") and rc:
            raise subprocess.CalledProcessError(rc, cmd, out, err)
        return subprocess.CompletedProcess(cmd, rc, stdout=out, stderr=err)

    def _answer(self, cmd):
        sub = cmd[1]
        ref = cmd[2] if sub == "rename" else cmd[-1]
        if sub in ("version", "image"):
            return 0, "null", ""
        if sub == "ps":
            labels = dict(f.removeprefix("label=").split("=", 1) for f in _flag_values(cmd, "--filter")
                          if f.startswith("label="))
            states = {f.removeprefix("status=") for f in _flag_values(cmd, "--filter") if f.startswith("status=")}
            hits = [(cid, c) for cid, c in self.containers.items()
                    if ("-a" in cmd or c["state"] == "running") and (not states or c["state"] in states)
                    and labels.items() <= c["labels"].items()]
            with_state = "{{.State}}" in cmd[cmd.index("--format") + 1]
            return 0, "".join(f"{cid}\t{c['state']}\n" if with_state else f"{cid}\n" for cid, c in hits), ""
        if sub == "run":
            if self.run_hook is not None:
                return self.run_hook(cmd)
            name = _flag_value(cmd, "--name")
            if (holder := self.find(name) or self.reserved.get(name)) is not None:
                return _name_conflict(name, holder)
            specs = [spec.split(":") for spec in _flag_values(cmd, "-v")]
            binds = [(spec[0], spec[1], spec[2:] != ["ro"]) for spec in specs]
            cid = self.add(name, _run_labels(cmd), binds=binds, image=cmd[-3],
                           network="none" if "--network=none" in cmd else "bridge")
            if self.create_window > 0:
                self.reserved[name] = cid
                timer = threading.Timer(self.create_window, self._create, (name, cid, self.containers.pop(cid)))
                self.timers.append(timer)
                timer.start()
            return 0, cid + "\n", ""
        if sub == "inspect" and cmd[2:4] == ["--type", "container"]:  # one JSON line per container found
            found = [cid for cid in map(self.find, cmd[cmd.index("--format") + 2:]) if cid is not None]
            out = "".join(json.dumps({"id": cid, "name": "/" + self.containers[cid]["name"],
                                      **{k: self.containers[cid][k] for k in ("state", "labels", "mounts")}})
                          + "\n" for cid in found)
            return (0 if len(found) == len(cmd) - cmd.index("--format") - 2 else 1), out, ""
        cid = self.find(ref)
        if cid is None:
            return 1, "", f"Error response from daemon: No such container: {ref}"
        c = self.containers[cid]
        if sub == "inspect":
            fmt = cmd[cmd.index("--format") + 1]
            if fmt == "{{.State.FinishedAt}}":
                return 0, c["finished"] + "\n", ""
            if fmt == "{{.HostConfig.NetworkMode}}":
                return 0, c["network"] + "\n", ""
            assert fmt == "{{.Config.Image}}", fmt
            return 0, c["image"] + "\n", ""
        if sub == "rename":
            if self.find(cmd[3]) is not None:
                return 1, "", f'Error response from daemon: Conflict. The container name "/{cmd[3]}" is already in use'
            c["name"] = cmd[3]
        elif sub == "start":
            if self.start_error:
                return 1, "", self.start_error
            c["state"] = "running"
        elif sub == "rm":
            if c["state"] == "running" and "-f" not in cmd:
                return 1, "", "Error response from daemon: cannot remove container: container is running"
            del self.containers[cid]
        return 0, "", ""

    def _exec(self, cmd, stdin_data=None, **kwargs):
        self.calls.append(list(cmd))
        cid = self.find(cmd[cmd.index("bash") - 1])
        alive = cid is not None and self.containers[cid]["state"] == "running"
        script = "print('alive')" if alive else (
            "print('Error response from daemon: No such container')\nraise SystemExit(1)")
        return _real_popen_bash([sys.executable, "-c", script], stdin_data)


_real_popen_bash = docker_env._popen_bash


def _flag_values(cmd, flag):
    return [cmd[i + 1] for i, arg in enumerate(cmd) if arg == flag]


def _flag_value(cmd, flag):
    return cmd[cmd.index(flag) + 1]


def _run_labels(cmd):
    return dict(label.split("=", 1) for label in _flag_values(cmd, "--label"))


def _name_conflict(name, holder):
    return 125, "", (f'docker: Error response from daemon: Conflict. The container name "/{name}" '
                     f'is already in use by container "{holder}".')


@pytest.fixture
def daemon(monkeypatch, tmp_path):
    return _FakeDaemon(monkeypatch, tmp_path)


def _spawn(**kwargs):
    kwargs = {"image": "python:3.11", "task_id": "default", "persistent_filesystem": True, "volumes": [], **kwargs}
    return docker_env.DockerEnvironment(**kwargs)


def _race_two_spawns():
    """``(envs, errors)`` of two concurrent spawns; errors are surfaced by the caller's assertions."""
    envs, errors = [], []

    def spawn():
        try:
            envs.append(_spawn())
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=spawn) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    return envs, errors


def _jail(tmp_path):
    path = tmp_path / "jail"
    path.mkdir()
    return str(path)


def _foreign_labels(**overrides):
    return {"hermes-agent": "1", "hermes-profile": "bot_1", "hermes-task-id": "default",
            "hermes-egress": "off", "hermes-environment": "f" * 24, **overrides}


# --- identity -------------------------------------------------------------------------------

_BASE_CONFIG = {"image": "python:3.11", "mount_args": ["-v", "/srv/jail:/home/bot"],
                "hermes_home": "/profiles/alpha", "egress": "off"}
_CHANGED_CONFIG = {"image": "python:3.12", "mount_args": ["-v", "/srv/jail:/home/bot", "-v", "data:/data"],
                   "hermes_home": "/profiles/beta", "egress": "0123456789abcdef01234567"}


@pytest.mark.parametrize("field", sorted(_CHANGED_CONFIG))
def test_fingerprint_is_the_single_identity_source(field):
    """Every jail-relevant input moves both the fingerprint and the name derived from it; an equal
    configuration lands on one name."""
    fingerprint = docker_env._reuse_environment_fingerprint(**_BASE_CONFIG)
    changed = docker_env._reuse_environment_fingerprint(**{**_BASE_CONFIG, field: _CHANGED_CONFIG[field]})

    assert fingerprint == docker_env._reuse_environment_fingerprint(**dict(_BASE_CONFIG))
    assert changed != fingerprint
    assert docker_env._canonical_container_name(changed) != docker_env._canonical_container_name(fingerprint)


def test_fingerprint_name_is_stable_across_processes():
    """Two processes of one configuration must derive the same name: no per-process entropy
    (hash seed, dict order) may reach the hash."""
    code = ("from tools.environments import docker as d; import json, sys; "
            "print(d._canonical_container_name(d._reuse_environment_fingerprint(**json.loads(sys.argv[1]))))")
    names = {subprocess.run([sys.executable, "-c", code, json.dumps(_BASE_CONFIG)], capture_output=True, text=True,
                            check=True, cwd=Path(__file__).parents[2], env={**os.environ, "PYTHONHASHSEED": seed},
                            stdin=subprocess.DEVNULL).stdout.strip() for seed in ("1", "2")}

    assert names == {docker_env._canonical_container_name(docker_env._reuse_environment_fingerprint(**_BASE_CONFIG))}


def test_volatile_tempdir_mount_source_keeps_the_name(tmp_path):
    """The symlink-safe skills copy is a fresh mkdtemp per process: its source must not move the
    name, or every process would spawn its own container. Where it lands still does."""
    proc_tmp = tmp_path / "proc-tmp"

    def name(source, dest="/root/.hermes/skills"):
        return docker_env._canonical_container_name(docker_env._reuse_environment_fingerprint(
            **{**_BASE_CONFIG, "mount_args": ["-v", f"{source}:{dest}:ro"]}))

    assert name(proc_tmp / "hermes-skills-safe-a1b2") == name(proc_tmp / "hermes-skills-safe-c3d4")
    assert name(proc_tmp / "hermes-skills-safe-a1b2") != name(proc_tmp / "hermes-skills-safe-a1b2", "/skills")


def test_tmpfs_sandboxes_of_distinct_tasks_get_distinct_names(daemon):
    """A tmpfs sandbox has no host path carrying its task bucket, so the bucket is hashed in —
    injectively: ``a:b`` and ``a_b`` share a label value but must not share a container."""
    one = _spawn(task_id="rollout:one", persistent_filesystem=False)
    two = _spawn(task_id="rollout_one", persistent_filesystem=False)

    assert one._name != two._name and one._container_id != two._container_id
    assert _spawn(task_id="rollout:one", persistent_filesystem=False)._container_id == one._container_id


def test_egress_posture_reaches_the_name(daemon, monkeypatch):
    """The constructor feeds the egress posture into the fingerprint: a container with baked-in proxy
    env and CA mounts must not be attached once egress is turned off, or the reverse."""
    names = set()
    for posture in ("off", "0123456789abcdef01234567"):
        monkeypatch.setattr(docker_env, "_egress_reuse_fingerprint", lambda *_args, posture=posture: posture)
        names.add(_spawn()._name)

    assert len(names) == 2 and len(daemon.containers) == 2


@pytest.mark.parametrize("persistent", [True, False], ids=["bind", "tmpfs"])
def test_shared_key_keeps_task_buckets_apart(daemon, persistent):
    """A shared key shares a profile's container across profiles, not across task buckets: a per-task
    rollout must not attach to (or, with a pinned image, replace) another task's container."""
    one = _spawn(task_id="rollout:one", shared_container_key="team", persistent_filesystem=persistent)
    two = _spawn(task_id="rollout:two", shared_container_key="team", persistent_filesystem=persistent,
                 image="python:3.12", image_pinned=True)

    assert one._labels["hermes-environment"] != two._labels["hermes-environment"]
    assert one._name != two._name and one._container_id != two._container_id
    assert len(daemon.containers) == 2 and not daemon.subcommands("rm")


def test_session_scoped_container_takes_no_shared_identity(daemon):
    """Without cross-process persistence the container is the session's alone: a unique name, no
    lookup, and no probe of other containers."""
    first = _spawn(persist_across_processes=False)
    second = _spawn(persist_across_processes=False)

    assert first._name != second._name
    assert first._labels["hermes-environment"] == second._labels["hermes-environment"]
    assert not daemon.subcommands("inspect", "ps")


# --- spawn ----------------------------------------------------------------------------------

def test_spawn_runs_under_the_canonical_name_with_every_label(daemon):
    env = _spawn()

    container = daemon.containers[daemon.find(env._name)]
    assert env._name == docker_env._canonical_container_name(env._labels["hermes-environment"])
    assert container["labels"] == {**env._labels, "hermes-spawn": container["labels"]["hermes-spawn"]}
    assert container["labels"].keys() == {
        "hermes-agent", "hermes-task-id", "hermes-profile", "hermes-egress", "hermes-environment", "hermes-spawn"}


def test_respawn_attaches_by_name_and_starts_a_stopped_container(daemon):
    first = _spawn()
    daemon.stop(first._container_id)

    second = _spawn()

    assert second._container_id == first._container_id
    assert daemon.containers[first._container_id]["state"] == "running"
    assert len(daemon.subcommands("run")) == 1


def test_cold_start_race_converges_on_one_container(daemon):
    """Two processes of one configuration probe before either runs: one ``docker run`` wins the
    name, the loser's fails "already in use" and it attaches to the winner's container."""
    daemon.run_barrier = threading.Barrier(2)
    envs, errors = _race_two_spawns()

    assert not errors
    assert len(daemon.containers) == 1
    assert {env._container_id for env in envs} == set(daemon.containers)
    assert len(daemon.subcommands("run")) == 2


def test_cold_start_race_attaches_through_the_create_window(daemon):
    """The daemon reserves the name before the winner's container is inspectable: the loser waits
    for it and attaches, and never removes by name — a plain ``rm`` would take the winner's
    still-created container (live scenario H, docker 29.1.3)."""
    daemon.run_barrier = threading.Barrier(2)
    daemon.create_window = 0.75
    envs, errors = _race_two_spawns()
    daemon.settle()

    assert not errors
    assert len(daemon.containers) == 1
    assert {env._container_id for env in envs} == set(daemon.containers)
    assert len(daemon.subcommands("run")) == 2
    assert not daemon.subcommands("rm")


def test_conflict_holder_that_never_appears_raises_without_removing(daemon, monkeypatch):
    """A name holder that stays uninspectable past the wait is a loud failure, never a removal."""
    monkeypatch.setattr(docker_env, "_CONFLICT_ATTACH_TIMEOUT", 0.5)
    daemon.create_window = 3.0
    winner = _spawn()

    with pytest.raises(RuntimeError, match=f"{winner._name}.*never became inspectable"):
        _spawn()
    assert not daemon.subcommands("rm")

    daemon.settle()
    assert daemon.containers[winner._container_id]["state"] == "running"


def test_failed_run_spares_a_siblings_created_container(daemon):
    """A run that failed before its create (pull error) has no container: the name holder is a
    sibling's still-Created one of the same identity but another spawn nonce, which the old cleanup
    by name removed."""
    def pull_fails_while_a_sibling_creates(cmd):
        daemon.created_by(cmd, **{"hermes-spawn": uuid.uuid4().hex})
        return 125, "", "docker: Error response from daemon: manifest unknown"

    daemon.run_hook = pull_fails_while_a_sibling_creates
    with pytest.raises(subprocess.CalledProcessError):
        _spawn()
    (sibling,) = daemon.containers.values()
    assert sibling["state"] == "created"
    assert not daemon.subcommands("rm")


@pytest.mark.parametrize("timed_out", [False, True], ids=["start-failed", "client-timed-out"])
def test_failed_run_removes_its_own_created_container(daemon, timed_out):
    """By its spawn nonce, whether the create returned or the client died before seeing it (timeout)."""
    ours = []

    def run_fails_after_create(cmd):
        ours.append(daemon.created_by(cmd))
        if timed_out:
            raise subprocess.TimeoutExpired(cmd, 120)
        return 125, "", "docker: Error response from daemon: failed to create task: mount source missing"

    daemon.run_hook = run_fails_after_create
    with pytest.raises((subprocess.CalledProcessError, subprocess.TimeoutExpired)):
        _spawn()
    assert not daemon.containers
    assert [c[2:] for c in daemon.subcommands("rm")] == [ours]


def test_spawn_nonce_labels_only_the_run(daemon):
    """Each ``docker run`` carries its own nonce label and no host cidfile (a snap client's private
    /tmp cannot see one); the nonce stays out of identity, so the canonical name is unchanged."""
    first = _spawn()
    daemon.containers.clear()
    second = _spawn()

    runs = daemon.subcommands("run")
    assert all("--cidfile" not in cmd for cmd in runs)
    nonces = [[v for v in _flag_values(cmd, "--label") if v.startswith("hermes-spawn=")] for cmd in runs]
    assert all(len(n) == 1 for n in nonces) and nonces[0] != nonces[1]
    assert first._name == second._name and "hermes-spawn" not in second._labels


@pytest.mark.parametrize("sibling, ours, reason", [
    ({"image": "python:3.10"}, {"image_pinned": True}, "runs image python:3.10 but docker_image is python:3.11"),
    ({"network": "bridge"}, {"network": False}, "has NetworkMode=bridge but docker_network=false"),
], ids=["pinned-image", "air-gap"])
def test_conflict_leg_never_removes_the_siblings_container(daemon, sibling, ours, reason):
    """The recreate guards remove a stale holder on attach, but a holder that wins our own ``docker
    run`` is a sibling's live container: the loser names the mismatch and leaves it running."""
    def sibling_wins(cmd):
        name = _flag_value(cmd, "--name")
        return _name_conflict(name, daemon.add(name, _run_labels(cmd), **sibling))

    daemon.run_hook = sibling_wins
    with pytest.raises(RuntimeError, match="sibling process won the name race") as raised:
        _spawn(**ours)
    (won,) = daemon.containers
    assert f"{won[:12]} {reason}" in str(raised.value)
    assert daemon.containers[won]["state"] == "running"
    assert not daemon.subcommands("rm")


def test_session_scoped_name_conflict_does_not_blame_a_sibling(daemon):
    daemon.run_hook = lambda cmd: _name_conflict(_flag_value(cmd, "--name"), "f" * 64)

    with pytest.raises(RuntimeError, match="random name is already held by another container") as raised:
        _spawn(persist_across_processes=False)
    assert "sibling" not in str(raised.value)
    assert not daemon.subcommands("rm")


@pytest.mark.parametrize("shared_key", ["", "team"], ids=["labeled-otherwise", "label-less-of-another-task"])
def test_name_holder_of_another_identity_refuses(daemon, shared_key):
    """The name is derived from the fingerprint, so a holder labeled otherwise is not ours — nor,
    under a shared key, a label-less one whose other labels are not the legacy lookup's."""
    probe = _spawn(shared_container_key=shared_key)
    labels = _foreign_labels() if not shared_key else {
        **{k: v for k, v in probe._labels.items() if k != "hermes-environment"}, "hermes-task-id": "other"}
    daemon.containers.clear()
    daemon.add(probe._name, labels)
    daemon.calls.clear()

    with pytest.raises(RuntimeError, match=f"Refusing to use container {probe._name}"):
        _spawn(shared_container_key=shared_key)
    assert not daemon.subcommands("run", "rename", "rm")


def test_failed_start_of_the_named_container_raises(daemon):
    """The name is taken, so there is no fresh container to fall back to: fail loudly."""
    first = _spawn()
    daemon.stop(first._container_id)
    daemon.start_error = "Error response from daemon: mount source missing"

    with pytest.raises(RuntimeError, match=f"{first._name}.*mount source missing"):
        _spawn()


# --- drift ----------------------------------------------------------------------------------

def test_drift_runs_a_new_name_and_leaves_the_stale_container_to_the_reaper(daemon):
    """A named volume added to the config is a new fingerprint, so a new name: the spawn never
    removes or adopts the stale container, which the orphan reaper takes once it has exited."""
    old = _spawn()
    new = _spawn(volumes=["gvol:/data"])

    assert new._name != old._name and new._container_id != old._container_id
    assert daemon.containers[old._container_id]["state"] == "running"
    assert not daemon.subcommands("rm", "stop")

    daemon.stop(old._container_id)
    removed = docker_env.reap_orphan_containers(max_age_seconds=60, profile_filter="bot_1")
    assert removed == 1 and set(daemon.containers) == {new._container_id}


@pytest.mark.parametrize("first, second, replaced", [
    ({"shared_container_key": "team", "image": "python:3.10"},
     {"shared_container_key": "team", "image_pinned": True}, ("image", "python:3.11")),
    ({}, {"network": False}, ("network", "none")),
], ids=["pinned-image", "air-gap"])
def test_name_holder_failing_a_recreate_guard_is_replaced_under_the_name(daemon, first, second, replaced):
    """The approved exceptions to "no replace", reached through the canonical-name holder: the image is
    no fingerprint input under a shared key and the network mode never is, so the holder of our name
    can be stale on either. It is removed and a fresh container runs under the same name."""
    old = _spawn(**first)
    new = _spawn(**second)

    assert new._name == old._name and new._container_id != old._container_id
    assert old._container_id not in daemon.containers
    assert daemon.containers[new._container_id][replaced[0]] == replaced[1]
    assert [c[2:] for c in daemon.subcommands("rm")] == [["-f", old._container_id]]


# --- refusal --------------------------------------------------------------------------------

@pytest.mark.parametrize("holder_labels", [
    {"hermes-task-id": "profile_forge"},  # another task bucket of the profile: the live leak
    {"hermes-profile": "bot_2"},  # another profile
    {},  # the same bucket under drifted configuration
], ids=["other-bucket", "other-profile", "drifted"])
def test_another_identity_holding_a_real_rw_path_refuses(daemon, tmp_path, holder_labels):
    jail = _jail(tmp_path)
    labels = _foreign_labels(**holder_labels)
    holder = daemon.add("hermes-holder", labels, binds=[(jail, "/home/bot", True)])

    with pytest.raises(RuntimeError, match=f"{jail}.*{holder[:12]}") as refused:
        _spawn(volumes=[f"{jail}:/home/bot"])
    assert f"task={labels['hermes-task-id']!r}, profile={labels['hermes-profile']!r}" in str(refused.value)
    assert not daemon.subcommands("run")


def test_same_fingerprint_under_another_name_holding_a_real_rw_path_refuses(daemon, tmp_path):
    """Only the holder of our name is ours to attach: a same-identity container left under another
    name (a failed legacy rename) is a second writer of the jail like any other."""
    jail = _jail(tmp_path)
    probe = _spawn(volumes=[f"{jail}:/home/bot"])
    daemon.containers.clear()
    twin = daemon.add("hermes-1a2b3c4d", probe._labels, binds=[(jail, "/home/bot", True)])
    daemon.calls.clear()

    with pytest.raises(RuntimeError, match=f"{jail}.*{twin[:12]}"):
        _spawn(volumes=[f"{jail}:/home/bot"])
    assert not daemon.subcommands("run")


def test_label_less_rw_holder_refuses_under_a_shared_key(daemon, tmp_path):
    """Under a shared key there is no legacy fingerprint, so a label-less container (any other key's
    pre-canonical one) matches ours on that label alone: holding our RW jail path under another
    name, it still refuses — the probe spares only our name holder."""
    jail = _jail(tmp_path)
    labels = _foreign_labels(**{"hermes-profile": "other_key-0123456789ab"})
    del labels["hermes-environment"]
    holder = daemon.add("hermes-holder", labels, binds=[(jail, "/home/bot", True)])

    with pytest.raises(RuntimeError, match=f"{jail}.*{holder[:12]}"):
        _spawn(volumes=[f"{jail}:/home/bot"], shared_container_key="team")


@pytest.mark.parametrize("holder_rw, ours", [(False, ":/home/bot"), (True, ":/home/bot:ro")], ids=["theirs-ro", "ours-ro"])
def test_read_only_side_never_conflicts(daemon, tmp_path, holder_rw, ours):
    jail = _jail(tmp_path)
    daemon.add("hermes-holder", _foreign_labels(), binds=[(jail, "/home/bot", holder_rw)])

    _spawn(volumes=[jail + ours])

    assert len(daemon.subcommands("run")) == 1


def test_stopped_container_start_is_refused_while_another_identity_holds_its_path(daemon, tmp_path):
    """Daemon-restart ordering: a forge spawn brought the jail path up first; restarting the
    default jail beside it would double-mount the path."""
    jail = _jail(tmp_path)
    ours = _spawn(volumes=[f"{jail}:/home/bot"])
    daemon.stop(ours._container_id)
    daemon.add("hermes-forge", _foreign_labels(**{"hermes-task-id": "profile_forge"}), binds=[(jail, "/home/bot", True)])

    with pytest.raises(RuntimeError, match="already bind-mounted read-write"):
        _spawn(volumes=[f"{jail}:/home/bot"])
    assert daemon.containers[ours._container_id]["state"] == "exited"


# --- recovery -------------------------------------------------------------------------------

def test_recovery_starts_the_named_container(daemon):
    env = _spawn()
    daemon.stop(env._container_id)

    result = env.execute("echo alive")

    assert result["returncode"] == 0 and "alive" in result["output"]
    assert len(daemon.subcommands("run")) == 1


def test_recovery_runs_under_the_canonical_name_when_the_container_is_gone(daemon):
    env = _spawn()
    del daemon.containers[env._container_id]

    result = env.execute("echo alive")

    assert result["returncode"] == 0
    assert daemon.find(env._name) == env._container_id
    assert [_flag_values(c, "--name") for c in daemon.subcommands("run")] == [[env._name], [env._name]]


def test_jail_guard_refused_recovery_retries_on_next_exec(daemon, tmp_path):
    """A refused recovery keeps the gone id: the next exec fails as container-gone and recovers once
    the conflicting holder is gone (a ``None`` id would assert "Container not started" forever)."""
    jail = _jail(tmp_path)
    env = _spawn(volumes=[f"{jail}:/home/bot"])
    gone = env._container_id
    del daemon.containers[gone]
    holder = daemon.add("hermes-holder", _foreign_labels(), binds=[(jail, "/home/bot", True)])

    refused = env.execute("echo alive")
    assert refused["returncode"] != 0 and env._container_id == gone

    del daemon.containers[holder]
    recovered = env.execute("echo alive")
    assert recovered["returncode"] == 0 and "alive" in recovered["output"]
    assert env._container_id == daemon.find(env._name)


# --- legacy containers ----------------------------------------------------------------------

def test_legacy_container_is_adopted_by_label_and_renamed(daemon):
    """A container from before canonical names (random name, fingerprint label without the egress
    input) is found by the labels its creator used and renamed to the canonical name; every label
    stays, so label-keyed tooling still resolves it."""
    probe = _spawn()
    labels = {**probe._labels, "hermes-environment": probe._legacy_fingerprint}
    daemon.containers.clear()
    legacy = daemon.add("hermes-1a2b3c4d", labels)
    daemon.calls.clear()

    env = _spawn()

    assert env._container_id == legacy
    assert daemon.containers[legacy]["name"] == env._name
    assert daemon.containers[legacy]["labels"] == labels
    assert not daemon.subcommands("run")
    by_label = daemon(["docker", "ps", "-a", "--filter", "label=hermes-profile=bot_1",
                       "--filter", "label=hermes-task-id=default", "--format", "{{.ID}}"])
    assert by_label.stdout.split() == [legacy]
    # The renamed holder carries the legacy label: the next process attaches by name, no refusal.
    assert _spawn()._container_id == legacy


def test_legacy_shared_key_container_is_adopted_and_stays_ours(daemon):
    """Pre-canonical shared-key containers carry no fingerprint label: after the rename the holder of
    our name is still recognised by its other labels, so the next process attaches, not refuses."""
    probe = _spawn(shared_container_key="team")
    labels = {k: v for k, v in probe._labels.items() if k != "hermes-environment"}
    daemon.containers.clear()
    legacy = daemon.add("hermes-1a2b3c4d", labels)

    assert _spawn(shared_container_key="team")._container_id == legacy
    assert daemon.containers[legacy]["name"] == probe._name
    assert _spawn(shared_container_key="team")._container_id == legacy
    assert len(daemon.subcommands("run")) == 1  # the probe's own


def test_shared_key_legacy_lookup_skips_another_tasks_canonical_container(daemon):
    """Under a shared key the legacy lookup has no fingerprint filter, and ``a:b`` and ``a_b`` share a
    sanitized task label: the other task's canonical container matches it but carries a fingerprint,
    so it is never renamed or attached — each task gets its own container."""
    assert docker_env._sanitize_label_value("a:b") == docker_env._sanitize_label_value("a_b")
    first = _spawn(task_id="a:b", shared_container_key="team")

    second = _spawn(task_id="a_b", shared_container_key="team")

    assert second._name != first._name and second._container_id != first._container_id
    assert daemon.containers[first._container_id]["name"] == first._name
    assert not daemon.subcommands("rename")
    assert len(daemon.subcommands("run")) == 2


# --- mount parsing --------------------------------------------------------------------------

def test_parse_bind_args():
    binds = docker_env._parse_bind_args([
        "-v", "/a:/x", "-v", "/b:/y:ro,z", "--mount", "type=bind,source=/c,destination=/z,readonly",
        "--mount", "type=bind,src=/d,dst=/w", "--mount", "type=volume,source=named,target=/v",
        "--tmpfs", "/tmp:rw,nosuid,size=512m"])

    assert binds == [("/a", "/x", True), ("/b", "/y", False), ("/c", "/z", False), ("/d", "/w", True)]


def test_parse_bind_args_joined_flags_and_named_volumes():
    parse = docker_env._parse_bind_args

    assert parse(["--volume=/a:/x", "--mount=type=bind,source=/b,target=/y,ro"]) == parse(
        ["--volume", "/a:/x", "--mount", "type=bind,source=/b,target=/y,ro"])
    assert parse(["-v/a:/x"]) == parse(["-v=/a:/x"]) == parse(["-v", "/a:/x"]) == [("/a", "/x", True)]
    assert parse(["-v", "named:/x", "-v", "./rel:/y"]) == [("./rel", "/y", True)]


@pytest.mark.require_symlinks
def test_shared_rw_sources_drop_sandbox_tempdir_and_read_only(tmp_path):
    root = tmp_path / "sandboxes" / "docker"
    (root / "default" / "home").mkdir(parents=True)
    jail = Path(_jail(tmp_path))
    (tmp_path / "jail-link").symlink_to(jail)
    binds = [(str(root / "default" / "home"), "/root", True), (str(tmp_path / "proc-tmp" / "skills"), "/s", True),
             (str(tmp_path / "jail-link"), "/home/bot", True), ("/srv/ro", "/ro", False)]

    assert docker_env._shared_rw_sources(binds, os.path.realpath(root)) == {os.path.realpath(jail)}
