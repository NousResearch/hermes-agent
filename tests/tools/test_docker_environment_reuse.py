"""Docker runtime identity and cross-process reuse tests."""

import errno
import logging
import os
import secrets
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from tools.environments import docker as docker_env


def _mock_subprocess_run(monkeypatch):
    """Mock subprocess.run to intercept docker run -d and docker version calls.

    Returns a list of captured (cmd, kwargs) tuples for inspection.

    Pre-seeds the cgroup-limit probe cache to ``True`` so the throwaway probe
    container (a ``docker run ... sleep 0``) does not run and pollute the
    captured call list — these tests inspect the real sandbox-start ``run``.
    Tests that exercise the probe itself live in test_docker_cgroup_limits.py.
    """
    monkeypatch.setattr(docker_env, "_cgroup_limits_ok", True)
    calls = []

    def _run(cmd, **kwargs):
        calls.append((list(cmd) if isinstance(cmd, list) else cmd, kwargs))
        if isinstance(cmd, list) and len(cmd) >= 2:
            if cmd[1] == "version":
                return subprocess.CompletedProcess(cmd, 0, stdout="Docker version", stderr="")
            if cmd[1] == "run":
                return subprocess.CompletedProcess(cmd, 0, stdout="fake-container-id\n", stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(docker_env.subprocess, "run", _run)
    return calls


def _make_dummy_env(**kwargs):
    """Helper to construct DockerEnvironment with minimal required args."""
    return docker_env.DockerEnvironment(
        image=kwargs.get("image", "python:3.11"),
        cwd=kwargs.get("cwd", "/root"),
        timeout=kwargs.get("timeout", 60),
        cpu=kwargs.get("cpu", 0),
        memory=kwargs.get("memory", 0),
        disk=kwargs.get("disk", 0),
        persistent_filesystem=kwargs.get("persistent_filesystem", False),
        task_id=kwargs.get("task_id", "test-task"),
        volumes=kwargs.get("volumes", []),
        forward_env=kwargs.get("forward_env"),
        network=kwargs.get("network", True),
        host_cwd=kwargs.get("host_cwd"),
        auto_mount_cwd=kwargs.get("auto_mount_cwd", False),
        env=kwargs.get("env"),
        run_as_host_user=kwargs.get("run_as_host_user", False),
        extra_args=kwargs.get("extra_args", []),
        persist_across_processes=kwargs.get("persist_across_processes", True),
        shared_container_key=kwargs.get("shared_container_key", ""),
        shm_size=kwargs.get("shm_size", docker_env._DEFAULT_SHM_SIZE),
        snap_compat=kwargs.get("snap_compat", False),
    )


def _run_args_from_calls(calls):
    """Pull the argv list passed to the first ``docker run`` invocation."""
    run_calls = [
        c for c in calls
        if isinstance(c[0], list) and len(c[0]) >= 2 and c[0][1] == "run"
    ]
    assert run_calls, "docker run should have been called"
    return run_calls[0][0]


def _labels_in_run_args(run_args):
    """Return the set of ``key=value`` strings passed via ``--label``."""
    return {
        run_args[i + 1]
        for i, flag in enumerate(run_args[:-1])
        if flag == "--label"
    }


def _mock_subprocess_run_with_reuse(
    monkeypatch,
    ps_state: str | None,
    start_succeeds: bool = True,
    expected_runtime_label: str | None = None,
    entrypoint_json: str | None = None,
    container_image: str | None = None,
):
    """Reuse-aware subprocess.run mock.

    ``ps_state`` controls what ``docker ps -a --filter ...`` returns:
      * ``None`` → no match (empty stdout). Forces a fresh ``docker run``.
      * ``"running"`` / ``"exited"`` / ... → emit ``CID\\tSTATE`` so the reuse
        path picks it up. ``"running"`` skips ``docker start``; other states
        trigger ``docker start`` (which can be forced to fail via
        ``start_succeeds=False``).

    Returns the captured call list so the test can verify which docker
    commands actually ran.
    """
    monkeypatch.setattr(docker_env, "_cgroup_limits_ok", True)
    calls = []

    def _run(cmd, **kwargs):
        calls.append((list(cmd) if isinstance(cmd, list) else cmd, kwargs))
        if isinstance(cmd, list) and len(cmd) >= 2:
            sub = cmd[1]
            if sub == "version":
                return subprocess.CompletedProcess(cmd, 0, stdout="Docker version", stderr="")
            if cmd[1:3] == ["image", "inspect"] and entrypoint_json is not None:
                return subprocess.CompletedProcess(
                    cmd, 0, stdout=f"{entrypoint_json}\n", stderr=""
                )
            if sub == "inspect" and container_image is not None:
                return subprocess.CompletedProcess(
                    cmd, 0, stdout=f"{container_image}\n", stderr=""
                )
            if sub == "ps":
                if ps_state is None:
                    return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
                if (
                    expected_runtime_label is not None
                    and f"label=hermes-runtime={expected_runtime_label}" not in cmd
                ):
                    return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
                # 2-field format: ID, State. The egress and runtime posture are
                # enforced by exact label filters on the ps command itself.
                return subprocess.CompletedProcess(
                    cmd, 0,
                    stdout=f"reused-cid\t{ps_state}\n",
                    stderr="",
                )
            if sub == "start":
                if not start_succeeds:
                    # Real subprocess.run with check=True raises on non-zero exit;
                    # mirror that so the production code's except clause fires.
                    raise subprocess.CalledProcessError(1, cmd, output="", stderr="no such container")
                return subprocess.CompletedProcess(cmd, 0, stdout="reused-cid\n", stderr="")
            if sub == "run":
                return subprocess.CompletedProcess(cmd, 0, stdout="fresh-cid\n", stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(docker_env.subprocess, "run", _run)
    return calls


def test_runtime_reuse_fingerprint_ignores_volatile_temp_mounts(tmp_path, monkeypatch):
    """A mount whose host source is a per-process tempdir must not churn the
    reuse label: symlinked skills trees are served from a fresh mkdtemp copy
    every process (``_safe_skills_path``), so hashing that path made the
    label differ across processes and container reuse never matched."""
    temp_root = tmp_path / "tmp"
    temp_root.mkdir()
    monkeypatch.setattr(docker_env.tempfile, "gettempdir", lambda: str(temp_root))
    stable_source = str(tmp_path / "data")
    process_a = docker_env._runtime_reuse_fingerprint(
        ["-v", f"{temp_root}/hermes-skills-safe-a1b2c3:/root/.hermes/skills:ro",
         "-v", f"{stable_source}:/data:ro"],
        {},
    )
    process_b = docker_env._runtime_reuse_fingerprint(
        ["-v", f"{temp_root}/hermes-skills-safe-d4e5f6:/root/.hermes/skills:ro",
         "-v", f"{stable_source}:/data:ro"],
        {},
    )
    assert process_a == process_b
    # A real mount change still forces a fresh container.
    assert process_a != docker_env._runtime_reuse_fingerprint(
        ["-v", f"{temp_root}/hermes-skills-safe-a1b2c3:/root/.hermes/skills:ro",
         "-v", f"{tmp_path}/other-data:/data:ro"],
        {},
    )


def test_runtime_reuse_fingerprint_ignores_windows_volatile_skills_mounts(monkeypatch):
    """A native Windows temp path keeps its drive colon when the volatile source is removed."""
    monkeypatch.setattr(docker_env.tempfile, "gettempdir", lambda: r"C:\Temp")

    process_a = docker_env._runtime_reuse_fingerprint(
        ["-v", r"C:\Temp\hermes-skills-safe-a1b2c3:/root/.hermes/skills:ro"],
        {},
    )
    process_b = docker_env._runtime_reuse_fingerprint(
        ["-v", r"c:\temp\hermes-skills-safe-d4e5f6:/root/.hermes/skills:ro"],
        {},
    )

    assert process_a == process_b


def test_runtime_reuse_fingerprint_tracks_ordinary_temp_mounts(tmp_path, monkeypatch):
    """Only generated skills copies are disposable; a user's temp bind is posture."""
    temp_root = tmp_path / "tmp"
    temp_root.mkdir()
    monkeypatch.setattr(docker_env.tempfile, "gettempdir", lambda: str(temp_root))

    process_a = docker_env._runtime_reuse_fingerprint(
        ["-v", f"{temp_root}/workspace-a:/workspace"],
        {},
    )
    process_b = docker_env._runtime_reuse_fingerprint(
        ["-v", f"{temp_root}/workspace-b:/workspace"],
        {},
    )

    assert process_a != process_b


def test_runtime_reuse_identity_tracks_run_args_without_exposing_them():
    """The label changes with immutable run posture but contains no mount, env, or arg text."""
    private_mount = "/private/operator/workspace:/workspace"
    private_env = {"PRIVATE_TOKEN": "operator-secret-a"}
    baseline = docker_env._runtime_reuse_fingerprint(
        ["--network", "none", "-v", private_mount], private_env
    )

    assert baseline == docker_env._runtime_reuse_fingerprint(
        ["--network", "none", "-v", private_mount], private_env
    )
    assert baseline != docker_env._runtime_reuse_fingerprint(
        ["--network", "none"], private_env
    )
    assert baseline != docker_env._runtime_reuse_fingerprint(
        ["--network", "none", "-v", private_mount],
        {"PRIVATE_TOKEN": "operator-secret-b"},
    )
    assert len(baseline) == 24
    assert set(baseline) <= set("0123456789abcdef")


def test_runtime_reuse_identity_is_keyed(monkeypatch):
    """A readable label must not be an offline oracle for guessed env values or host paths."""
    args = ["--network", "none", "-v", "/private/operator/workspace:/workspace"]
    env = {"PRIVATE_TOKEN": "operator-secret-a"}

    monkeypatch.setattr(docker_env, "_runtime_reuse_key", lambda: b"a" * 32)
    first_installation = docker_env._runtime_reuse_fingerprint(args, env)
    monkeypatch.setattr(docker_env, "_runtime_reuse_key", lambda: b"b" * 32)
    second_installation = docker_env._runtime_reuse_fingerprint(args, env)

    assert first_installation != second_installation


def test_runtime_reuse_identity_handles_surrogate_escaped_posture():
    """POSIX paths and env values may contain bytes decoded through surrogateescape."""
    label = docker_env._runtime_reuse_fingerprint(
        ["-v", "/mnt/caf\udce9:/workspace"],
        {"PRIVATE_TOKEN": "a\udcffb"},
    )

    assert len(label) == 24
    assert set(label) <= set("0123456789abcdef")


def test_transient_cgroup_probe_does_not_reuse_unlimited_container(monkeypatch):
    """Effective limits are immutable posture, not label noise from the probe."""
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    monkeypatch.setattr(docker_env, "_get_active_profile_name", lambda: "default")
    _mock_subprocess_run(monkeypatch)
    monkeypatch.setattr(docker_env, "_cgroup_limits_available", lambda _image: False)

    unlimited = _make_dummy_env(
        task_id="transient-cgroup-probe",
        cpu=1.0,
        memory=512,
        persist_across_processes=False,
    )
    stale_runtime_label = unlimited._labels["hermes-runtime"]
    assert "--cpus" not in unlimited._all_run_args
    assert "--memory" not in unlimited._all_run_args
    assert "--pids-limit" not in unlimited._all_run_args

    calls = _mock_subprocess_run_with_reuse(
        monkeypatch,
        ps_state="running",
        expected_runtime_label=stale_runtime_label,
    )
    monkeypatch.setattr(docker_env, "_cgroup_limits_available", lambda _image: True)

    limited = _make_dummy_env(
        task_id="transient-cgroup-probe",
        cpu=1.0,
        memory=512,
    )

    assert limited._labels["hermes-runtime"] != stale_runtime_label
    assert limited._container_id == "fresh-cid"
    assert "--cpus" in limited._all_run_args
    assert "--memory" in limited._all_run_args
    assert "--pids-limit" in limited._all_run_args
    assert any(cmd[1] == "ps" for cmd, _kwargs in calls if len(cmd) >= 2)


def test_transient_storage_probe_does_not_reuse_unlimited_container(monkeypatch):
    """A disk-quota probe recovery must start a quota-bearing container."""
    if docker_env.sys.platform == "darwin":
        pytest.skip("Docker Desktop does not use the Linux storage-opt path")
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    monkeypatch.setattr(docker_env, "_get_active_profile_name", lambda: "default")
    _mock_subprocess_run(monkeypatch)
    monkeypatch.setattr(
        docker_env.DockerEnvironment,
        "_storage_opt_supported",
        staticmethod(lambda: False),
    )

    unlimited = _make_dummy_env(
        task_id="transient-storage-probe",
        disk=1024,
        persist_across_processes=False,
    )
    stale_runtime_label = unlimited._labels["hermes-runtime"]
    assert "--storage-opt" not in unlimited._all_run_args

    calls = _mock_subprocess_run_with_reuse(
        monkeypatch,
        ps_state="running",
        expected_runtime_label=stale_runtime_label,
    )
    monkeypatch.setattr(
        docker_env.DockerEnvironment,
        "_storage_opt_supported",
        staticmethod(lambda: True),
    )

    limited = _make_dummy_env(task_id="transient-storage-probe", disk=1024)

    assert limited._labels["hermes-runtime"] != stale_runtime_label
    assert limited._container_id == "fresh-cid"
    assert "--storage-opt" in limited._all_run_args
    assert "size=1024m" in limited._all_run_args
    assert any(cmd[1] == "ps" for cmd, _kwargs in calls if len(cmd) >= 2)


def test_runtime_reuse_key_is_private_and_stable(tmp_path):
    """Concurrent Hermes processes need one persistent machine-local HMAC key."""
    load_key = getattr(docker_env, "_load_or_create_runtime_reuse_key", lambda _path: None)
    key_path = tmp_path / "docker-runtime-reuse.key"

    first = load_key(key_path)
    second = load_key(key_path)

    assert isinstance(first, bytes)
    assert len(first) == 32
    assert second == first
    if os.name != "nt":
        assert key_path.stat().st_mode & 0o777 == 0o600


def test_runtime_reuse_key_creation_is_race_safe(tmp_path):
    key_path = tmp_path / "docker-runtime-reuse.key"

    with ThreadPoolExecutor(max_workers=16) as pool:
        keys = list(pool.map(docker_env._load_or_create_runtime_reuse_key, [key_path] * 32))

    assert len(set(keys)) == 1
    assert key_path.read_bytes() == keys[0]
    assert not list(tmp_path.glob(f".{key_path.name}.*.tmp"))


def test_runtime_reuse_key_rejects_fifo_without_hanging(tmp_path):
    if not hasattr(os, "mkfifo"):
        pytest.skip("FIFOs are not available on this platform")
    fifo = tmp_path / "fifo-key"
    os.mkfifo(fifo)
    result = {}

    def _load():
        try:
            docker_env._load_or_create_runtime_reuse_key(fifo)
        except BaseException as exc:  # captured for the parent test thread
            result["exc"] = exc

    worker = threading.Thread(target=_load, daemon=True)
    worker.start()
    worker.join(timeout=1)

    assert not worker.is_alive(), "opening a FIFO key path blocked indefinitely"
    assert isinstance(result.get("exc"), RuntimeError)
    assert "not a regular file" in str(result["exc"])


@pytest.mark.platforms("posix")
def test_runtime_reuse_lock_opens_nonblocking(tmp_path, monkeypatch):
    lock_path = tmp_path / ".docker-runtime-reuse.key.lock"

    class _OSProxy:
        def __getattr__(self, name):
            return getattr(os, name)

        @staticmethod
        def open(path, flags, *args):
            assert path == lock_path
            assert flags & os.O_NONBLOCK
            raise RuntimeError("open flags verified")

    monkeypatch.setattr(docker_env, "os", _OSProxy())

    with pytest.raises(RuntimeError, match="open flags verified"):
        with docker_env._RuntimeReuseFileLock(lock_path):
            pass


@pytest.mark.parametrize(
    "link_errno",
    [errno.EPERM, errno.EACCES, errno.EINVAL, errno.EOPNOTSUPP, errno.ENOSYS],
)
def test_runtime_reuse_key_falls_back_when_hardlinks_are_unsupported(
    tmp_path, monkeypatch, link_errno
):
    key_path = tmp_path / "docker-runtime-reuse.key"
    real_link = os.link

    class _OSProxy:
        def __getattr__(self, name):
            return getattr(os, name)

        @staticmethod
        def link(_source, _target):
            raise OSError(link_errno, "hardlinks unsupported")

    monkeypatch.setattr(docker_env, "os", _OSProxy())

    first = docker_env._load_or_create_runtime_reuse_key(key_path)
    second = docker_env._load_or_create_runtime_reuse_key(key_path)

    assert os.link is real_link
    assert second == first
    assert key_path.read_bytes() == first
    assert not list(tmp_path.glob(f".{key_path.name}.*.tmp"))


def test_runtime_reuse_key_converges_without_hardlinks_under_race(tmp_path, monkeypatch):
    """Two simultaneous fallback publishers must return one atomically published key."""
    key_path = tmp_path / "docker-runtime-reuse.key"
    link_barrier = threading.Barrier(2)
    link_attempts = []
    replace_calls = []

    class _OSProxy:
        def __getattr__(self, name):
            return getattr(os, name)

        @staticmethod
        def link(_source, _target):
            link_attempts.append(threading.get_ident())
            link_barrier.wait(timeout=5)
            raise OSError(errno.EOPNOTSUPP, "hardlinks unsupported")

        @staticmethod
        def replace(source, target):
            replace_calls.append((source, target))
            return os.replace(source, target)

    monkeypatch.setattr(docker_env, "os", _OSProxy())

    with ThreadPoolExecutor(max_workers=2) as pool:
        keys = list(pool.map(docker_env._load_or_create_runtime_reuse_key, [key_path] * 2))

    assert len(link_attempts) == 2, "the fallback publishers never raced"
    assert len(replace_calls) == 1, "more than one publisher replaced the final key"
    assert len(set(keys)) == 1
    assert key_path.read_bytes() == keys[0]
    assert not list(tmp_path.glob(f".{key_path.name}.*.tmp"))


def test_runtime_reuse_key_recovers_when_target_vanishes_after_eexist(tmp_path, monkeypatch):
    key_path = tmp_path / "docker-runtime-reuse.key"
    link_calls = []

    class _OSProxy:
        def __getattr__(self, name):
            return getattr(os, name)

        @staticmethod
        def link(source, target):
            link_calls.append((source, target))
            if len(link_calls) == 1:
                raise FileExistsError(errno.EEXIST, "target vanished")
            return os.link(source, target)

    monkeypatch.setattr(docker_env, "os", _OSProxy())

    key = docker_env._load_or_create_runtime_reuse_key(key_path)

    assert len(link_calls) >= 1
    assert key_path.read_bytes() == key
    assert not list(tmp_path.glob(f".{key_path.name}.*.tmp"))


def test_runtime_reuse_key_recovers_abandoned_empty_target(tmp_path):
    key_path = tmp_path / "docker-runtime-reuse.key"
    key_path.write_bytes(b"")

    key = docker_env._load_or_create_runtime_reuse_key(key_path)

    assert isinstance(key, bytes)
    assert len(key) == 32
    assert key_path.read_bytes() == key
    assert docker_env._load_or_create_runtime_reuse_key(key_path) == key


@pytest.mark.platforms("posix")
def test_runtime_reuse_key_rejects_corrupt_and_non_regular_files(tmp_path):
    corrupt = tmp_path / "corrupt-key"
    corrupt.write_bytes(b"short")
    with pytest.raises(RuntimeError, match="exactly 32 bytes"):
        docker_env._load_or_create_runtime_reuse_key(corrupt)

    directory = tmp_path / "directory-key"
    directory.mkdir()
    with pytest.raises(RuntimeError, match="not a regular file"):
        docker_env._load_or_create_runtime_reuse_key(directory)


def test_runtime_reuse_key_error_uses_safe_ephemeral_identity(monkeypatch, caplog):
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    monkeypatch.setattr(docker_env, "_get_active_profile_name", lambda: "default")
    _mock_subprocess_run(monkeypatch)
    monkeypatch.setattr(docker_env, "_RUNTIME_REUSE_KEYS", {})
    ephemeral_keys = iter((b"a" * 32, b"b" * 32))

    class _SecretsProxy:
        def __getattr__(self, name):
            return getattr(secrets, name)

        @staticmethod
        def token_bytes(_size):
            return next(ephemeral_keys)

    monkeypatch.setattr(docker_env, "secrets", _SecretsProxy())

    def _denied(_path):
        raise PermissionError("owner-only key cannot be written")

    monkeypatch.setattr(docker_env, "_load_or_create_runtime_reuse_key", _denied)

    with caplog.at_level(logging.WARNING):
        first = _make_dummy_env(task_id="unreadable-runtime-key")
        second = _make_dummy_env(task_id="unreadable-runtime-key")
        docker_env._RUNTIME_REUSE_KEYS.clear()  # simulate a fresh process
        third = _make_dummy_env(task_id="unreadable-runtime-key")

    assert first._labels["hermes-runtime"] == second._labels["hermes-runtime"]
    assert third._labels["hermes-runtime"] != first._labels["hermes-runtime"]
    assert "cross-process Docker reuse is disabled" in caplog.text


def test_runtime_label_excludes_image_but_tracks_env_posture(monkeypatch):
    """Image policy runs after reuse lookup; secret-bearing env posture stays in the label."""
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    monkeypatch.setattr(docker_env, "_get_active_profile_name", lambda: "default")
    _mock_subprocess_run(monkeypatch)

    baseline = _make_dummy_env(
        task_id="runtime-posture", image="python:3.11", env={"PRIVATE_TOKEN": "secret-a"}
    )
    changed_image = _make_dummy_env(
        task_id="runtime-posture", image="python:3.12", env={"PRIVATE_TOKEN": "secret-a"}
    )
    changed_env = _make_dummy_env(
        task_id="runtime-posture", image="python:3.11", env={"PRIVATE_TOKEN": "secret-b"}
    )

    assert baseline._labels["hermes-runtime"] == changed_image._labels["hermes-runtime"]
    assert baseline._labels["hermes-runtime"] != changed_env._labels["hermes-runtime"]
    for other_key in ("hermes-agent", "hermes-task-id", "hermes-profile", "hermes-egress"):
        assert baseline._labels[other_key] == changed_image._labels[other_key]
        assert baseline._labels[other_key] == changed_env._labels[other_key]


def test_runtime_label_changes_with_automatic_cwd_mount(monkeypatch, tmp_path):
    """The auto-mounted host cwd is resolved before the runtime label is captured."""
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    monkeypatch.setattr(docker_env, "_get_active_profile_name", lambda: "default")
    _mock_subprocess_run(monkeypatch)

    isolated = _make_dummy_env(task_id="cwd-posture")
    host_bound = _make_dummy_env(
        task_id="cwd-posture", host_cwd=str(tmp_path), auto_mount_cwd=True
    )

    assert isolated._labels["hermes-runtime"] != host_bound._labels["hermes-runtime"]
    assert f"{tmp_path}:/workspace" not in isolated._all_run_args
    assert f"{tmp_path}:/workspace" in host_bound._all_run_args


def test_reuse_probe_filters_on_runtime_fingerprint(monkeypatch, tmp_path):
    """Reuse and recovery must select the requested mount posture, not stale mounts."""
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    monkeypatch.setattr(docker_env, "_get_active_profile_name", lambda: "default")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "alpha"))
    # Keep the auto-mounted skills directory present from the first startup.
    for name in ("alpha", "beta"):
        (tmp_path / name / "skills").mkdir(parents=True)
    calls = _mock_subprocess_run(monkeypatch)

    def reuse_filters():
        return tuple(
            arg for cmd, _ in calls if isinstance(cmd, list) and cmd[1] == "ps"
            for arg in cmd if arg.startswith("label=")
        )

    config = {"image": "python:3.11", "volumes": ["volume-a:/workspace"]}
    _make_dummy_env(**config)
    original_filters = reuse_filters()
    assert original_filters
    calls.clear()
    _make_dummy_env(**config)
    assert reuse_filters() == original_filters

    config["volumes"] = ["volume-b:/workspace"]
    calls.clear()
    env = _make_dummy_env(**config)
    changed_filters = reuse_filters()
    assert changed_filters != original_filters
    requested_labels = {
        value.removeprefix("label=") for value in changed_filters
    }
    requested_runtime_labels = {
        value for value in requested_labels
        if value.startswith("hermes-runtime=")
    }
    assert f"hermes-runtime={env._labels['hermes-runtime']}" in requested_runtime_labels
    assert len(requested_runtime_labels) == 2
    assert requested_labels - requested_runtime_labels <= _labels_in_run_args(
        _run_args_from_calls(calls)
    )

    calls.clear()
    assert env._recreate_container()
    recovery_filters = reuse_filters()
    recovery_labels = {
        value.removeprefix("label=") for value in recovery_filters
    }
    assert recovery_labels <= _labels_in_run_args(_run_args_from_calls(calls))
    assert {
        value for value in recovery_labels
        if value.startswith("hermes-runtime=")
    } == {f"hermes-runtime={env._labels['hermes-runtime']}"}


def test_default_image_flip_reaches_existing_reuse_guard(monkeypatch):
    """An unpinned default change must find the old container before image policy runs."""
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    monkeypatch.setattr(docker_env, "_get_active_profile_name", lambda: "default")
    _mock_subprocess_run(monkeypatch)
    existing = _make_dummy_env(
        task_id="default-image-flip",
        image="old/image:1",
        persist_across_processes=False,
    )
    existing_runtime_label = existing._labels["hermes-runtime"]
    calls = _mock_subprocess_run_with_reuse(
        monkeypatch,
        ps_state="running",
        expected_runtime_label=existing_runtime_label,
    )
    monkeypatch.setattr(
        docker_env.DockerEnvironment,
        "_container_image",
        lambda self, container_id: "old/image:1",
    )

    current = _make_dummy_env(
        task_id="default-image-flip",
        image="python:3.11",
    )

    assert current._labels["hermes-runtime"] == existing_runtime_label
    assert current._container_id == "reused-cid"
    assert not any(
        isinstance(cmd, list) and len(cmd) >= 2 and cmd[1] == "run"
        for cmd, _kwargs in calls
    )


def test_default_image_flip_to_s6_reaches_existing_reuse_guard(monkeypatch):
    """Image-derived s6 run args must not hide an old unpinned sandbox."""
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    monkeypatch.setattr(docker_env, "_get_active_profile_name", lambda: "default")
    _mock_subprocess_run_with_reuse(
        monkeypatch, ps_state=None, entrypoint_json="null"
    )
    existing = _make_dummy_env(
        task_id="default-image-s6-flip",
        image="old/image:1",
        persist_across_processes=False,
    )
    existing_runtime_label = existing._labels["hermes-runtime"]
    _mock_subprocess_run_with_reuse(
        monkeypatch,
        ps_state="running",
        expected_runtime_label=existing_runtime_label,
        entrypoint_json='["/init"]',
        container_image="old/image:1",
    )

    current = _make_dummy_env(
        task_id="default-image-s6-flip",
        image="hermes-agent:latest",
    )

    assert current._labels["hermes-runtime"] != existing_runtime_label
    assert current._container_id == "reused-cid"


def test_same_image_init_style_change_does_not_cross_reuse_posture(monkeypatch):
    """An alternate-init lookup is valid only for an actual image-name transition."""
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    monkeypatch.setattr(docker_env, "_get_active_profile_name", lambda: "default")
    _mock_subprocess_run_with_reuse(
        monkeypatch, ps_state=None, entrypoint_json="null"
    )
    existing = _make_dummy_env(
        task_id="same-image-s6-flip",
        image="mutable/image:latest",
        persist_across_processes=False,
    )
    existing_runtime_label = existing._labels["hermes-runtime"]
    _mock_subprocess_run_with_reuse(
        monkeypatch,
        ps_state="running",
        expected_runtime_label=existing_runtime_label,
        entrypoint_json='["/init"]',
        container_image="mutable/image:latest",
    )

    current = _make_dummy_env(
        task_id="same-image-s6-flip",
        image="mutable/image:latest",
    )

    assert current._labels["hermes-runtime"] != existing_runtime_label
    assert current._container_id == "fresh-cid"


def test_uninspectable_alternate_init_match_starts_fresh(monkeypatch):
    """A broader init-style fallback must not attach when its image cannot be verified."""
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    monkeypatch.setattr(docker_env, "_get_active_profile_name", lambda: "default")
    _mock_subprocess_run_with_reuse(
        monkeypatch, ps_state=None, entrypoint_json="null"
    )
    existing = _make_dummy_env(
        task_id="uninspectable-s6-flip",
        image="old/image:1",
        persist_across_processes=False,
    )
    existing_runtime_label = existing._labels["hermes-runtime"]
    _mock_subprocess_run_with_reuse(
        monkeypatch,
        ps_state="running",
        expected_runtime_label=existing_runtime_label,
        entrypoint_json='["/init"]',
    )

    current = _make_dummy_env(
        task_id="uninspectable-s6-flip",
        image="hermes-agent:latest",
    )

    assert current._labels["hermes-runtime"] != existing_runtime_label
    assert current._container_id == "fresh-cid"


def test_reuse_rejects_container_from_different_mount_posture(monkeypatch):
    """Removing a host bind must not retain it through cross-process reuse.

    The terminal guard derives ``has_host_access`` from the current config. If
    Docker reuses an older container whose bind mounts differ, the guard sees
    an isolated sandbox while commands still reach the host.
    """
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    monkeypatch.setattr(docker_env, "_get_active_profile_name", lambda: "default")
    monkeypatch.setattr(docker_env, "_cgroup_limits_ok", True)
    calls = []
    stale_runtime_label = None

    def _run(cmd, **kwargs):
        calls.append(list(cmd) if isinstance(cmd, list) else cmd)
        if isinstance(cmd, list) and len(cmd) >= 2:
            if cmd[1] == "version":
                return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")
            if cmd[1] == "ps":
                # Model Docker's exact label filtering. The stale host-bound
                # container is visible only if the requested runtime label is
                # identical to the one captured under the old mount posture.
                stale_filter = f"label=hermes-runtime={stale_runtime_label}"
                stdout = (
                    "stale-host-bound\trunning\n"
                    if stale_runtime_label is not None and stale_filter in cmd
                    else ""
                )
                return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")
            if cmd[1] == "run":
                return subprocess.CompletedProcess(cmd, 0, stdout="fresh-container\n", stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(docker_env.subprocess, "run", _run)

    host_bound = _make_dummy_env(
        task_id="mount-posture",
        volumes=["/private/operator/workspace:/workspace"],
        persist_across_processes=False,
    )
    stale_runtime_label = host_bound._labels["hermes-runtime"]
    calls.clear()

    isolated = _make_dummy_env(task_id="mount-posture", volumes=[])

    assert isolated._labels["hermes-runtime"] != stale_runtime_label
    assert isolated._container_id == "fresh-container"
    ps_call = next(cmd for cmd in calls if isinstance(cmd, list) and cmd[1] == "ps")
    run_call = next(cmd for cmd in calls if isinstance(cmd, list) and cmd[1] == "run")
    runtime_label = isolated._labels["hermes-runtime"]
    assert f"label=hermes-runtime={runtime_label}" in ps_call
    assert f"hermes-runtime={runtime_label}" in run_call
