"""Unit tests for tools/docker_host_mapping.py.

Detection fixtures redirect the module's path constants at temp files; the Docker
CLI is faked with a shell script that replays canned ``docker inspect`` payloads,
so no real Docker socket is involved.
"""

import json
import types

import pytest

import tools.docker_host_mapping as dhm

# These tests drive the detection fixtures directly; the root-conftest autouse stub
# that pins gateway Docker detection to "bare metal" would defeat them.
pytestmark = pytest.mark.real_container_detection


@pytest.fixture(autouse=True)
def _reset_caches():
    dhm.reset_host_mapping_caches()
    yield
    dhm.reset_host_mapping_caches()


@pytest.fixture()
def fake_proc(tmp_path, monkeypatch):
    """Point every host-side probe at an empty/absent tmp layout by default."""
    monkeypatch.setattr(dhm, "_DOCKERENV_PATHS", (str(tmp_path / "no-dockerenv"),))
    monkeypatch.setattr(dhm, "_ETC_HOSTNAME_PATH", str(tmp_path / "hostname"))
    monkeypatch.setattr(dhm, "_PROC_1_CGROUP_PATH", str(tmp_path / "cgroup1"))
    monkeypatch.setattr(dhm, "_SELF_CGROUP_PATH", str(tmp_path / "cgroupself"))
    monkeypatch.setattr(dhm, "_SELF_MOUNTINFO_PATH", str(tmp_path / "mountinfo"))


class TestIsRunningInDocker:
    def test_dockerenv_present(self, tmp_path, monkeypatch, fake_proc):
        marker = tmp_path / "dockerenv"
        marker.touch()
        monkeypatch.setattr(dhm, "_DOCKERENV_PATHS", (str(marker),))
        assert dhm.is_running_in_docker() is True

    def test_containerenv_present(self, tmp_path, monkeypatch, fake_proc):
        marker = tmp_path / "containerenv"
        marker.touch()
        monkeypatch.setattr(dhm, "_DOCKERENV_PATHS", (str(marker),))
        assert dhm.is_running_in_docker() is True

    def test_cgroup_names_containerd(self, tmp_path, monkeypatch, fake_proc):
        (tmp_path / "cgroup1").write_text("0::/system.slice/cri-containerd-abc.scope\n")
        assert dhm.is_running_in_docker() is True

    def test_bare_metal_false(self, fake_proc):
        assert dhm.is_running_in_docker() is False

    def test_lxc_style_cgroup_false(self, tmp_path, monkeypatch, fake_proc):
        (tmp_path / "cgroup1").write_text("0::/\n")
        assert dhm.is_running_in_docker() is False


class TestContainerIdDetection:
    def test_hex_hostname_first(self, tmp_path, monkeypatch, fake_proc):
        (tmp_path / "hostname").write_text("3f2a9b1c0d4e\n")
        (tmp_path / "cgroupself").write_text("0::/docker-3f2a9b1c0d4e5f60718293a4b5c6d7e8f90a1b2c3d4e5f60718293a4b5c6d7e8.scope\n")
        assert dhm.get_own_container_id() == "3f2a9b1c0d4e"
        cands = dhm._container_id_candidates()
        assert cands == ["3f2a9b1c0d4e",
                         "3f2a9b1c0d4e5f60718293a4b5c6d7e8f90a1b2c3d4e5f60718293a4b5c6d7e8"]

    def test_cgroup_fallback_when_hostname_not_hex(self, tmp_path, monkeypatch, fake_proc):
        (tmp_path / "hostname").write_text("hermes-gateway\n")
        (tmp_path / "cgroupself").write_text("0::/system.slice/docker-90abcdef0123456789abcdef0123456789abcdef0123456789abcdef01234567.scope\n")
        assert dhm.get_own_container_id() == (
            "90abcdef0123456789abcdef0123456789abcdef0123456789abcdef01234567")
        # Custom hostname stays as last-resort inspect candidate.
        assert dhm._container_id_candidates()[-1] == "hermes-gateway"

    def test_mountinfo_fallback(self, tmp_path, monkeypatch, fake_proc):
        (tmp_path / "mountinfo").write_text(
            '36 31 8:1 / /var/lib/docker/containers/'
            '1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef/merged rw\n')
        assert dhm.get_own_container_id() == (
            "1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef")

    def test_nothing_detectable_returns_none(self, fake_proc):
        assert dhm.get_own_container_id() is None
        assert dhm._container_id_candidates() == []


class _FakeProcResult:
    def __init__(self, returncode: int = 0, stdout: str = "", stderr: str = ""):
        self.returncode, self.stdout, self.stderr = returncode, stdout, stderr


def _stub_subprocess(monkeypatch, respond):
    """Replace ``docker_host_mapping.subprocess`` with a run() stub replaying canned
    inspect responses (``respond(argv) -> _FakeProcResult | None``); returns the stub
    so tests can assert on the argv sequence. tmp_path is noexec in some sandboxes,
    so a real fake CLI script is not portable."""
    import subprocess as real_subprocess

    calls: list[list[str]] = []

    def run(argv, **kwargs):
        calls.append(list(argv))
        return respond(list(argv)) or _FakeProcResult(1, "", "unhandled argv")

    monkeypatch.setattr(dhm, "subprocess", types.SimpleNamespace(
        run=run, TimeoutExpired=real_subprocess.TimeoutExpired, OSError=OSError))
    return calls


class TestGetOwnMounts:
    MOUNTS = [
        {"Type": "bind", "Source": "/home/user/hermes/data", "Destination": "/opt/data", "Mode": "rw", "RW": True},
        {"Type": "volume", "Source": "/var/lib/docker/volumes/x/_data", "Destination": "/persist", "RW": True},
        {"Type": "tmpfs", "Source": "", "Destination": "/tmp", "RW": True},
    ]

    def test_template_inspect_roundtrip(self, tmp_path, monkeypatch, fake_proc):
        (tmp_path / "hostname").write_text("abcd1234ef56\n")
        monkeypatch.setattr(dhm, "_find_docker_cli", lambda: "/usr/bin/docker")
        calls = _stub_subprocess(monkeypatch, lambda argv:
                                 _FakeProcResult(0, json.dumps(self.MOUNTS)) if "--format" in argv else None)
        assert dhm.get_own_mounts() == self.MOUNTS
        assert calls == [["/usr/bin/docker", "inspect", "--format", "{{json .Mounts}}", "abcd1234ef56"]]

    def test_full_inspect_fallback_when_template_unsupported(self, tmp_path, monkeypatch, fake_proc):
        (tmp_path / "hostname").write_text("abcd1234ef56\n")
        payload = json.dumps([{"Id": "x", "Mounts": self.MOUNTS}])
        monkeypatch.setattr(dhm, "_find_docker_cli", lambda: "/usr/bin/docker")

        def respond(argv):
            if "--format" in argv:
                return _FakeProcResult(1, "", "template unsupported")
            return _FakeProcResult(0, payload)

        calls = _stub_subprocess(monkeypatch, respond)
        assert dhm.get_own_mounts() == self.MOUNTS
        assert len(calls) == 2 and calls[1] == ["/usr/bin/docker", "inspect", "abcd1234ef56"]

    def test_no_cli_returns_empty(self, monkeypatch, caplog, fake_proc):
        monkeypatch.setattr(dhm, "_find_docker_cli", lambda: None)
        assert dhm.get_own_mounts() == []
        assert any(r.levelname == "WARNING" for r in caplog.records)

    def test_all_candidates_fail_warns(self, tmp_path, monkeypatch, caplog, fake_proc):
        (tmp_path / "hostname").write_text("abcd1234ef56\n")
        monkeypatch.setattr(dhm, "_find_docker_cli", lambda: "/usr/bin/docker")
        _stub_subprocess(monkeypatch, lambda argv: _FakeProcResult(1, "", "no such object"))
        assert dhm.get_own_mounts() == []
        assert any("could not resolve" in r.message for r in caplog.records)

    def test_first_bad_candidate_skipped(self, tmp_path, monkeypatch, fake_proc):
        # cgroup-derived id first, hostname second; fake CLI only resolves the hostname.
        (tmp_path / "hostname").write_text("gateway-name\n")
        (tmp_path / "cgroupself").write_text(
            "0::/docker-9999999999999999999999999999999999999999999999999999999999999999.scope\n")
        monkeypatch.setattr(dhm, "_find_docker_cli", lambda: "/usr/bin/docker")

        def respond(argv):
            if argv[-1] == "gateway-name":
                return _FakeProcResult(0, '[{"Type":"bind","Source":"/h","Destination":"/c"}]')
            return _FakeProcResult(1, "", "no such object")

        calls = _stub_subprocess(monkeypatch, respond)
        assert dhm.get_own_mounts() == [{"Type": "bind", "Source": "/h", "Destination": "/c"}]
        assert [argv[-1] for argv in calls] == [
            "9999999999999999999999999999999999999999999999999999999999999999",  # template
            "9999999999999999999999999999999999999999999999999999999999999999",  # full inspect
            "gateway-name"]


class TestBuildAutoHostMapping:
    def test_filters_and_orders_longest_first(self, monkeypatch):
        monkeypatch.setattr(dhm, "get_own_mounts", lambda: [
            {"Type": "bind", "Source": "/host/home", "Destination": "/opt"},
            {"Type": "bind", "Source": "/host/data", "Destination": "/opt/data"},
            {"Type": "volume", "Source": "/var/lib/docker/volumes/v/_data", "Destination": "/vol"},
            {"Type": "tmpfs", "Source": "", "Destination": "/tmp"},
            {"Type": "bind", "Source": "", "Destination": "/empty-src"},
            {"Type": "bind", "Source": "/host/root", "Destination": "/"},
            {"Type": "bind", "Source": "relative-nope", "Destination": "/rel"},
        ])
        mapping = dhm.build_auto_host_mapping()
        assert mapping == {"/opt/data": "/host/data", "/opt": "/host/home"}
        assert list(mapping) == ["/opt/data", "/opt"]  # longest prefix first

    def test_cached_until_reset(self, monkeypatch):
        calls = []

        def mounts():
            calls.append(1)
            return [{"Type": "bind", "Source": "/h", "Destination": "/c"}]

        monkeypatch.setattr(dhm, "get_own_mounts", mounts)
        dhm.build_auto_host_mapping()
        dhm.build_auto_host_mapping()
        assert calls == [1]
        dhm.reset_host_mapping_caches()
        dhm.build_auto_host_mapping()
        assert len(calls) == 2

    def test_empty_when_detection_fails(self, monkeypatch):
        monkeypatch.setattr(dhm, "get_own_mounts", list)
        assert dhm.build_auto_host_mapping() == {}


class TestApplyHostMapping:
    MAPPING = {"/opt/data": "/home/user/hermes/data", "/opt": "/host/opt"}

    def test_exact_prefix(self):
        assert dhm.apply_host_mapping("/opt/data", self.MAPPING) == "/home/user/hermes/data"

    def test_nested_path(self):
        assert dhm.apply_host_mapping("/opt/data/cache/web/x.md", self.MAPPING) == \
               "/home/user/hermes/data/cache/web/x.md"

    def test_longest_prefix_wins(self):
        assert dhm.apply_host_mapping("/opt/data", {"/opt": "/host/opt", "/opt/data": "/host/data"}) == \
               "/host/data"

    def test_no_match_is_noop(self):
        assert dhm.apply_host_mapping("/elsewhere/x", self.MAPPING) == "/elsewhere/x"

    def test_partial_segment_not_matched(self):
        # "/opt/datax" must NOT be rewritten by the "/opt/data" prefix (segment boundary).
        assert dhm.apply_host_mapping("/opt/datax", {"/opt/data": "/host/data"}) == "/opt/datax"

    def test_empty_mapping_noop(self):
        assert dhm.apply_host_mapping("/opt/data", {}) == "/opt/data"

    def test_trailing_slashes_in_config_tolerated(self):
        assert dhm.apply_host_mapping("/opt/data/x", {"/opt/data/": "/host/data/"}) == "/host/data/x"


class TestMountBuilderIntegration:
    """The Docker mount builder emits HOST-view ``-v`` sources when the gateway
    is itself containerized and gateway-view sources everywhere else. The gateway-view source
    must exist before the mount is emitted (DinD auto-creates missing sources as dirs)."""

    @staticmethod
    def _ro_args(monkeypatch, mapping, cache_mounts):
        import tools.credential_files as cf
        import tools.environments.docker as docker_env
        monkeypatch.setattr(dhm, "resolve_host_mapping", lambda: dict(mapping))
        monkeypatch.setattr(cf, "get_cache_directory_mounts", lambda: list(cache_mounts))
        monkeypatch.setattr(cf, "get_skills_directory_mount", list)
        monkeypatch.setattr(cf, "get_credential_file_mounts", list)
        return docker_env._readonly_skill_mount_args()

    @staticmethod
    def _volumes(args):
        return [args[i + 1] for i, a in enumerate(args) if a == "-v"]

    def _existing_cache_mount(self, tmp_path):
        web = tmp_path / "opt-data" / "cache" / "web"
        web.mkdir(parents=True)
        return web, [{"host_path": str(web), "container_path": "/root/.hermes/cache/web"}]

    def test_containerized_gateway_emits_host_paths(self, tmp_path, monkeypatch):
        _web, mounts = self._existing_cache_mount(tmp_path)
        args = self._ro_args(monkeypatch, {str(tmp_path / "opt-data"): "/home/user/hermes/data"}, mounts)
        assert self._volumes(args) == ["/home/user/hermes/data/cache/web:/root/.hermes/cache/web:ro"]

    def test_bare_metal_keeps_current_behavior(self, tmp_path, monkeypatch):
        web, mounts = self._existing_cache_mount(tmp_path)
        args = self._ro_args(monkeypatch, {}, mounts)
        assert self._volumes(args) == [f"{web}:/root/.hermes/cache/web:ro"]

    def test_missing_source_skipped_despite_mapping(self, tmp_path, monkeypatch):
        mounts = [{"host_path": str(tmp_path / "gone"), "container_path": "/root/.hermes/cache/web"}]
        assert self._ro_args(monkeypatch, {str(tmp_path): "/host"}, mounts) == []

    def test_sandbox_dirs_mapped_gateway_view_kept(self, tmp_path, monkeypatch):
        import tools.environments.docker as docker_env
        monkeypatch.setenv("TERMINAL_SANDBOX_DIR", str(tmp_path / "sandboxes"))
        monkeypatch.setattr(dhm, "resolve_host_mapping",
                            lambda: {str(tmp_path / "sandboxes"): "/host/hermes/data/sandboxes"})
        env = object.__new__(docker_env.DockerEnvironment)
        env._persistent = True
        env._sandbox_mounts = []
        _volume_args, writable = docker_env.DockerEnvironment._mount_args(env, [], None, False, "t1")
        assert self._volumes(writable) == [
            "/host/hermes/data/sandboxes/docker/t1/home:/root",
            "/host/hermes/data/sandboxes/docker/t1/workspace:/workspace",
        ]
        # Gateway-side bookkeeping (chown/reclaim) keeps the gateway-view paths.
        assert env._sandbox_mounts == [
            (str(tmp_path / "sandboxes" / "docker" / "t1" / "home"), "/root"),
            (str(tmp_path / "sandboxes" / "docker" / "t1" / "workspace"), "/workspace"),
        ]

    def test_host_cwd_bind_translated(self, tmp_path, monkeypatch):
        import tools.environments.docker as docker_env
        repo = tmp_path / "opt-data" / "repo"
        repo.mkdir(parents=True)
        monkeypatch.setattr(dhm, "resolve_host_mapping", lambda: {str(tmp_path / "opt-data"): "/host/data"})
        env = object.__new__(docker_env.DockerEnvironment)
        env._persistent = False
        env._sandbox_mounts = []
        volume_args, _writable = docker_env.DockerEnvironment._mount_args(env, [], str(repo), True, "t1")
        assert self._volumes(volume_args) == ["/host/data/repo:/workspace"]
        assert env.host_cwd == str(repo)  # tools still see the gateway view

    def test_user_volumes_never_rewritten(self, tmp_path, monkeypatch):
        import tools.environments.docker as docker_env
        monkeypatch.setattr(dhm, "resolve_host_mapping", lambda: {"/opt/data": "/host/data"})
        env = object.__new__(docker_env.DockerEnvironment)
        env._persistent = False
        env._sandbox_mounts = []
        volume_args, _writable = docker_env.DockerEnvironment._mount_args(
            env, ["/host/data/x:/in-sandbox"], None, False, "t1")
        assert self._volumes(volume_args) == ["/host/data/x:/in-sandbox"]
