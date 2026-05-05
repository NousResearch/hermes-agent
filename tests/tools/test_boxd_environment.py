"""Unit tests for the boxd cloud VM environment backend."""

import io
import tarfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Helpers to build mock boxd SDK objects
# ---------------------------------------------------------------------------


@dataclass
class _ExecResult:
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0

    @property
    def success(self) -> bool:
        return self.exit_code == 0


def _make_box(box_id: str = "vm-123", status: str = "running") -> MagicMock:
    box = MagicMock()
    box.id = box_id
    box.name = box_id
    box.status = status
    box.exec.return_value = _ExecResult()
    return box


def _patch_boxd_imports(monkeypatch):
    """Patch the boxd SDK so BoxdEnvironment can be imported without it installed."""
    import sys
    import types as _types

    boxd_mod = _types.ModuleType("boxd")
    boxd_mod.Compute = MagicMock(name="Compute")

    # Captured BoxConfig / LifecycleConfig calls — tests assert against the
    # kwargs to verify resource conversion.
    @dataclass
    class _LifecycleConfig:
        auto_suspend_timeout: int = 0
        auto_destroy_timeout: int = 0

    @dataclass
    class _BoxConfig:
        vcpu: int = 0
        memory: str = ""
        disk: str = ""
        env: dict = field(default_factory=dict)
        cmd: list = field(default_factory=list)
        restart_policy: str = "always"
        lifecycle: _LifecycleConfig | None = None

    boxd_mod.BoxConfig = _BoxConfig
    boxd_mod.LifecycleConfig = _LifecycleConfig

    class _BoxdError(Exception):
        pass

    class _NotFoundError(_BoxdError):
        pass

    boxd_mod.BoxdError = _BoxdError
    boxd_mod.NotFoundError = _NotFoundError

    monkeypatch.setitem(sys.modules, "boxd", boxd_mod)
    return boxd_mod


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def boxd_sdk(monkeypatch):
    """Provide a mock boxd SDK module and return it for assertions."""
    return _patch_boxd_imports(monkeypatch)


@pytest.fixture()
def make_env(boxd_sdk, monkeypatch):
    """Factory that constructs a BoxdEnvironment with a mocked SDK."""
    # Stop is_interrupted from interfering with execute() during tests
    monkeypatch.setattr("tools.environments.base.is_interrupted", lambda: False)

    def _factory(
        box=None,
        get_side_effect=None,
        home_dir: str = "/root",
        persistent: bool = True,
        **kwargs,
    ):
        box = box or _make_box()
        # Default $HOME detection result if the test didn't override exec
        if not box.exec.side_effect:
            box.exec.return_value = _ExecResult(stdout=home_dir)

        mock_compute = MagicMock()
        mock_compute.box.create.return_value = box

        if get_side_effect is not None:
            mock_compute.box.get.side_effect = get_side_effect
        else:
            mock_compute.box.get.side_effect = boxd_sdk.NotFoundError("not found")

        boxd_sdk.Compute = MagicMock(return_value=mock_compute)

        from tools.environments.boxd import BoxdEnvironment

        env = BoxdEnvironment(
            image=kwargs.pop("image", ""),
            persistent_filesystem=persistent,
            **kwargs,
        )
        env._mock_compute = mock_compute
        env._mock_box = box
        return env

    return _factory


# ---------------------------------------------------------------------------
# Constructor / cwd resolution
# ---------------------------------------------------------------------------


class TestCwdResolution:
    def test_default_cwd_resolves_home(self, make_env):
        env = make_env(home_dir="/home/ubuntu")
        assert env.cwd == "/home/ubuntu"

    def test_tilde_cwd_resolves_home(self, make_env):
        env = make_env(cwd="~", home_dir="/home/ubuntu")
        assert env.cwd == "/home/ubuntu"

    def test_explicit_cwd_not_overridden(self, make_env):
        env = make_env(cwd="/workspace", home_dir="/root")
        assert env.cwd == "/workspace"

    def test_home_detection_failure_keeps_default_cwd(self, make_env):
        box = _make_box()
        box.exec.side_effect = RuntimeError("exec failed")
        env = make_env(box=box)
        # Falls back to constructor default ("/root") since $HOME detection
        # raised before init_session ran.
        assert env.cwd == "/root"

    def test_empty_home_keeps_default_cwd(self, make_env):
        env = make_env(home_dir="")
        assert env.cwd == "/root"


# ---------------------------------------------------------------------------
# Sandbox persistence / resume
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_persistent_resumes_via_get(self, make_env):
        existing = _make_box(box_id="vm-existing", status="suspended")
        existing.exec.return_value = _ExecResult(stdout="/root")
        env = make_env(get_side_effect=lambda name: existing,
                       persistent=True, task_id="mytask")
        existing.resume.assert_called_once()
        env._mock_compute.box.get.assert_called_once_with("hermes-mytask")
        env._mock_compute.box.create.assert_not_called()

    def test_persistent_resume_skipped_when_already_running(self, make_env):
        existing = _make_box(box_id="vm-existing", status="running")
        existing.exec.return_value = _ExecResult(stdout="/root")
        env = make_env(get_side_effect=lambda name: existing,
                       persistent=True, task_id="mytask")
        existing.resume.assert_not_called()
        env._mock_compute.box.create.assert_not_called()

    def test_persistent_creates_new_when_not_found(self, make_env, boxd_sdk):
        env = make_env(
            get_side_effect=boxd_sdk.NotFoundError("not found"),
            persistent=True,
            task_id="mytask",
        )
        env._mock_compute.box.create.assert_called_once()
        kwargs = env._mock_compute.box.create.call_args.kwargs
        assert kwargs["name"] == "hermes-mytask"
        assert "image" not in kwargs  # empty image -> not passed through

    def test_persistent_creates_new_on_other_resume_failure(self, make_env, boxd_sdk):
        # An error other than NotFoundError during resume should still fall
        # back to a fresh create (matching daytona's defensive resume path).
        env = make_env(
            get_side_effect=RuntimeError("transient"),
            persistent=True,
        )
        env._mock_compute.box.create.assert_called_once()

    def test_non_persistent_skips_get(self, make_env):
        env = make_env(persistent=False)
        env._mock_compute.box.get.assert_not_called()
        env._mock_compute.box.create.assert_called_once()

    def test_image_passed_through_when_provided(self, make_env):
        env = make_env(image="ubuntu:22.04", persistent=False)
        kwargs = env._mock_compute.box.create.call_args.kwargs
        assert kwargs["image"] == "ubuntu:22.04"


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------


class TestCleanup:
    def test_persistent_cleanup_suspends_box(self, make_env):
        env = make_env(persistent=True)
        box = env._mock_box
        env.cleanup()
        box.suspend.assert_called_once()
        box.destroy.assert_not_called()

    def test_non_persistent_cleanup_destroys_box(self, make_env):
        env = make_env(persistent=False)
        box = env._mock_box
        env.cleanup()
        box.destroy.assert_called_once()
        box.suspend.assert_not_called()

    def test_cleanup_idempotent(self, make_env):
        env = make_env(persistent=True)
        env.cleanup()
        env.cleanup()  # should not raise

    def test_cleanup_swallows_errors(self, make_env):
        env = make_env(persistent=True)
        env._mock_box.suspend.side_effect = RuntimeError("suspend failed")
        env.cleanup()  # should not raise
        assert env._box is None

    def test_cleanup_closes_compute(self, make_env):
        env = make_env(persistent=True)
        env.cleanup()
        env._mock_compute.close.assert_called_once()


# ---------------------------------------------------------------------------
# Execute
# ---------------------------------------------------------------------------


class TestExecute:
    def test_basic_command(self, make_env):
        box = _make_box()
        # (1) $HOME, (2) init_session bootstrap, (3) actual command
        box.exec.side_effect = [
            _ExecResult(stdout="/root"),
            _ExecResult(stdout="", exit_code=0),
            _ExecResult(stdout="hello", exit_code=0),
        ]
        env = make_env(box=box)
        result = env.execute("echo hello")
        assert "hello" in result["output"]
        assert result["returncode"] == 0

    def test_sdk_timeout_passed_to_exec(self, make_env):
        box = _make_box()
        box.exec.side_effect = [
            _ExecResult(stdout="/root"),
            _ExecResult(stdout="", exit_code=0),
            _ExecResult(stdout="ok", exit_code=0),
        ]
        env = make_env(box=box, timeout=42)
        env.execute("echo hello")
        call_kwargs = box.exec.call_args_list[-1].kwargs
        assert call_kwargs["timeout"] == 42

    def test_nonzero_exit_code(self, make_env):
        box = _make_box()
        box.exec.side_effect = [
            _ExecResult(stdout="/root"),
            _ExecResult(stdout="", exit_code=0),
            _ExecResult(stdout="", stderr="not found", exit_code=127),
        ]
        env = make_env(box=box)
        result = env.execute("bad_cmd")
        assert result["returncode"] == 127
        assert "not found" in result["output"]

    def test_stderr_merged_into_output(self, make_env):
        box = _make_box()
        box.exec.side_effect = [
            _ExecResult(stdout="/root"),
            _ExecResult(stdout="", exit_code=0),
            _ExecResult(stdout="line1", stderr="warning", exit_code=0),
        ]
        env = make_env(box=box)
        result = env.execute("noisy")
        assert "line1" in result["output"]
        assert "warning" in result["output"]

    def test_stdin_data_wraps_heredoc(self, make_env):
        box = _make_box()
        box.exec.side_effect = [
            _ExecResult(stdout="/root"),
            _ExecResult(stdout="", exit_code=0),
            _ExecResult(stdout="ok", exit_code=0),
        ]
        env = make_env(box=box)
        env.execute("python3", stdin_data="print('hi')")
        # stdin_mode=heredoc => stdin gets embedded as a heredoc inside the
        # bash command string itself, so look for the marker.
        cmd_string = box.exec.call_args_list[-1].args[2]
        assert "HERMES_STDIN_" in cmd_string
        assert "print" in cmd_string
        assert "hi" in cmd_string

    def test_custom_cwd_in_command_wrapper(self, make_env):
        box = _make_box()
        box.exec.side_effect = [
            _ExecResult(stdout="/root"),
            _ExecResult(stdout="", exit_code=0),
            _ExecResult(stdout="/tmp", exit_code=0),
        ]
        env = make_env(box=box)
        env.execute("pwd", cwd="/tmp")
        cmd_string = box.exec.call_args_list[-1].args[2]
        assert "cd" in cmd_string
        assert "/tmp" in cmd_string
        # CWD must NOT be a kwarg to box.exec — it's wrapped into the cmd.
        assert "cwd" not in box.exec.call_args_list[-1].kwargs

    def test_boxd_error_returns_rc1(self, make_env, boxd_sdk):
        box = _make_box()
        box.exec.side_effect = [
            _ExecResult(stdout="/root"),
            _ExecResult(stdout="", exit_code=0),
            boxd_sdk.BoxdError("transient"),
        ]
        env = make_env(box=box)
        result = env.execute("echo x")
        # Errors raised from inside the SDK exec call surface through
        # _ThreadedProcessHandle as returncode=1 (no retry).
        assert result["returncode"] == 1


# ---------------------------------------------------------------------------
# Resource conversion (MB -> "NG")
# ---------------------------------------------------------------------------


class TestResourceConversion:
    def _box_config_kwargs(self, boxd_sdk):
        # The mock BoxConfig is a dataclass — capture the last instance built.
        # We grab it from the create call's BoxConfig argument.
        call_kwargs = (
            boxd_sdk.Compute.return_value.box.create.call_args.kwargs
        )
        return call_kwargs.get("config")

    def test_memory_converted_to_gib_string(self, make_env, boxd_sdk):
        env = make_env(memory=8192, persistent=False)
        cfg = self._box_config_kwargs(boxd_sdk)
        assert cfg.memory == "8G"

    def test_disk_converted_to_gib_string(self, make_env, boxd_sdk):
        env = make_env(disk=102400, persistent=False)
        cfg = self._box_config_kwargs(boxd_sdk)
        assert cfg.disk == "100G"

    def test_small_memory_uses_mb_suffix(self, make_env, boxd_sdk):
        env = make_env(memory=512, disk=1024, persistent=False)
        cfg = self._box_config_kwargs(boxd_sdk)
        # 512 MB -> "512M"; 1024 MB -> "1G"
        assert cfg.memory == "512M"
        assert cfg.disk == "1G"

    def test_vcpu_passes_through(self, make_env, boxd_sdk):
        env = make_env(cpu=4, persistent=False)
        cfg = self._box_config_kwargs(boxd_sdk)
        assert cfg.vcpu == 4

    def test_auto_suspend_timeout_passed_to_lifecycle(self, make_env, boxd_sdk):
        env = make_env(auto_suspend_timeout=120, persistent=False)
        cfg = self._box_config_kwargs(boxd_sdk)
        assert cfg.lifecycle.auto_suspend_timeout == 120


# ---------------------------------------------------------------------------
# Interrupt handling
# ---------------------------------------------------------------------------


class TestInterrupt:
    def test_interrupt_suspends_box_and_returns_130(self, make_env, monkeypatch):
        box = _make_box()
        event = threading.Event()
        calls = {"n": 0}

        def exec_side_effect(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                return _ExecResult(stdout="/root")  # $HOME
            if calls["n"] == 2:
                return _ExecResult(stdout="", exit_code=0)  # init_session
            event.wait(timeout=5)  # "long-running"
            return _ExecResult(stdout="done", exit_code=0)

        box.exec.side_effect = exec_side_effect
        env = make_env(box=box)

        monkeypatch.setattr(
            "tools.environments.base.is_interrupted", lambda: True
        )
        try:
            result = env.execute("sleep 10")
            assert result["returncode"] == 130
            box.suspend.assert_called()
        finally:
            event.set()


# ---------------------------------------------------------------------------
# Wake / ensure ready
# ---------------------------------------------------------------------------


class TestEnsureBoxReady:
    def test_suspended_status_triggers_resume(self, make_env):
        env = make_env()
        env._box.status = "suspended"
        env._box.resume.reset_mock()
        env._ensure_box_ready()
        env._box.resume.assert_called_once()

    def test_running_status_no_resume(self, make_env):
        env = make_env()
        env._box.status = "running"
        env._box.resume.reset_mock()
        env._ensure_box_ready()
        env._box.resume.assert_not_called()


# ---------------------------------------------------------------------------
# File sync
# ---------------------------------------------------------------------------


class TestFileSync:
    def test_single_upload_calls_write_file(self, make_env, tmp_path):
        env = make_env()
        host_file = tmp_path / "skills" / "x.md"
        host_file.parent.mkdir(parents=True)
        host_file.write_bytes(b"hello world")
        env._mock_box.exec.reset_mock()
        env._mock_box.write_file.reset_mock()

        env._boxd_upload(str(host_file), "/root/.hermes/skills/x.md")

        # First exec sets up the parent dir
        first_call = env._mock_box.exec.call_args_list[0]
        assert "mkdir -p" in first_call.args[2]
        # Then write_file uploads the bytes verbatim
        env._mock_box.write_file.assert_called_once()
        args = env._mock_box.write_file.call_args.args
        assert args[0] == b"hello world"
        assert args[1] == "/root/.hermes/skills/x.md"

    def test_bulk_upload_packs_tar_and_extracts(self, make_env, tmp_path):
        env = make_env()
        # Capture write_file payload so we can decode the tarball
        captured_payload = {}

        def _capture(content, path):
            captured_payload["content"] = content
            captured_payload["path"] = path

        env._mock_box.write_file.side_effect = _capture
        env._mock_box.exec.reset_mock()
        env._mock_box.exec.return_value = _ExecResult(exit_code=0)

        # Build two host files
        f1 = tmp_path / "a.txt"
        f1.write_bytes(b"alpha")
        f2 = tmp_path / "b.txt"
        f2.write_bytes(b"beta")

        env._boxd_bulk_upload([
            (str(f1), "/root/.hermes/a.txt"),
            (str(f2), "/root/.hermes/b.txt"),
        ])

        # 1) Tar uploaded as a single file in /tmp
        assert captured_payload["path"].startswith("/tmp/")
        assert captured_payload["path"].endswith(".tar.gz")

        # 2) Tar contents match what we expected
        with tarfile.open(fileobj=io.BytesIO(captured_payload["content"]),
                          mode="r:gz") as tar:
            members = {m.name: m for m in tar.getmembers()}
            assert "root/.hermes/a.txt" in members
            assert "root/.hermes/b.txt" in members
            assert tar.extractfile("root/.hermes/a.txt").read() == b"alpha"

        # 3) Extract command runs after the upload
        ran = " ".join(c.args[2] for c in env._mock_box.exec.call_args_list)
        assert "tar xzf" in ran
        assert "rm -f" in ran  # staging file cleaned up

    def test_bulk_upload_noop_for_empty_list(self, make_env):
        env = make_env()
        env._mock_box.write_file.reset_mock()
        env._mock_box.exec.reset_mock()
        env._boxd_bulk_upload([])
        env._mock_box.write_file.assert_not_called()
        env._mock_box.exec.assert_not_called()

    def test_bulk_upload_raises_on_extract_failure(self, make_env, tmp_path):
        env = make_env()
        env._mock_box.exec.return_value = _ExecResult(
            exit_code=2, stderr="tar broken"
        )
        f1 = tmp_path / "a.txt"
        f1.write_bytes(b"a")
        with pytest.raises(RuntimeError, match="bulk upload extract failed"):
            env._boxd_bulk_upload([(str(f1), "/root/.hermes/a.txt")])

    def test_bulk_download_tars_and_reads_file(self, make_env, tmp_path):
        env = make_env()
        env._mock_box.exec.reset_mock()
        env._mock_box.exec.return_value = _ExecResult(exit_code=0)
        env._mock_box.read_file.return_value = b"tar bytes here"

        dest = tmp_path / "downloaded.tar"
        env._boxd_bulk_download(dest)

        # First post-reset exec is the tar create command
        first_cmd = env._mock_box.exec.call_args_list[0].args[2]
        assert "tar cf" in first_cmd
        assert "root/.hermes" in first_cmd  # matches both /root/.hermes and root/.hermes

        # read_file pulls the tar back to host
        env._mock_box.read_file.assert_called_once()
        assert dest.read_bytes() == b"tar bytes here"

    def test_delete_uses_quoted_rm(self, make_env):
        env = make_env()
        env._mock_box.exec.reset_mock()
        env._boxd_delete(["/root/.hermes/a", "/root/.hermes/b"])
        cmd = env._mock_box.exec.call_args.args[2]
        assert "rm" in cmd
        assert "/root/.hermes/a" in cmd
        assert "/root/.hermes/b" in cmd


# ---------------------------------------------------------------------------
# _mb_to_size_str (pure function)
# ---------------------------------------------------------------------------


class TestMbToSizeStr:
    def test_zero_returns_empty(self):
        from tools.environments.boxd import _mb_to_size_str
        assert _mb_to_size_str(0) == ""
        assert _mb_to_size_str(-1) == ""

    def test_below_gb_returns_mb(self):
        from tools.environments.boxd import _mb_to_size_str
        assert _mb_to_size_str(100) == "100M"
        assert _mb_to_size_str(1023) == "1023M"

    def test_at_or_above_gb_rounds_up(self):
        from tools.environments.boxd import _mb_to_size_str
        assert _mb_to_size_str(1024) == "1G"
        assert _mb_to_size_str(8192) == "8G"
        # Rounding: 1500 MB -> 2 GB
        assert _mb_to_size_str(1500) == "2G"


# ---------------------------------------------------------------------------
# Login-shell branch
# ---------------------------------------------------------------------------


class TestLoginShell:
    def test_login_true_uses_bash_l_c(self, make_env):
        """login=True must invoke bash -l -c so user's login profile loads."""
        box = _make_box()
        box.exec.side_effect = [
            _ExecResult(stdout="/root"),         # $HOME
            _ExecResult(stdout="", exit_code=0),  # init_session uses login=True too
            _ExecResult(stdout="ok", exit_code=0),
        ]
        env = make_env(box=box)
        # init_session ran with login=True. Its call args:
        init_call = box.exec.call_args_list[1]
        assert init_call.args[:3] == ("bash", "-l", "-c")

    def test_login_false_uses_bash_c(self, make_env):
        """Non-login execute() (snapshot ready) uses bash -c, not bash -l -c."""
        box = _make_box()
        box.exec.side_effect = [
            _ExecResult(stdout="/root"),
            _ExecResult(stdout="", exit_code=0),  # init_session OK -> snapshot_ready
            _ExecResult(stdout="hi", exit_code=0),
        ]
        env = make_env(box=box)
        # snapshot_ready was set; the next execute() should use bash -c
        assert env._snapshot_ready is True
        env.execute("echo hi")
        # Last call (the actual execute) should be bash -c, not bash -l -c
        last_call = box.exec.call_args_list[-1]
        assert last_call.args[0] == "bash"
        assert last_call.args[1] == "-c"


# ---------------------------------------------------------------------------
# init_session: SDK exec error during bootstrap is captured, never raised
# ---------------------------------------------------------------------------


class TestInitSessionFailure:
    def test_sdk_exec_error_during_bootstrap_does_not_crash_init(self, make_env):
        """SDK errors inside init_session's bootstrap exec must be captured by
        _ThreadedProcessHandle, not propagated. Otherwise __init__ explodes
        and the user never gets a usable environment."""
        box = _make_box()
        # $HOME succeeds; init_session bootstrap raises inside the SDK call
        box.exec.side_effect = [
            _ExecResult(stdout="/root"),
            RuntimeError("bootstrap blew up inside SDK"),
        ]
        # Constructor must not raise even if bootstrap exec errors
        env = make_env(box=box)
        assert env._box is not None
        assert env._sync_manager is not None


# ---------------------------------------------------------------------------
# Wake-box on additional statuses
# ---------------------------------------------------------------------------


class TestWakeBoxStatuses:
    def test_paused_status_triggers_resume(self, make_env):
        env = make_env()
        env._box.status = "paused"
        env._box.resume.reset_mock()
        env._wake_box(env._box)
        env._box.resume.assert_called_once()

    def test_starting_status_triggers_resume(self, make_env):
        env = make_env()
        env._box.status = "starting"
        env._box.resume.reset_mock()
        env._wake_box(env._box)
        env._box.resume.assert_called_once()

    def test_empty_status_skips_resume(self, make_env):
        """Box from get() may not populate status; skip the resume RPC and let
        the next exec surface real state."""
        env = make_env()
        env._box.status = ""
        env._box.resume.reset_mock()
        env._wake_box(env._box)
        env._box.resume.assert_not_called()

    def test_resume_failure_swallowed(self, make_env):
        """resume() returning an error on already-running VM must not raise —
        the next exec is the real liveness check."""
        env = make_env()
        env._box.status = "suspended"
        env._box.resume.side_effect = RuntimeError("already running")
        # Should not raise
        env._wake_box(env._box)


# ---------------------------------------------------------------------------
# auto_suspend_timeout edge cases
# ---------------------------------------------------------------------------


class TestAutoSuspendTimeoutEdges:
    def _box_config(self, boxd_sdk):
        return boxd_sdk.Compute.return_value.box.create.call_args.kwargs["config"]

    def test_zero_disables_auto_suspend(self, make_env, boxd_sdk):
        env = make_env(auto_suspend_timeout=0, persistent=False)
        cfg = self._box_config(boxd_sdk)
        # 0 must round-trip exactly (boxd treats 0 as disabled, NOT default)
        assert cfg.lifecycle.auto_suspend_timeout == 0

    def test_very_long_timeout_passes_through(self, make_env, boxd_sdk):
        env = make_env(auto_suspend_timeout=86400, persistent=False)
        cfg = self._box_config(boxd_sdk)
        assert cfg.lifecycle.auto_suspend_timeout == 86400


# ---------------------------------------------------------------------------
# Compute() constructor errors
# ---------------------------------------------------------------------------


class TestComputeInitErrors:
    def test_compute_constructor_error_propagates(self, boxd_sdk, monkeypatch):
        """If the SDK can't authenticate at construction time, the error must
        bubble up cleanly so the user can fix their credentials."""
        monkeypatch.setattr("tools.environments.base.is_interrupted", lambda: False)

        class _AuthErr(Exception):
            pass

        boxd_sdk.Compute = MagicMock(side_effect=_AuthErr("no API key"))
        from tools.environments.boxd import BoxdEnvironment
        with pytest.raises(_AuthErr, match="no API key"):
            BoxdEnvironment(persistent_filesystem=False)


# ---------------------------------------------------------------------------
# Image kwarg behavior
# ---------------------------------------------------------------------------


class TestImageKwarg:
    def test_empty_image_omitted_from_create(self, make_env):
        env = make_env(image="", persistent=False)
        kwargs = env._mock_compute.box.create.call_args.kwargs
        # Must NOT pass image="" — let the server default kick in.
        assert "image" not in kwargs


# ---------------------------------------------------------------------------
# Factory dispatch via terminal_tool._create_environment
# ---------------------------------------------------------------------------


class TestFactoryDispatch:
    def test_create_environment_dispatches_to_boxd(self, boxd_sdk, monkeypatch):
        """terminal_tool._create_environment("boxd", ...) must construct
        BoxdEnvironment with the right kwargs."""
        monkeypatch.setattr("tools.environments.base.is_interrupted", lambda: False)

        # Wire a mock box+compute so __init__ doesn't blow up
        box = _make_box()
        box.exec.return_value = _ExecResult(stdout="/root")
        mock_compute = MagicMock()
        mock_compute.box.create.return_value = box
        mock_compute.box.get.side_effect = boxd_sdk.NotFoundError("not found")
        boxd_sdk.Compute = MagicMock(return_value=mock_compute)

        from tools.terminal_tool import _create_environment
        env = _create_environment(
            env_type="boxd",
            image="ubuntu:22.04",
            cwd="/root",
            timeout=60,
            container_config={
                "container_cpu": 2,
                "container_memory": 8192,
                "container_disk": 102400,
                "container_persistent": False,
            },
            task_id="factorytest",
        )

        from tools.environments.boxd import BoxdEnvironment
        assert isinstance(env, BoxdEnvironment)
        # VM was created with the right name + image
        kwargs = mock_compute.box.create.call_args.kwargs
        assert kwargs["name"] == "hermes-factorytest"
        assert kwargs["image"] == "ubuntu:22.04"
        assert kwargs["config"].vcpu == 2


# ---------------------------------------------------------------------------
# _get_env_config picks up boxd_image from env vars
# ---------------------------------------------------------------------------


class TestEnvConfig:
    def test_default_boxd_image_is_empty(self, monkeypatch):
        monkeypatch.delenv("TERMINAL_BOXD_IMAGE", raising=False)
        monkeypatch.setenv("TERMINAL_ENV", "boxd")
        from tools.terminal_tool import _get_env_config
        cfg = _get_env_config()
        assert cfg["env_type"] == "boxd"
        assert cfg["boxd_image"] == ""

    def test_boxd_image_env_var_override(self, monkeypatch):
        monkeypatch.setenv("TERMINAL_ENV", "boxd")
        monkeypatch.setenv("TERMINAL_BOXD_IMAGE", "my/custom-image:tag")
        from tools.terminal_tool import _get_env_config
        cfg = _get_env_config()
        assert cfg["boxd_image"] == "my/custom-image:tag"
