"""Unit tests for the Daytona cloud sandbox environment backend."""

from datetime import datetime, timedelta, timezone
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Helpers to build mock Daytona SDK objects
# ---------------------------------------------------------------------------

def _make_exec_response(result="", exit_code=0):
    return SimpleNamespace(result=result, exit_code=exit_code)


def _make_sandbox(sandbox_id="sb-123", state="started"):
    sb = MagicMock()
    sb.id = sandbox_id
    sb.state = state
    sb.name = f"hermes-{sandbox_id}"
    sb.labels = {}
    sb.process.exec.return_value = _make_exec_response()
    return sb


def _patch_daytona_imports(monkeypatch):
    """Patch the daytona SDK so DaytonaEnvironment can be imported without it."""
    import types as _types

    import enum

    class _SandboxState(str, enum.Enum):
        STARTED = "started"
        STOPPED = "stopped"
        ARCHIVED = "archived"
        ERROR = "error"

    daytona_mod = _types.ModuleType("daytona")
    daytona_mod.Daytona = MagicMock
    daytona_mod.CreateSandboxFromImageParams = MagicMock(
        name="CreateSandboxFromImageParams"
    )
    daytona_mod.DaytonaError = type("DaytonaError", (Exception,), {})
    daytona_mod.Resources = MagicMock(name="Resources")
    daytona_mod.SandboxState = _SandboxState

    monkeypatch.setitem(__import__("sys").modules, "daytona", daytona_mod)
    return daytona_mod


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def daytona_sdk(monkeypatch):
    """Provide a mock daytona SDK module and return it for assertions."""
    return _patch_daytona_imports(monkeypatch)


@pytest.fixture()
def make_env(daytona_sdk, monkeypatch):
    """Factory that creates a DaytonaEnvironment with a mocked SDK."""
    # Prevent is_interrupted from interfering — patch where it's used (base.py)
    monkeypatch.setattr("tools.environments.base.is_interrupted", lambda: False)
    # Prevent skills/credential sync from consuming mock exec calls
    monkeypatch.setattr("tools.credential_files.get_credential_file_mounts", lambda: [])
    monkeypatch.setattr("tools.credential_files.get_skills_directory_mount", lambda **kw: None)
    monkeypatch.setattr("tools.credential_files.iter_skills_files", lambda **kw: [])

    def _factory(
        sandbox=None,
        get_side_effect=None,
        list_return=None,
        home_dir="/root",
        persistent=True,
        **kwargs,
    ):
        sandbox = sandbox or _make_sandbox()
        # Mock the $HOME detection
        sandbox.process.exec.return_value = _make_exec_response(result=home_dir)

        mock_client = MagicMock()
        mock_client.create.return_value = sandbox

        if get_side_effect is not None:
            mock_client.get.side_effect = get_side_effect
        else:
            # Default: no existing sandbox found via get()
            mock_client.get.side_effect = daytona_sdk.DaytonaError("not found")

        # Default: no legacy sandbox found via list()
        if list_return is not None:
            mock_client.list.return_value = list_return
        else:
            mock_client.list.return_value = iter([])

        daytona_sdk.Daytona = MagicMock(return_value=mock_client)

        from tools.environments.daytona import DaytonaEnvironment

        kwargs.setdefault("disk", 10240)
        env = DaytonaEnvironment(
            image="test-image:latest",
            persistent_filesystem=persistent,
            **kwargs,
        )
        env._mock_client = mock_client  # expose for assertions
        return env

    return _factory


# ---------------------------------------------------------------------------
# Constructor / cwd resolution
# ---------------------------------------------------------------------------

class TestCwdResolution:
    def test_default_cwd_resolves_home(self, make_env):
        env = make_env(home_dir="/home/testuser")
        assert env.cwd == "/home/testuser"

    def test_tilde_cwd_resolves_home(self, make_env):
        env = make_env(cwd="~", home_dir="/home/testuser")
        assert env.cwd == "/home/testuser"

    def test_explicit_cwd_not_overridden(self, make_env):
        env = make_env(cwd="/workspace", home_dir="/root")
        assert env.cwd == "/workspace"

    def test_home_detection_failure_keeps_default_cwd(self, make_env):
        sb = _make_sandbox()
        sb.process.exec.side_effect = RuntimeError("exec failed")
        env = make_env(sandbox=sb)
        assert env.cwd == "/home/daytona"  # keeps constructor default

    def test_empty_home_keeps_default_cwd(self, make_env):
        env = make_env(home_dir="")
        assert env.cwd == "/home/daytona"  # keeps constructor default


# ---------------------------------------------------------------------------
# Sandbox persistence / resume
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_persistent_resumes_same_profile_and_refreshes_owner(self, make_env):
        existing = _make_sandbox(sandbox_id="sb-existing")
        existing.labels = {
            "hermes_task_id": "mytask",
            "hermes_profile": "default",
            "hermes_owner_pid": "99999",
        }
        existing.process.exec.return_value = _make_exec_response(result="/root")
        env = make_env(list_return=iter([existing]), persistent=True,
                       task_id="mytask")
        existing.start.assert_called_once()
        env._mock_client.list.assert_called_once_with(
            labels={"hermes_task_id": "mytask", "hermes_profile": "default"},
            limit=1,
        )
        existing.set_labels.assert_called_once()
        assert existing.set_labels.call_args.args[0]["hermes_owner_pid"].isdigit()
        env._mock_client.create.assert_not_called()

    def test_persistent_does_not_reuse_profileless_legacy_sandbox(self, make_env):
        legacy = _make_sandbox(sandbox_id="sb-legacy")
        legacy.process.exec.return_value = _make_exec_response(result="/root")
        env = make_env(list_return=iter([legacy]), persistent=True, task_id="mytask")
        legacy.start.assert_not_called()
        env._mock_client.create.assert_called_once()
        env._mock_client.list.assert_called_once_with(
            labels={"hermes_task_id": "mytask", "hermes_profile": "default"},
            limit=1,
        )

    def test_persistent_does_not_reuse_cross_profile_sandbox(self, make_env):
        other = _make_sandbox(sandbox_id="sb-other-profile")
        other.labels = {
            "hermes_task_id": "mytask",
            "hermes_profile": "other",
            "hermes_owner_pid": "99999",
        }
        env = make_env(list_return=iter([other]), persistent=True, task_id="mytask")
        other.start.assert_not_called()
        env._mock_client.create.assert_called_once()

    def test_persistent_creates_new_when_none_found(self, make_env, daytona_sdk):
        env = make_env(persistent=True, task_id="mytask")
        env._mock_client.create.assert_called_once()
        params = daytona_sdk.CreateSandboxFromImageParams.call_args.kwargs
        assert params["name"].startswith("hermes-")
        assert params["name"] != "hermes-mytask"
        env._mock_client.list.assert_called_with(
            labels={"hermes_task_id": "mytask", "hermes_profile": "default"},
            limit=1,
        )

    def test_created_sandbox_gets_profile_and_owner_labels(
        self, make_env, daytona_sdk, monkeypatch
    ):
        monkeypatch.setattr(
            "tools.environments.daytona._get_active_profile_name",
            lambda: "research",
        )
        env = make_env(
            get_side_effect=daytona_sdk.DaytonaError("not found"),
            persistent=True,
            task_id="mytask",
        )
        labels = daytona_sdk.CreateSandboxFromImageParams.call_args.kwargs["labels"]
        assert labels["hermes_task_id"] == "mytask"
        assert labels["hermes_profile"] == "research"
        assert labels["hermes_owner_pid"].isdigit()
        env._mock_client.list.assert_called_with(
            labels={"hermes_task_id": "mytask", "hermes_profile": "research"},
            limit=1,
        )

    def test_non_persistent_skips_lookup(self, make_env):
        env = make_env(persistent=False)
        env._mock_client.list.assert_not_called()
        env._mock_client.create.assert_called_once()


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

class TestCleanup:
    def test_persistent_cleanup_stops_sandbox(self, make_env):
        env = make_env(persistent=True)
        sb = env._sandbox
        env.cleanup()
        sb.stop.assert_called_once()

    def test_non_persistent_cleanup_deletes_sandbox(self, make_env):
        env = make_env(persistent=False)
        sb = env._sandbox
        env.cleanup()
        env._mock_client.delete.assert_called_once_with(sb)

    def test_cleanup_idempotent(self, make_env):
        env = make_env(persistent=True)
        env.cleanup()
        env.cleanup()  # should not raise

    def test_cleanup_swallows_errors(self, make_env):
        env = make_env(persistent=True)
        env._sandbox.stop.side_effect = RuntimeError("stop failed")
        env.cleanup()  # should not raise
        assert env._sandbox is None


# ---------------------------------------------------------------------------
# Execute
# ---------------------------------------------------------------------------

class TestExecute:
    def test_basic_command(self, make_env):
        sb = _make_sandbox()
        # Calls: (1) $HOME detection, (2) init_session bootstrap, (3) actual command
        sb.process.exec.side_effect = [
            _make_exec_response(result="/root"),       # $HOME
            _make_exec_response(result="", exit_code=0),  # init_session
            _make_exec_response(result="hello", exit_code=0),  # actual cmd
        ]
        sb.state = "started"
        env = make_env(sandbox=sb)

        result = env.execute("echo hello")
        assert "hello" in result["output"]
        assert result["returncode"] == 0

    def test_sdk_timeout_passed_to_exec(self, make_env):
        """SDK native timeout is passed to sandbox.process.exec()."""
        sb = _make_sandbox()
        sb.process.exec.side_effect = [
            _make_exec_response(result="/root"),
            _make_exec_response(result="", exit_code=0),  # init_session
            _make_exec_response(result="ok", exit_code=0),
        ]
        sb.state = "started"
        env = make_env(sandbox=sb, timeout=42)

        env.execute("echo hello")
        # The exec call should receive timeout= kwarg (SDK native timeout)
        call_args = sb.process.exec.call_args_list[-1]
        assert call_args[1]["timeout"] == 42
        # The command should NOT have a shell `timeout` prefix
        cmd = call_args[0][0]
        assert not cmd.startswith("timeout ")

    def test_timeout_returns_exit_code_124(self, make_env):
        """SDK-level timeout surfaces as exit code 124 via _wait_for_process."""
        sb = _make_sandbox()
        sb.process.exec.side_effect = [
            _make_exec_response(result="/root"),
            _make_exec_response(result="", exit_code=0),  # init_session
            _make_exec_response(result="", exit_code=124),  # actual cmd
        ]
        sb.state = "started"
        env = make_env(sandbox=sb)

        result = env.execute("sleep 300", timeout=5)
        assert result["returncode"] == 124

    def test_nonzero_exit_code(self, make_env):
        sb = _make_sandbox()
        sb.process.exec.side_effect = [
            _make_exec_response(result="/root"),
            _make_exec_response(result="", exit_code=0),  # init_session
            _make_exec_response(result="not found", exit_code=127),
        ]
        sb.state = "started"
        env = make_env(sandbox=sb)

        result = env.execute("bad_cmd")
        assert result["returncode"] == 127

    def test_stdin_data_wraps_heredoc(self, make_env):
        sb = _make_sandbox()
        sb.process.exec.side_effect = [
            _make_exec_response(result="/root"),
            _make_exec_response(result="", exit_code=0),  # init_session
            _make_exec_response(result="ok", exit_code=0),
        ]
        sb.state = "started"
        env = make_env(sandbox=sb)

        env.execute("python3", stdin_data="print('hi')")
        # Check that the command passed to exec contains heredoc markers
        # Base class uses HERMES_STDIN_ prefix for heredoc delimiters
        call_args = sb.process.exec.call_args_list[-1]
        cmd = call_args[0][0]
        assert "HERMES_STDIN_" in cmd
        assert "print" in cmd
        assert "hi" in cmd


    def test_daytona_error_triggers_retry(self, make_env, daytona_sdk):
        sb = _make_sandbox()
        sb.state = "started"
        sb.process.exec.side_effect = [
            _make_exec_response(result="/root"),  # $HOME
            _make_exec_response(result="", exit_code=0),  # init_session
            daytona_sdk.DaytonaError("transient"),  # first attempt fails
            _make_exec_response(result="ok", exit_code=0),  # retry succeeds
        ]
        env = make_env(sandbox=sb)

        result = env.execute("echo retry")
        # DaytonaError now surfaces directly through _ThreadedProcessHandle
        # (no retry logic) — the error becomes returncode=1
        assert result["returncode"] == 1


# ---------------------------------------------------------------------------
# Resource conversion
# ---------------------------------------------------------------------------

class TestResourceConversion:
    def _get_resources_kwargs(self, daytona_sdk):
        return daytona_sdk.Resources.call_args.kwargs

    def test_memory_converted_to_gib(self, make_env, daytona_sdk):
        env = make_env(memory=5120)
        assert self._get_resources_kwargs(daytona_sdk)["memory"] == 5

    def test_disk_converted_to_gib(self, make_env, daytona_sdk):
        env = make_env(disk=10240)
        assert self._get_resources_kwargs(daytona_sdk)["disk"] == 10

    def test_small_values_clamped_to_1(self, make_env, daytona_sdk):
        env = make_env(memory=100, disk=100)
        kw = self._get_resources_kwargs(daytona_sdk)
        assert kw["memory"] == 1
        assert kw["disk"] == 1


# ---------------------------------------------------------------------------
# Ensure sandbox ready
# ---------------------------------------------------------------------------

class TestInterrupt:
    def test_interrupt_stops_sandbox_and_returns_130(self, make_env, monkeypatch):
        sb = _make_sandbox()
        sb.state = "started"
        event = threading.Event()
        calls = {"n": 0}

        def exec_side_effect(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                return _make_exec_response(result="/root")  # $HOME detection
            if calls["n"] == 2:
                return _make_exec_response(result="", exit_code=0)  # init_session
            event.wait(timeout=5)  # simulate long-running command
            return _make_exec_response(result="done", exit_code=0)

        sb.process.exec.side_effect = exec_side_effect
        env = make_env(sandbox=sb)

        # is_interrupted is checked by base.py's _wait_for_process,
        # patch where it's actually referenced (base.py's local binding)
        monkeypatch.setattr(
            "tools.environments.base.is_interrupted", lambda: True
        )
        try:
            result = env.execute("sleep 10")
            assert result["returncode"] == 130
            sb.stop.assert_called()
        finally:
            event.set()


# ---------------------------------------------------------------------------
# DaytonaError surfaces directly (no retry)
# ---------------------------------------------------------------------------

class TestRetryExhausted:
    def test_both_attempts_fail(self, make_env, daytona_sdk):
        """DaytonaError surfaces directly as rc=1 (retry logic was removed)."""
        sb = _make_sandbox()
        sb.state = "started"
        sb.process.exec.side_effect = [
            _make_exec_response(result="/root"),       # $HOME
            _make_exec_response(result="", exit_code=0),  # init_session
            daytona_sdk.DaytonaError("fail1"),         # actual command fails
        ]
        env = make_env(sandbox=sb)

        result = env.execute("echo x")
        # Error surfaces directly through _ThreadedProcessHandle (rc=1)
        assert result["returncode"] == 1


# ---------------------------------------------------------------------------
# Ensure sandbox ready
# ---------------------------------------------------------------------------

class TestEnsureSandboxReady:
    def test_restarts_stopped_sandbox(self, make_env):
        env = make_env()
        env._sandbox.state = "stopped"
        env._ensure_sandbox_ready()
        env._sandbox.start.assert_called()

    def test_no_restart_when_running(self, make_env):
        env = make_env()
        env._sandbox.state = "started"
        env._ensure_sandbox_ready()
        env._sandbox.start.assert_not_called()


# ---------------------------------------------------------------------------
# Sync safety: shell-metacharacter quoting
# ---------------------------------------------------------------------------

class TestSyncSafety:
    def test_single_upload_quotes_parent_path(self, make_env, tmp_path):
        """A remote path with shell metacharacters must be quoted, not injected."""
        env = make_env()
        env._sandbox.process.exec.reset_mock()

        host_file = tmp_path / "token.txt"
        host_file.write_text("secret", encoding="utf-8")
        remote_path = "/root/.hermes/skills/evil; touch /tmp/daytona-owned/file.txt"

        env._daytona_upload(str(host_file), remote_path)

        mkdir_cmd = env._sandbox.process.exec.call_args_list[0][0][0]
        # The whole parent dir is a single quoted argument — the ';' cannot
        # break out into a second command.
        assert mkdir_cmd == (
            "mkdir -p '/root/.hermes/skills/evil; touch /tmp/daytona-owned'"
        )
        assert "; touch" not in mkdir_cmd.replace(
            "'/root/.hermes/skills/evil; touch /tmp/daytona-owned'", ""
        )


# ---------------------------------------------------------------------------
# Orphan reaper
# ---------------------------------------------------------------------------

class TestOrphanReaper:
    def _stale_sandbox(self, task_id="oldtask", profile="default", state="started"):
        sb = _make_sandbox(sandbox_id=f"sb-{task_id}", state=state)
        sb.name = f"hermes-{task_id}"
        sb.labels = {
            "hermes_task_id": task_id,
            "hermes_profile": profile,
            "hermes_owner_pid": "12345",
        }
        sb.created_at = datetime.now(timezone.utc) - timedelta(seconds=1000)
        return sb

    def test_reaper_stops_matching_stale_sandbox(self, monkeypatch):
        from tools.environments import daytona

        stale = self._stale_sandbox()
        client = MagicMock()
        client.list.return_value = iter([stale])
        monkeypatch.setattr(daytona, "_pid_is_alive", lambda pid: False)

        stopped = daytona.reap_orphan_sandboxes(
            max_age_seconds=600,
            current_task_id="current",
            profile_filter="default",
            daytona_client=client,
        )

        assert stopped == 1
        stale.stop.assert_called_once()

    def test_reaper_skips_current_task_sandbox(self, monkeypatch):
        from tools.environments import daytona

        current = self._stale_sandbox(task_id="current")
        client = MagicMock()
        client.list.return_value = iter([current])
        monkeypatch.setattr(daytona, "_pid_is_alive", lambda pid: False)

        stopped = daytona.reap_orphan_sandboxes(
            max_age_seconds=600,
            current_task_id="current",
            profile_filter="default",
            daytona_client=client,
        )

        assert stopped == 0
        current.stop.assert_not_called()

    def test_reaper_skips_recent_sandbox(self, monkeypatch):
        from tools.environments import daytona

        recent = self._stale_sandbox()
        recent.created_at = datetime.now(timezone.utc) - timedelta(seconds=10)
        client = MagicMock()
        client.list.return_value = iter([recent])
        monkeypatch.setattr(daytona, "_pid_is_alive", lambda pid: False)

        stopped = daytona.reap_orphan_sandboxes(
            max_age_seconds=600,
            current_task_id="current",
            profile_filter="default",
            daytona_client=client,
        )

        assert stopped == 0
        recent.stop.assert_not_called()

    def test_reaper_skips_live_owner(self, monkeypatch):
        from tools.environments import daytona

        active = self._stale_sandbox()
        client = MagicMock()
        client.list.return_value = iter([active])
        monkeypatch.setattr(daytona, "_pid_is_alive", lambda pid: True)

        stopped = daytona.reap_orphan_sandboxes(
            max_age_seconds=600,
            current_task_id="current",
            profile_filter="default",
            daytona_client=client,
        )

        assert stopped == 0
        active.stop.assert_not_called()

    @pytest.mark.parametrize("owner_pid", [None, "bad", "0", "-1"])
    def test_reaper_skips_incomplete_owner_metadata(self, owner_pid):
        from tools.environments import daytona

        stale = self._stale_sandbox()
        if owner_pid is None:
            stale.labels.pop("hermes_owner_pid")
        else:
            stale.labels["hermes_owner_pid"] = owner_pid
        client = MagicMock()
        client.list.return_value = iter([stale])
        assert daytona.reap_orphan_sandboxes(
            max_age_seconds=600, current_task_id="current",
            profile_filter="default", daytona_client=client,
        ) == 0
        stale.stop.assert_not_called()

    @pytest.mark.parametrize("profile", [None, "other"])
    def test_reaper_skips_missing_or_mismatched_profile(self, monkeypatch, profile):
        from tools.environments import daytona

        stale = self._stale_sandbox(profile=profile or "default")
        if profile is None:
            stale.labels.pop("hermes_profile")
        client = MagicMock()
        client.list.return_value = iter([stale])
        monkeypatch.setattr(daytona, "_pid_is_alive", lambda pid: False)

        assert daytona.reap_orphan_sandboxes(
            max_age_seconds=600, current_task_id="current",
            profile_filter="default", daytona_client=client,
        ) == 0
        stale.stop.assert_not_called()

    def test_reaper_skips_unknown_pid_state(self, monkeypatch):
        from tools.environments import daytona

        stale = self._stale_sandbox()
        client = MagicMock()
        client.list.return_value = iter([stale])
        monkeypatch.setattr(daytona, "_pid_is_alive", lambda pid: None)

        assert daytona.reap_orphan_sandboxes(
            max_age_seconds=600, current_task_id="current",
            profile_filter="default", daytona_client=client,
        ) == 0
        stale.stop.assert_not_called()

    def test_reaper_skips_missing_task_id(self, monkeypatch):
        from tools.environments import daytona

        stale = self._stale_sandbox()
        stale.labels.pop("hermes_task_id")
        client = MagicMock()
        client.list.return_value = iter([stale])
        monkeypatch.setattr(daytona, "_pid_is_alive", lambda pid: False)

        assert daytona.reap_orphan_sandboxes(
            max_age_seconds=600, current_task_id="current",
            profile_filter="default", daytona_client=client,
        ) == 0
        stale.stop.assert_not_called()

    def test_reaper_requires_profile_filter(self):
        from tools.environments import daytona

        client = MagicMock()
        assert daytona.reap_orphan_sandboxes(daytona_client=client) == 0
        client.list.assert_not_called()

    def test_reaper_swallows_sdk_failures(self):
        from tools.environments import daytona

        client = MagicMock()
        client.list.side_effect = RuntimeError("sdk down")

        stopped = daytona.reap_orphan_sandboxes(
            max_age_seconds=600,
            current_task_id="current",
            profile_filter="default",
            daytona_client=client,
        )

        assert stopped == 0
