"""Regression tests for Daytona expansion features.

These tests cover new Daytona backend capabilities BEFORE implementation:
- Snapshot-based sandbox creation (daytona_create_mode / daytona_snapshot)
- Image-mode backward compatibility when new config keys are unset
- Profile-scoped sandbox names and labels (daytona_name_prefix, daytona_name_scope, daytona_labels)
- Lifecycle intervals (auto_stop, auto_archive, auto_delete) and ephemeral flag
- Environment variables (daytona_env_vars)
- Network parameters (daytona_network_block_all, daytona_network_allow_list)
- Volume mounts (daytona_volume_mounts)
- GPU resources (daytona_gpu)

All tests use SDK mocks — no live Daytona credentials required.

RED/GREEN status:
  These tests are written against the EXPANSION specification.  They are
  expected to FAIL (RED) until the implementation adds the corresponding
  constructor parameters, config/env bridge entries, and terminal_tool wiring.
  The purpose is to document the expected API shape and serve as a gate:
  the implementation branch should make these tests GREEN without modifying them.
"""

import enum
import threading
import types as _types
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers to build mock Daytona SDK objects (reused from existing test file)
# ---------------------------------------------------------------------------

def _make_exec_response(result="", exit_code=0):
    from types import SimpleNamespace
    return SimpleNamespace(result=result, exit_code=exit_code)


def _make_sandbox(sandbox_id="sb-123", state="started"):
    sb = MagicMock()
    sb.id = sandbox_id
    sb.state = state
    sb.process.exec.return_value = _make_exec_response()
    return sb


def _patch_daytona_imports(monkeypatch):
    """Patch the daytona SDK so DaytonaEnvironment can be imported without it."""

    class _SandboxState(str, enum.Enum):
        STARTED = "started"
        STOPPED = "stopped"
        ARCHIVED = "archived"
        ERROR = "error"

    daytona_mod = _types.ModuleType("daytona")
    daytona_mod.Daytona = MagicMock
    daytona_mod.CreateSandboxFromImageParams = MagicMock(name="CreateSandboxFromImageParams")
    daytona_mod.CreateSandboxFromSnapshotParams = MagicMock(name="CreateSandboxFromSnapshotParams")
    daytona_mod.DaytonaError = type("DaytonaError", (Exception,), {})
    daytona_mod.Resources = MagicMock(name="Resources")
    daytona_mod.SandboxState = _SandboxState

    monkeypatch.setitem(__import__("sys").modules, "daytona", daytona_mod)

    # Also mock the volume module in the SDK
    vol_mod = _types.ModuleType("daytona_sdk.common.volume")
    vol_mod.VolumeMount = MagicMock(name="VolumeMount")
    monkeypatch.setitem(__import__("sys").modules, "daytona_sdk", _types.ModuleType("daytona_sdk"))
    monkeypatch.setitem(__import__("sys").modules, "daytona_sdk.common", _types.ModuleType("daytona_sdk.common"))
    monkeypatch.setitem(__import__("sys").modules, "daytona_sdk.common.volume", vol_mod)

    return daytona_mod


@pytest.fixture()
def daytona_sdk(monkeypatch):
    return _patch_daytona_imports(monkeypatch)


@pytest.fixture()
def make_env(daytona_sdk, monkeypatch):
    monkeypatch.setattr("tools.environments.base.is_interrupted", lambda: False)
    # Prevent lazy_deps from trying to pip-install the real daytona SDK
    monkeypatch.setattr("tools.lazy_deps.ensure", lambda *a, **kw: None)
    monkeypatch.setattr("tools.credential_files.get_credential_file_mounts", lambda: [])
    monkeypatch.setattr("tools.credential_files.get_skills_directory_mount", lambda **kw: None)
    monkeypatch.setattr("tools.credential_files.iter_skills_files", lambda **kw: [])
    # Mock _derive_profile_id to return a stable test value
    monkeypatch.setattr("tools.environments.daytona._derive_profile_id",
                        lambda: "abcd1234")

    def _factory(
        sandbox=None,
        get_side_effect=None,
        list_return=None,
        home_dir="/root",
        persistent=True,
        **kwargs,
    ):
        sandbox = sandbox or _make_sandbox()
        sandbox.process.exec.return_value = _make_exec_response(result=home_dir)

        mock_client = MagicMock()
        mock_client.create.return_value = sandbox

        if get_side_effect is not None:
            mock_client.get.side_effect = get_side_effect
        else:
            mock_client.get.side_effect = daytona_sdk.DaytonaError("not found")

        if list_return is not None:
            mock_client.list.return_value = list_return
        else:
            mock_client.list.return_value = iter([])

        daytona_sdk.Daytona = MagicMock(return_value=mock_client)

        from tools.environments.daytona import DaytonaEnvironment

        kwargs.setdefault("image", "test-image:latest")
        kwargs.setdefault("disk", 10240)
        env = DaytonaEnvironment(
            persistent_filesystem=persistent,
            **kwargs,
        )
        env._mock_client = mock_client
        return env

    return _factory


# ---------------------------------------------------------------------------
# Snapshot mode tests (daytona_create_mode / daytona_snapshot)
# ---------------------------------------------------------------------------

class TestSnapshotMode:
    """When create_mode='snapshot', DaytonaEnvironment should use
    CreateSandboxFromSnapshotParams instead of CreateSandboxFromImageParams."""

    def test_create_mode_snapshot_uses_snapshot_params(self, make_env, daytona_sdk):
        """Passing create_mode='snapshot' should invoke daytona.create()
        with a CreateSandboxFromSnapshotParams instance."""
        env = make_env(create_mode="snapshot", snapshot="my-snap-001")
        # Inspect CreateSandboxFromSnapshotParams.call_args.kwargs
        # to verify the snapshot param was passed.
        call_kw = daytona_sdk.CreateSandboxFromSnapshotParams.call_args.kwargs
        assert call_kw.get("snapshot") == "my-snap-001", (
            "Expected create_mode='snapshot' to pass a snapshot name to SDK params"
        )

    def test_create_mode_snapshot_omits_image(self, make_env, daytona_sdk):
        """Snapshot mode should NOT set the image field on create params."""
        env = make_env(create_mode="snapshot", snapshot="my-snap-001")
        # Inspect CreateSandboxFromSnapshotParams.call_args.kwargs;
        # image should be absent or None in snapshot params.
        call_kw = daytona_sdk.CreateSandboxFromSnapshotParams.call_args.kwargs
        assert call_kw.get("image") in (None, ""), (
            "Snapshot mode should not set an image on create params"
        )

    def test_create_mode_image_is_default(self, make_env, daytona_sdk):
        """Default create_mode should be 'image' (backward compatible)."""
        env = make_env(image="python:3.11")
        # Verify CreateSandboxFromImageParams was called (not
        # CreateSandboxFromSnapshotParams) by checking its call_args.
        call_kw = daytona_sdk.CreateSandboxFromImageParams.call_args.kwargs
        assert call_kw.get("image") == "python:3.11", (
            "Default create mode should produce image params"
        )

    # --- validation and error handling ---

    def test_create_mode_snapshot_without_snapshot_raises(self, make_env, daytona_sdk):
        """create_mode='snapshot' with empty snapshot must raise ValueError."""
        with pytest.raises(ValueError, match="snapshot"):
            make_env(create_mode="snapshot", snapshot="")

    def test_create_mode_snapshot_none_snapshot_raises(self, make_env, daytona_sdk):
        """create_mode='snapshot' with snapshot=None must raise ValueError."""
        with pytest.raises(ValueError, match="snapshot"):
            make_env(create_mode="snapshot", snapshot=None)

    def test_create_mode_invalid_raises(self, make_env, daytona_sdk):
        """Invalid create_mode (e.g. 'docker') must raise ValueError."""
        with pytest.raises(ValueError, match="Invalid.*create_mode"):
            make_env(create_mode="docker")

    def test_create_mode_empty_string_raises(self, make_env, daytona_sdk):
        """Empty string create_mode must raise ValueError."""
        with pytest.raises(ValueError, match="Invalid.*create_mode"):
            make_env(create_mode="")
class TestValidationBeforeSideEffects:
    """Regression: mode/snapshot validation must fire BEFORE any SDK
    resume/list/create/start calls. If validation fails, the Daytona
    SDK must not have been called at all — no Daytona(), get(), list(),
    or create() side effects.
    """

    def test_invalid_create_mode_raises_before_sdk(self, daytona_sdk, monkeypatch):
        """Invalid create_mode must raise ValueError without calling
        Daytona(), get(), list(), or create()."""
        monkeypatch.setattr("tools.environments.base.is_interrupted", lambda: False)
        monkeypatch.setattr("tools.lazy_deps.ensure", lambda *a, **kw: None)
        monkeypatch.setattr("tools.environments.daytona._derive_profile_id",
                            lambda: "abcd1234")

        # Spy on Daytona() to verify it is never called when validation fails
        daytona_spy = MagicMock()
        monkeypatch.setitem(__import__("sys").modules, "daytona", daytona_sdk)
        # Replace Daytona class with a spy that tracks calls
        daytona_sdk.Daytona = daytona_spy

        from tools.environments.daytona import DaytonaEnvironment
        with pytest.raises(ValueError, match="Invalid.*create_mode"):
            DaytonaEnvironment(
                image="test-image:latest",
                create_mode="docker",
                persistent_filesystem=True,
            )

        # Daytona() constructor must not have been called
        daytona_spy.assert_not_called()

    def test_snapshot_empty_snapshot_raises_before_sdk(self, daytona_sdk, monkeypatch):
        """create_mode='snapshot' with empty snapshot must raise ValueError
        without calling Daytona()."""
        monkeypatch.setattr("tools.environments.base.is_interrupted", lambda: False)
        monkeypatch.setattr("tools.lazy_deps.ensure", lambda *a, **kw: None)
        monkeypatch.setattr("tools.environments.daytona._derive_profile_id",
                            lambda: "abcd1234")

        daytona_spy = MagicMock()
        monkeypatch.setitem(__import__("sys").modules, "daytona", daytona_sdk)
        daytona_sdk.Daytona = daytona_spy

        from tools.environments.daytona import DaytonaEnvironment
        with pytest.raises(ValueError, match="snapshot"):
            DaytonaEnvironment(
                image="test-image:latest",
                create_mode="snapshot",
                snapshot="",
                persistent_filesystem=True,
            )

        daytona_spy.assert_not_called()

    def test_snapshot_none_snapshot_raises_before_sdk(self, daytona_sdk, monkeypatch):
        """create_mode='snapshot' with snapshot=None must raise ValueError
        without calling Daytona()."""
        monkeypatch.setattr("tools.environments.base.is_interrupted", lambda: False)
        monkeypatch.setattr("tools.lazy_deps.ensure", lambda *a, **kw: None)
        monkeypatch.setattr("tools.environments.daytona._derive_profile_id",
                            lambda: "abcd1234")

        daytona_spy = MagicMock()
        monkeypatch.setitem(__import__("sys").modules, "daytona", daytona_sdk)
        daytona_sdk.Daytona = daytona_spy

        from tools.environments.daytona import DaytonaEnvironment
        with pytest.raises(ValueError, match="snapshot"):
            DaytonaEnvironment(
                image="test-image:latest",
                create_mode="snapshot",
                snapshot=None,
                persistent_filesystem=True,
            )

        daytona_spy.assert_not_called()


# ---------------------------------------------------------------------------
# Snapshot-mode shared fields
# ---------------------------------------------------------------------------

class TestSnapshotModeSharedFields:
    """Verify that all shared fields (labels, env_vars, language, network,
    lifecycle, volumes) are forwarded correctly when create_mode='snapshot',
    and that resources/GPU are intentionally omitted from snapshot params.

    In snapshot mode, resources are NOT passed to
    CreateSandboxFromSnapshotParams because the Daytona SDK (>=0.155.0)
    silently drops `resources` from model_dump(), and snapshot-owned
    resources take precedence. The DaytonaEnvironment constructor still
    computes a Resources object internally, but it is filtered out before
    constructing snapshot params.
    """

    def test_snapshot_labels_forwarded(self, make_env, daytona_sdk, monkeypatch):
        """Labels should be forwarded in snapshot mode just like image mode."""
        monkeypatch.setattr("tools.environments.daytona._derive_profile_id",
                            lambda: "abcd1234")
        env = make_env(
            create_mode="snapshot", snapshot="snap-001",
            labels={"team": "infra"},
        )
        call_kw = daytona_sdk.CreateSandboxFromSnapshotParams.call_args.kwargs
        labels = call_kw.get("labels", {})
        assert labels.get("hermes_task_id") == "default"
        assert labels.get("hermes_profile_id") == "abcd1234"
        assert labels.get("hermes_backend") == "daytona"
        assert labels.get("team") == "infra"

    def test_snapshot_env_vars_forwarded(self, make_env, daytona_sdk):
        """env_vars should be forwarded in snapshot mode."""
        env = make_env(
            create_mode="snapshot", snapshot="snap-001",
            env_vars={"DEBUG": "1", "STAGE": "prod"},
        )
        call_kw = daytona_sdk.CreateSandboxFromSnapshotParams.call_args.kwargs
        assert call_kw.get("env_vars") == {"DEBUG": "1", "STAGE": "prod"}

    def test_snapshot_language_forwarded(self, make_env, daytona_sdk):
        """language should be forwarded in snapshot mode."""
        env = make_env(
            create_mode="snapshot", snapshot="snap-001",
            language="python",
        )
        call_kw = daytona_sdk.CreateSandboxFromSnapshotParams.call_args.kwargs
        assert call_kw.get("language") == "python"

    def test_snapshot_network_params_forwarded(self, make_env, daytona_sdk):
        """Network params should be forwarded in snapshot mode."""
        env = make_env(
            create_mode="snapshot", snapshot="snap-001",
            network_block_all=True,
            network_allow_list="10.0.0.0/8",
        )
        call_kw = daytona_sdk.CreateSandboxFromSnapshotParams.call_args.kwargs
        assert call_kw.get("network_block_all") is True
        assert call_kw.get("network_allow_list") == "10.0.0.0/8"

    def test_snapshot_lifecycle_interval_forwarded(self, make_env, daytona_sdk):
        """Lifecycle intervals should be forwarded in snapshot mode."""
        env = make_env(
            create_mode="snapshot", snapshot="snap-001",
            auto_stop_interval=60,
            auto_archive_interval=1440,
        )
        call_kw = daytona_sdk.CreateSandboxFromSnapshotParams.call_args.kwargs
        assert call_kw.get("auto_stop_interval") == 60
        assert call_kw.get("auto_archive_interval") == 1440

    def test_snapshot_volume_mounts_forwarded(self, make_env, daytona_sdk):
        """Volume mounts should be forwarded in snapshot mode."""
        volumes = [
            {"volume_id": "vol-001", "mount_path": "/data"},
        ]
        env = make_env(
            create_mode="snapshot", snapshot="snap-001",
            volume_mounts=volumes,
        )
        call_kw = daytona_sdk.CreateSandboxFromSnapshotParams.call_args.kwargs
        param_volumes = call_kw.get("volumes")
        assert param_volumes is not None, "volumes should be set on snapshot params"
        assert len(param_volumes) == 1

    def test_snapshot_ephemeral_forwarded(self, make_env, daytona_sdk):
        """ephemeral=True should be forwarded in snapshot mode."""
        env = make_env(
            create_mode="snapshot", snapshot="snap-001",
            ephemeral=True,
        )
        call_kw = daytona_sdk.CreateSandboxFromSnapshotParams.call_args.kwargs
        assert call_kw.get("ephemeral") is True
        assert call_kw.get("auto_delete_interval") == 0

    def test_snapshot_resources_not_passed_to_snapshot_params(self, make_env, daytona_sdk):
        """Resources should NOT be passed to CreateSandboxFromSnapshotParams.

        The Daytona SDK (>=0.155.0) does not expose a `resources` field on
        CreateSandboxFromSnapshotParams — it silently drops `resources` from
        model_dump(). Snapshot-owned resources take precedence, so we omit
        `resources` from snapshot params to avoid silently passing a field the
        SDK ignores.
        """
        env = make_env(
            create_mode="snapshot", snapshot="snap-001",
            cpu=4, memory=8192, disk=10240,
        )
        snap_kw = daytona_sdk.CreateSandboxFromSnapshotParams.call_args.kwargs
        assert "resources" not in snap_kw, (
            "resources should NOT be passed to CreateSandboxFromSnapshotParams; "
            "snapshot-owned resources take precedence in daytona>=0.155.0"
        )

    def test_snapshot_name_forwarded(self, make_env, daytona_sdk, monkeypatch):
        """Sandbox name (from name_prefix/name_scope) should be forwarded
        in snapshot mode."""
        monkeypatch.setattr("tools.environments.daytona._derive_profile_id",
                            lambda: "abcd1234")
        env = make_env(
            create_mode="snapshot", snapshot="snap-001",
            task_id="snap-task",
            name_prefix="ci",
        )
        call_kw = daytona_sdk.CreateSandboxFromSnapshotParams.call_args.kwargs
        assert call_kw.get("name") == "ci-snap-task"

    def test_snapshot_resources_omitted_with_gpu(self, make_env, daytona_sdk):
        """Snapshot params must omit resources even when gpu is set.

        The Resources object is constructed internally (gpu=2 is stored on
        it), but resources is intentionally excluded from
        CreateSandboxFromSnapshotParams because snapshot-owned resources take
        precedence in daytona>=0.155.0.
        """
        env = make_env(
            create_mode="snapshot", snapshot="snap-001",
            gpu=2,
        )
        snap_kw = daytona_sdk.CreateSandboxFromSnapshotParams.call_args.kwargs
        assert "resources" not in snap_kw, (
            "resources should NOT be passed to CreateSandboxFromSnapshotParams "
            "even when gpu is set; snapshot-owned resources take precedence in "
            "daytona>=0.155.0"
        )


# ---------------------------------------------------------------------------
# Image-mode backward compatibility
# ---------------------------------------------------------------------------

class TestImageModeBackwardCompat:
    """Ensure existing image-based Daytona behavior is preserved when
    new expansion keys are NOT set."""

    def test_image_mode_default_no_extra_params(self, make_env, daytona_sdk, monkeypatch):
        """When no expansion keys are set, CreateSandboxFromImageParams
        should have the same shape as before (image, name, labels, resources,
        auto_stop_interval=0) plus mandatory labels."""
        monkeypatch.setattr("tools.environments.daytona._derive_profile_id",
                            lambda: "abcd1234")
        env = make_env(image="test-image:latest", task_id="compat-task")
        # Inspect the kwargs passed to CreateSandboxFromImageParams, not
        # attribute access on the MagicMock instance it returns.
        call_kw = daytona_sdk.CreateSandboxFromImageParams.call_args.kwargs

        # Existing invariants that must not regress:
        assert call_kw.get("image") == "test-image:latest"
        assert call_kw.get("name") == "hermes-compat-task"
        assert call_kw.get("auto_stop_interval") == 0
        labels = call_kw.get("labels", {})
        assert labels.get("hermes_task_id") == "compat-task"
        # Mandatory labels:
        assert labels.get("hermes_profile_id") == "abcd1234"
        assert labels.get("hermes_backend") == "daytona"

    def test_resources_still_passed_in_image_mode(self, make_env, daytona_sdk):
        """cpu, memory, disk should still be passed via Resources in image mode."""
        env = make_env(cpu=2, memory=4096, disk=10240)
        call_args = daytona_sdk.Resources.call_args
        kw = call_args.kwargs if call_args else {}
        assert kw.get("cpu") == 2, f"cpu={kw.get('cpu')}"
        assert kw.get("memory") == 4, f"memory GiB={kw.get('memory')}"
        assert kw.get("disk") == 10, f"disk GiB={kw.get('disk')}"


# ---------------------------------------------------------------------------
# Profile-scoped names and labels
# ---------------------------------------------------------------------------

class TestProfileScopedNamesLabels:
    """daytona_name_prefix, daytona_name_scope, daytona_labels.
    
    All sandboxes carry three mandatory labels:
    - hermes_task_id:    the task ID
    - hermes_profile_id: short SHA-256 hash of the profile home dir
    - hermes_backend:    always 'daytona' (identifies the sandbox backend)
    """

    def test_default_name_prefix_is_hermes(self, make_env, daytona_sdk, monkeypatch):
        """Without name_prefix, sandbox name should be 'hermes-{task_id}'."""
        monkeypatch.setattr("tools.environments.daytona._derive_profile_id",
                            lambda: "abcd1234")
        env = make_env(task_id="task1")
        env._mock_client.get.assert_called_with("hermes-task1")

    def test_custom_name_prefix(self, make_env, daytona_sdk, monkeypatch):
        """With name_prefix='myapp', sandbox name should be 'myapp-{task_id}'."""
        monkeypatch.setattr("tools.environments.daytona._derive_profile_id",
                            lambda: "abcd1234")
        env = make_env(task_id="task2", name_prefix="myapp")
        env._mock_client.get.assert_called_with("myapp-task2")

    def test_name_scope_global_uses_prefix_only(self, make_env, daytona_sdk, monkeypatch):
        """name_scope='global' should produce prefix-only names (no task_id)."""
        monkeypatch.setattr("tools.environments.daytona._derive_profile_id",
                            lambda: "abcd1234")
        env = make_env(name_prefix="ci", name_scope="global")
        env._mock_client.get.assert_called_with("ci")

    def test_name_scope_task_uses_prefix_and_task_id(self, make_env, daytona_sdk, monkeypatch):
        """name_scope='task' (default) should produce '{prefix}-{task_id}'."""
        monkeypatch.setattr("tools.environments.daytona._derive_profile_id",
                            lambda: "abcd1234")
        env = make_env(task_id="t5", name_prefix="dev", name_scope="task")
        env._mock_client.get.assert_called_with("dev-t5")

    def test_name_scope_profile_includes_profile_id(self, make_env, daytona_sdk, monkeypatch):
        """name_scope='profile' should produce '{prefix}-{profile_id}-{task_id}'."""
        monkeypatch.setattr("tools.environments.daytona._derive_profile_id",
                            lambda: "a1b2c3d4")
        env = make_env(task_id="t99", name_prefix="dev", name_scope="profile")
        env._mock_client.get.assert_called_with("dev-a1b2c3d4-t99")

    def test_name_scope_legacy_uses_hermes_prefix(self, make_env, daytona_sdk, monkeypatch):
        """name_scope='legacy' should produce exact 'hermes-{task_id}'
        regardless of name_prefix for backward compatibility."""
        monkeypatch.setattr("tools.environments.daytona._derive_profile_id",
                            lambda: "abcd1234")
        env = make_env(task_id="abc123", name_prefix="custom", name_scope="legacy")
        # legacy always uses 'hermes-', ignoring name_prefix
        env._mock_client.get.assert_called_with("hermes-abc123")

    def test_default_labels_include_profile_id_and_backend(
        self, make_env, daytona_sdk, monkeypatch
    ):
        """Default labels must include hermes_task_id, hermes_profile_id,
        and hermes_backend."""
        monkeypatch.setattr("tools.environments.daytona._derive_profile_id",
                            lambda: "feedbeef")
        env = make_env(task_id="lbl-default")
        call_kw = daytona_sdk.CreateSandboxFromImageParams.call_args.kwargs
        merged = call_kw.get("labels", {})
        assert merged.get("hermes_task_id") == "lbl-default"
        assert merged.get("hermes_profile_id") == "feedbeef"
        assert merged.get("hermes_backend") == "daytona"

    def test_extra_labels_merged(self, make_env, daytona_sdk, monkeypatch):
        """daytona_labels should merge with default hermes_task_id,
        hermes_profile_id, and hermes_backend labels."""
        monkeypatch.setattr("tools.environments.daytona._derive_profile_id",
                            lambda: "abcd1234")
        env = make_env(
            task_id="lbl-test",
            labels={"team": "infra", "env": "staging"},
        )
        # Inspect kwargs passed to CreateSandboxFromImageParams constructor.
        call_kw = daytona_sdk.CreateSandboxFromImageParams.call_args.kwargs
        merged = call_kw.get("labels", {})
        assert merged.get("hermes_task_id") == "lbl-test", (
            "Default hermes_task_id label must be preserved"
        )
        assert merged.get("hermes_profile_id") == "abcd1234", (
            "Default hermes_profile_id label must be preserved"
        )
        assert merged.get("hermes_backend") == "daytona", (
            "Default hermes_backend label must be preserved"
        )
        assert merged.get("team") == "infra"
        assert merged.get("env") == "staging"

    def test_user_labels_cannot_override_reserved_defaults(self, make_env, daytona_sdk, monkeypatch):
        """User-provided labels cannot override reserved hermes_* labels."""
        monkeypatch.setattr("tools.environments.daytona._derive_profile_id",
                            lambda: "abcd1234")
        env = make_env(
            task_id="override-test",
            labels={"hermes_task_id": "custom", "hermes_backend": "custom"},
        )
        call_kw = daytona_sdk.CreateSandboxFromImageParams.call_args.kwargs
        merged = call_kw.get("labels", {})
        assert merged.get("hermes_task_id") == "override-test"
        assert merged.get("hermes_backend") == "daytona"
        assert merged.get("hermes_profile_id") == "abcd1234"

    def test_invalid_name_scope_rejected(self, make_env):
        """Typos in daytona_name_scope should fail closed."""
        with pytest.raises(ValueError, match="daytona_name_scope"):
            make_env(name_scope="typo")


# ---------------------------------------------------------------------------
# Lifecycle intervals
# ---------------------------------------------------------------------------

class TestLifecycleIntervals:
    """auto_stop_interval, auto_archive_interval, auto_delete_interval,
    and ephemeral flag."""

    def test_auto_stop_interval(self, make_env, daytona_sdk):
        """Passing auto_stop_interval should forward it to create params."""
        env = make_env(auto_stop_interval=30)
        # Inspect kwargs passed to CreateSandboxFromImageParams constructor.
        call_kw = daytona_sdk.CreateSandboxFromImageParams.call_args.kwargs
        assert call_kw.get("auto_stop_interval") == 30

    def test_auto_archive_interval(self, make_env, daytona_sdk):
        """Passing auto_archive_interval should forward it to create params."""
        env = make_env(auto_archive_interval=1440)
        # Inspect kwargs passed to CreateSandboxFromImageParams constructor.
        call_kw = daytona_sdk.CreateSandboxFromImageParams.call_args.kwargs
        assert call_kw.get("auto_archive_interval") == 1440

    def test_auto_delete_interval(self, make_env, daytona_sdk):
        """Passing auto_delete_interval should forward it to create params."""
        env = make_env(auto_delete_interval=60)
        # Inspect kwargs passed to CreateSandboxFromImageParams constructor.
        call_kw = daytona_sdk.CreateSandboxFromImageParams.call_args.kwargs
        assert call_kw.get("auto_delete_interval") == 60

    def test_ephemeral_sets_auto_delete_to_zero(self, make_env, daytona_sdk):
        """ephemeral=True should set auto_delete_interval=0 on create params."""
        env = make_env(ephemeral=True)
        # Inspect kwargs passed to CreateSandboxFromImageParams constructor.
        call_kw = daytona_sdk.CreateSandboxFromImageParams.call_args.kwargs
        assert call_kw.get("auto_delete_interval") == 0

    def test_ephemeral_forwards_to_sdk_image_params(self, make_env, daytona_sdk):
        """ephemeral=True must appear as ephemeral=True in SDK create params.

        The Daytona SDK exposes ephemeral: bool | None on
        CreateSandboxFromImageParams.  The config key daytona_ephemeral is
        accepted through terminal_tool.py and wired into the DaytonaEnvironment
        constructor, but the constructor must also forward ephemeral=True to
        the SDK create params so that the sandbox is actually created in
        ephemeral mode (rather than just setting auto_delete_interval=0).
        """
        env = make_env(ephemeral=True)
        call_kw = daytona_sdk.CreateSandboxFromImageParams.call_args.kwargs
        assert call_kw.get("ephemeral") is True, (
            "ephemeral=True must be forwarded to CreateSandboxFromImageParams"
        )

    def test_ephemeral_forwards_to_sdk_snapshot_params(self, make_env, daytona_sdk):
        """ephemeral=True must appear in snapshot-mode SDK params too."""
        env = make_env(create_mode="snapshot", snapshot="snap-001", ephemeral=True)
        call_kw = daytona_sdk.CreateSandboxFromSnapshotParams.call_args.kwargs
        assert call_kw.get("ephemeral") is True, (
            "ephemeral=True must be forwarded to CreateSandboxFromSnapshotParams"
        )

    def test_ephemeral_false_not_in_params(self, make_env, daytona_sdk):
        """ephemeral=False (default) should NOT add ephemeral to create params."""
        env = make_env(ephemeral=False)
        call_kw = daytona_sdk.CreateSandboxFromImageParams.call_args.kwargs
        assert "ephemeral" not in call_kw, (
            "ephemeral=False should not appear in SDK create params"
        )

    def test_default_auto_stop_is_zero(self, make_env, daytona_sdk):
        """Without auto_stop_interval, the current default is 0 (never auto-stop).
        This preserves backward compatibility with the existing behavior where
        auto_stop_interval=0 is passed explicitly."""
        env = make_env()
        # Inspect kwargs passed to CreateSandboxFromImageParams constructor.
        call_kw = daytona_sdk.CreateSandboxFromImageParams.call_args.kwargs
        assert call_kw.get("auto_stop_interval") == 0


# ---------------------------------------------------------------------------
# Environment variables
# ---------------------------------------------------------------------------

class TestEnvVars:
    """daytona_env_vars should inject env vars into sandbox creation."""

    def test_env_vars_passed_to_create(self, make_env, daytona_sdk):
        """env_vars should be forwarded to CreateSandboxParams."""
        env = make_env(env_vars={"DEBUG": "1", "NODE_ENV": "test"})
        # Inspect kwargs passed to CreateSandboxFromImageParams constructor.
        call_kw = daytona_sdk.CreateSandboxFromImageParams.call_args.kwargs
        assert call_kw.get("env_vars") == {"DEBUG": "1", "NODE_ENV": "test"}

    def test_env_vars_empty_by_default(self, make_env, daytona_sdk):
        """Without env_vars, the SDK should not receive env_vars."""
        env = make_env()
        # Inspect kwargs passed to CreateSandboxFromImageParams constructor;
        # env_vars should either be absent or None when not explicitly set.
        call_kw = daytona_sdk.CreateSandboxFromImageParams.call_args.kwargs
        assert call_kw.get("env_vars") in (None, {})


# ---------------------------------------------------------------------------
# Network parameters
# ---------------------------------------------------------------------------

class TestNetworkParams:
    """daytona_network_block_all and daytona_network_allow_list."""

    def test_network_block_all_true(self, make_env, daytona_sdk):
        """network_block_all=True should forward to create params."""
        env = make_env(network_block_all=True)
        # Inspect kwargs passed to CreateSandboxFromImageParams constructor.
        call_kw = daytona_sdk.CreateSandboxFromImageParams.call_args.kwargs
        assert call_kw.get("network_block_all") is True

    def test_network_block_all_default_false(self, make_env, daytona_sdk):
        """Default network_block_all should be None (SDK default: no blocking)."""
        env = make_env()
        # Inspect kwargs passed to CreateSandboxFromImageParams constructor;
        # network_block_all should be absent from kwargs when not set.
        call_kw = daytona_sdk.CreateSandboxFromImageParams.call_args.kwargs
        assert "network_block_all" not in call_kw or call_kw.get("network_block_all") is None

    def test_network_allow_list(self, make_env, daytona_sdk):
        """network_allow_list should forward to create params."""
        env = make_env(network_allow_list="10.0.0.0/8,172.16.0.0/12")
        # Inspect kwargs passed to CreateSandboxFromImageParams constructor.
        call_kw = daytona_sdk.CreateSandboxFromImageParams.call_args.kwargs
        assert call_kw.get("network_allow_list") == "10.0.0.0/8,172.16.0.0/12"


# ---------------------------------------------------------------------------
# Volume mounts
# ---------------------------------------------------------------------------

class TestVolumeMounts:
    """daytona_volume_mounts should forward VolumeMount params."""

    def test_volume_mounts_passed_to_create(self, make_env, daytona_sdk):
        """volume_mounts list should be forwarded to create params."""
        volumes = [
            {"volume_id": "vol-123", "mount_path": "/data"},
            {"volume_id": "vol-456", "mount_path": "/cache", "subpath": "npm"},
        ]
        env = make_env(volume_mounts=volumes)
        # Inspect kwargs passed to CreateSandboxFromImageParams constructor.
        call_kw = daytona_sdk.CreateSandboxFromImageParams.call_args.kwargs
        param_volumes = call_kw.get("volumes")
        assert param_volumes is not None, "volumes should be set on create params"
        assert len(param_volumes) == 2

    def test_volume_mounts_empty_by_default(self, make_env, daytona_sdk):
        """Without volume_mounts, create params should not set volumes."""
        env = make_env()
        # Inspect kwargs passed to CreateSandboxFromImageParams constructor;
        # volumes should be absent or empty when not set.
        call_kw = daytona_sdk.CreateSandboxFromImageParams.call_args.kwargs
        assert call_kw.get("volumes") in (None, [])

    def test_missing_volume_mount_class_fails_closed(self, make_env, monkeypatch):
        """Do not pass raw dict volumes if the SDK VolumeMount model is unavailable."""
        from tools.environments import daytona as daytona_env

        real_import = daytona_env.importlib.import_module

        def fake_import_module(module_name):
            if module_name in {"daytona.common.volume", "daytona_sdk.common.volume"}:
                raise ImportError(module_name)
            return real_import(module_name)

        monkeypatch.setattr(daytona_env.importlib, "import_module", fake_import_module)

        with pytest.raises(ImportError, match="VolumeMount"):
            make_env(volume_mounts=[{"volume_id": "vol-123", "mount_path": "/data"}])


# ---------------------------------------------------------------------------
# GPU resources
# ---------------------------------------------------------------------------

class TestGPUResources:
    """daytona_gpu should forward GPU count to Resources."""

    def test_gpu_passed_to_resources(self, make_env, daytona_sdk):
        """Passing gpu=1 should include gpu in the Resources call."""
        env = make_env(gpu=1)
        call_args = daytona_sdk.Resources.call_args
        kw = call_args.kwargs if call_args else {}
        assert kw.get("gpu") == 1, f"gpu={kw.get('gpu')}"

    def test_gpu_default_not_set(self, make_env, daytona_sdk):
        """Without gpu, Resources should not include gpu."""
        env = make_env()
        call_args = daytona_sdk.Resources.call_args
        kw = call_args.kwargs if call_args else {}
        # Resources(cpu=1, memory=5, disk=10) - no gpu
        assert "gpu" not in kw or kw.get("gpu") is None

    def test_gpu_forwards_auto_delete_zero_for_daytona_requirement(self, make_env, daytona_sdk):
        """GPU image-mode sandboxes must explicitly set auto_delete_interval=0.

        Live Daytona validation rejects GPU sandbox creation with
        "GPU sandboxes must be ephemeral - set autoDeleteInterval to 0" when
        the field is omitted, even though 0 is Hermes' default.  Forwarding the
        explicit zero keeps terminal.daytona_gpu from failing before capacity
        or runtime GPU visibility can be tested.
        """
        env = make_env(gpu=1)
        call_kw = daytona_sdk.CreateSandboxFromImageParams.call_args.kwargs
        assert call_kw.get("auto_delete_interval") == 0

    def test_gpu_forces_auto_delete_zero_even_when_configured_nonzero(self, make_env, daytona_sdk):
        """GPU requests override non-zero auto_delete_interval to Daytona-required zero."""
        env = make_env(gpu=1, auto_delete_interval=5)
        call_kw = daytona_sdk.CreateSandboxFromImageParams.call_args.kwargs
        assert call_kw.get("auto_delete_interval") == 0


# ---------------------------------------------------------------------------
# Language parameter
# ---------------------------------------------------------------------------

class TestLanguageParam:
    """daytona_language should be forwarded to the SDK create params."""

    def test_language_passed_to_create(self, make_env, daytona_sdk):
        """Passing language='python' should set it on CreateSandboxParams."""
        env = make_env(language="python")
        # Inspect kwargs passed to CreateSandboxFromImageParams constructor.
        call_kw = daytona_sdk.CreateSandboxFromImageParams.call_args.kwargs
        assert call_kw.get("language") == "python"

    def test_language_default_none(self, make_env, daytona_sdk):
        """Without language, SDK uses its own default (python)."""
        env = make_env()
        # Inspect kwargs passed to CreateSandboxFromImageParams constructor;
        # language should be absent or None when not explicitly set.
        call_kw = daytona_sdk.CreateSandboxFromImageParams.call_args.kwargs
        assert call_kw.get("language") is None or "language" not in call_kw


# ---------------------------------------------------------------------------
# JSON type validation for dict/list env vars
# ---------------------------------------------------------------------------

class TestJsonTypeValidation:
    """_get_env_config() must validate that dict/list Daytona JSON settings
    receive the correct JSON shape.

    TERMINAL_DAYTONA_LABELS must be a JSON object (dict), not a list.
    TERMINAL_DAYTONA_ENV_VARS must be a JSON object (dict), not a list.
    TERMINAL_DAYTONA_VOLUME_MOUNTS must be a JSON list, not a dict.

    Failing to validate means valid-but-wrong JSON (like passing "[]"
    for labels) silently produces incorrect runtime behavior instead of
    a clear, actionable error message.
    """

    def test_labels_rejects_list(self, monkeypatch):
        """TERMINAL_DAYTONA_LABELS=[] is valid JSON but wrong shape (should be dict)."""
        import json
        monkeypatch.setenv("TERMINAL_DAYTONA_LABELS", "[]")
        monkeypatch.setenv("TERMINAL_ENV", "daytona")
        # Force re-import to pick up the patched env
        import importlib
        import tools.terminal_tool as tt
        importlib.reload(tt)
        with pytest.raises(ValueError, match="TERMINAL_DAYTONA_LABELS.*dict"):
            tt._get_env_config()

    def test_labels_rejects_string(self, monkeypatch):
        """TERMINAL_DAYTONA_LABELS='\"hello\"' is valid JSON but not a dict."""
        monkeypatch.setenv("TERMINAL_DAYTONA_LABELS", '"hello"')
        monkeypatch.setenv("TERMINAL_ENV", "daytona")
        import importlib
        import tools.terminal_tool as tt
        importlib.reload(tt)
        with pytest.raises(ValueError, match="TERMINAL_DAYTONA_LABELS.*dict"):
            tt._get_env_config()

    def test_labels_accepts_dict(self, monkeypatch):
        """TERMINAL_DAYTONA_LABELS='{"team":"infra"}' is valid and correct type."""
        monkeypatch.setenv("TERMINAL_DAYTONA_LABELS", '{"team":"infra"}')
        monkeypatch.setenv("TERMINAL_ENV", "daytona")
        import importlib
        import tools.terminal_tool as tt
        importlib.reload(tt)
        config = tt._get_env_config()
        assert config["daytona_labels"] == {"team": "infra"}

    def test_env_vars_rejects_list(self, monkeypatch):
        """TERMINAL_DAYTONA_ENV_VARS=[] is valid JSON but wrong shape (should be dict)."""
        monkeypatch.setenv("TERMINAL_DAYTONA_ENV_VARS", "[]")
        monkeypatch.setenv("TERMINAL_ENV", "daytona")
        import importlib
        import tools.terminal_tool as tt
        importlib.reload(tt)
        with pytest.raises(ValueError, match="TERMINAL_DAYTONA_ENV_VARS.*dict"):
            tt._get_env_config()

    def test_env_vars_rejects_string(self, monkeypatch):
        """TERMINAL_DAYTONA_ENV_VARS='\"hello\"' is valid JSON but not a dict."""
        monkeypatch.setenv("TERMINAL_DAYTONA_ENV_VARS", '"hello"')
        monkeypatch.setenv("TERMINAL_ENV", "daytona")
        import importlib
        import tools.terminal_tool as tt
        importlib.reload(tt)
        with pytest.raises(ValueError, match="TERMINAL_DAYTONA_ENV_VARS.*dict"):
            tt._get_env_config()

    def test_env_vars_accepts_dict(self, monkeypatch):
        """TERMINAL_DAYTONA_ENV_VARS='{"DEBUG":"1"}' is valid and correct type."""
        monkeypatch.setenv("TERMINAL_DAYTONA_ENV_VARS", '{"DEBUG":"1"}')
        monkeypatch.setenv("TERMINAL_ENV", "daytona")
        import importlib
        import tools.terminal_tool as tt
        importlib.reload(tt)
        config = tt._get_env_config()
        assert config["daytona_env_vars"] == {"DEBUG": "1"}

    def test_volume_mounts_rejects_dict(self, monkeypatch):
        """TERMINAL_DAYTONA_VOLUME_MOUNTS={} is valid JSON but wrong shape (should be list)."""
        monkeypatch.setenv("TERMINAL_DAYTONA_VOLUME_MOUNTS", "{}")
        monkeypatch.setenv("TERMINAL_ENV", "daytona")
        import importlib
        import tools.terminal_tool as tt
        importlib.reload(tt)
        with pytest.raises(ValueError, match="TERMINAL_DAYTONA_VOLUME_MOUNTS.*list"):
            tt._get_env_config()

    def test_volume_mounts_rejects_string(self, monkeypatch):
        """TERMINAL_DAYTONA_VOLUME_MOUNTS='\"hello\"' is valid JSON but not a list."""
        monkeypatch.setenv("TERMINAL_DAYTONA_VOLUME_MOUNTS", '"hello"')
        monkeypatch.setenv("TERMINAL_ENV", "daytona")
        import importlib
        import tools.terminal_tool as tt
        importlib.reload(tt)
        with pytest.raises(ValueError, match="TERMINAL_DAYTONA_VOLUME_MOUNTS.*list"):
            tt._get_env_config()

    def test_volume_mounts_accepts_list(self, monkeypatch):
        """TERMINAL_DAYTONA_VOLUME_MOUNTS='[{"volume_id":"vol-1","mount_path":"/data"}]' is valid."""
        monkeypatch.setenv("TERMINAL_DAYTONA_VOLUME_MOUNTS", '[{"volume_id":"vol-1","mount_path":"/data"}]')
        monkeypatch.setenv("TERMINAL_ENV", "daytona")
        import importlib
        import tools.terminal_tool as tt
        importlib.reload(tt)
        config = tt._get_env_config()
        assert config["daytona_volume_mounts"] == [{"volume_id": "vol-1", "mount_path": "/data"}]

    def test_labels_default_empty_dict(self, monkeypatch):
        """Default TERMINAL_DAYTONA_LABELS should be an empty dict (valid type)."""
        monkeypatch.delenv("TERMINAL_DAYTONA_LABELS", raising=False)
        monkeypatch.setenv("TERMINAL_ENV", "daytona")
        import importlib
        import tools.terminal_tool as tt
        importlib.reload(tt)
        config = tt._get_env_config()
        assert config["daytona_labels"] == {}

    def test_env_vars_default_empty_dict(self, monkeypatch):
        """Default TERMINAL_DAYTONA_ENV_VARS should be an empty dict (valid type)."""
        monkeypatch.delenv("TERMINAL_DAYTONA_ENV_VARS", raising=False)
        monkeypatch.setenv("TERMINAL_ENV", "daytona")
        import importlib
        import tools.terminal_tool as tt
        importlib.reload(tt)
        config = tt._get_env_config()
        assert config["daytona_env_vars"] == {}

    def test_volume_mounts_default_empty_list(self, monkeypatch):
        """Default TERMINAL_DAYTONA_VOLUME_MOUNTS should be an empty list (valid type)."""
        monkeypatch.delenv("TERMINAL_DAYTONA_VOLUME_MOUNTS", raising=False)
        monkeypatch.setenv("TERMINAL_ENV", "daytona")
        import importlib
        import tools.terminal_tool as tt
        importlib.reload(tt)
        config = tt._get_env_config()
        assert config["daytona_volume_mounts"] == []

    def test_terminal_tool_bad_json_returns_clean_error(self, monkeypatch):
        """Runtime path should return a clean tool error for malformed JSON config."""
        import importlib
        import json
        import tools.terminal_tool as tt

        monkeypatch.setenv("TERMINAL_ENV", "daytona")
        monkeypatch.setenv("TERMINAL_DAYTONA_LABELS", "{not-json")
        importlib.reload(tt)

        result = json.loads(tt.terminal_tool(command="true"))

        assert result["exit_code"] == -1
        assert "Invalid value for TERMINAL_DAYTONA_LABELS" in result["error"]
        assert "traceback" not in result


# ---------------------------------------------------------------------------
# Profile ID derivation
# ---------------------------------------------------------------------------

class TestDeriveProfileId:
    """_derive_profile_id() should produce a stable, non-secret 8-char
    hex identifier from the resolved hermes home path."""

    def test_derive_profile_id_returns_8_hex_chars(self, monkeypatch):
        """_derive_profile_id should return exactly 8 hexadecimal chars."""
        from pathlib import Path
        monkeypatch.setattr("hermes_constants.get_hermes_home",
                            lambda: Path("/home/testuser/.hermes"))
        # Force re-import to apply the mock
        import importlib
        import tools.environments.daytona as daytona_mod
        importlib.reload(daytona_mod)
        pid = daytona_mod._derive_profile_id()
        assert len(pid) == 8, f"Expected 8-char profile_id, got {len(pid)}: {pid}"
        assert all(c in "0123456789abcdef" for c in pid), f"Expected hex chars, got {pid}"

    def test_derive_profile_id_stable_for_same_path(self, monkeypatch):
        """Same hermes home should always produce the same profile_id."""
        from pathlib import Path
        monkeypatch.setattr("hermes_constants.get_hermes_home",
                            lambda: Path("/home/testuser/.hermes"))
        import importlib
        import tools.environments.daytona as daytona_mod
        importlib.reload(daytona_mod)
        assert daytona_mod._derive_profile_id() == daytona_mod._derive_profile_id()

    def test_derive_profile_id_different_for_different_paths(self, monkeypatch):
        """Different hermes home paths should produce different profile_ids."""
        from pathlib import Path
        import importlib
        import tools.environments.daytona as daytona_mod
        results = set()
        for path_str in [
            "/home/user1/.hermes",
            "/home/user2/.hermes",
            "/home/user1/.hermes/profiles/dev-docker",
        ]:
            monkeypatch.setattr("hermes_constants.get_hermes_home",
                                lambda p=path_str: Path(p))
            importlib.reload(daytona_mod)
            results.add(daytona_mod._derive_profile_id())
        assert len(results) == 3, f"Expected 3 distinct profile_ids, got {len(results)}"

    def test_derive_profile_id_uses_resolved_path(self, monkeypatch):
        """_derive_profile_id should resolve symlinks before hashing."""
        from pathlib import Path
        real_path = Path("/Users/real/.hermes")
        monkeypatch.setattr("hermes_constants.get_hermes_home",
                            lambda: real_path)
        import importlib
        import tools.environments.daytona as daytona_mod
        importlib.reload(daytona_mod)
        pid1 = daytona_mod._derive_profile_id()
        pid2 = daytona_mod._derive_profile_id()
        assert pid1 == pid2

    def test_derive_profile_id_non_reversible(self, monkeypatch):
        """The profile_id should be a one-way hash; it must NOT contain
        the original path in cleartext."""
        from pathlib import Path
        test_path = "/home/verylong-username-with-details/.hermes"
        monkeypatch.setattr("hermes_constants.get_hermes_home",
                            lambda: Path(test_path))
        import importlib
        import tools.environments.daytona as daytona_mod
        importlib.reload(daytona_mod)
        pid = daytona_mod._derive_profile_id()
        # The 8-char hex hash must not contain the original path
        assert test_path not in pid
        assert len(pid) == 8


# ---------------------------------------------------------------------------
# Real Daytona SDK compatibility smoke
# ---------------------------------------------------------------------------

class TestDaytonaSdkParamCompatibility:
    """Forwarded kwargs must be accepted by the real daytona 0.155.x models."""

    def test_real_daytona_0155_accepts_forwarded_create_kwargs(self):
        daytona = pytest.importorskip("daytona")
        volume_mod = pytest.importorskip("daytona.common.volume")

        fields = set(daytona.CreateSandboxFromImageParams.model_fields)
        assert {
            "name",
            "labels",
            "auto_stop_interval",
            "auto_archive_interval",
            "auto_delete_interval",
            "resources",
            "ephemeral",
            "env_vars",
            "network_block_all",
            "network_allow_list",
            "language",
            "volumes",
            "image",
        }.issubset(fields)

        snapshot_fields = set(daytona.CreateSandboxFromSnapshotParams.model_fields)
        assert {
            "name",
            "labels",
            "auto_stop_interval",
            "auto_archive_interval",
            "auto_delete_interval",
            "ephemeral",
            "env_vars",
            "network_block_all",
            "network_allow_list",
            "language",
            "volumes",
            "snapshot",
        }.issubset(snapshot_fields)
        assert "resources" not in snapshot_fields

        volume = volume_mod.VolumeMount(volume_id="vol-123", mount_path="/data")
        resources = daytona.Resources(cpu=2, memory=4, disk=10, gpu=1)
        daytona.CreateSandboxFromImageParams(
            name="hermes-test",
            labels={"hermes_backend": "daytona"},
            auto_stop_interval=30,
            auto_archive_interval=60,
            auto_delete_interval=0,
            resources=resources,
            ephemeral=True,
            env_vars={"DEBUG": "1"},
            network_block_all=True,
            network_allow_list="10.0.0.0/8",
            language="python",
            volumes=[volume],
            image="python:3.11",
        )
        daytona.CreateSandboxFromSnapshotParams(
            name="hermes-test",
            labels={"hermes_backend": "daytona"},
            auto_stop_interval=30,
            auto_archive_interval=60,
            auto_delete_interval=0,
            ephemeral=True,
            env_vars={"DEBUG": "1"},
            network_block_all=True,
            network_allow_list="10.0.0.0/8",
            language="python",
            volumes=[volume],
            snapshot="snap-123",
        )
