"""Unit tests for the Sprites cloud sandbox environment backend.

These exercise SpritesEnvironment against a mocked sprites-py SDK; no
network or token required. Live-API checks live under
tests/integration/test_sprites_terminal.py.
"""

import sys
import types
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Mock sprites-py SDK
# ---------------------------------------------------------------------------

class _NotFoundError(Exception):
    pass


class _SpriteError(Exception):
    pass


class _ExitError(Exception):
    """Mirror of sprites.exceptions.ExitError."""

    def __init__(self, message, exit_code, stdout=b"", stderr=b""):
        super().__init__(message)
        self._exit_code = exit_code
        self.stdout = stdout
        self.stderr = stderr

    def exit_code(self):
        return self._exit_code


class _SpritesTimeoutError(Exception):
    pass


def _patch_sprites_imports(monkeypatch):
    """Inject a fake sprites SDK so SpritesEnvironment can import it."""
    sprites_mod = types.ModuleType("sprites")
    sprites_mod.SpritesClient = MagicMock(name="SpritesClient")

    exc_mod = types.ModuleType("sprites.exceptions")
    exc_mod.NotFoundError = _NotFoundError
    exc_mod.SpriteError = _SpriteError
    exc_mod.ExitError = _ExitError
    exc_mod.TimeoutError = _SpritesTimeoutError
    sprites_mod.exceptions = exc_mod

    monkeypatch.setitem(sys.modules, "sprites", sprites_mod)
    monkeypatch.setitem(sys.modules, "sprites.exceptions", exc_mod)
    return sprites_mod, exc_mod


def _make_sprite(name="hermes-default"):
    sprite = MagicMock()
    sprite.name = name

    # $HOME detection returns "/home/sprite" by default
    home_cmd = MagicMock()
    home_cmd.combined_output.return_value = b"/home/sprite\n"

    # init_session() bootstrap also goes through sprite.command(...).
    # combined_output() must succeed (return bytes) for snapshot_ready=True.
    bootstrap_cmd = MagicMock()
    bootstrap_cmd.combined_output.return_value = b"\n__HERMES_CWD_xxx__/home/sprite__HERMES_CWD_xxx__\n"

    sprite.command.side_effect = [home_cmd, bootstrap_cmd]
    sprite.filesystem.return_value = MagicMock()
    return sprite


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def sprites_sdk(monkeypatch):
    return _patch_sprites_imports(monkeypatch)


@pytest.fixture()
def make_env(sprites_sdk, monkeypatch):
    """Build a SpritesEnvironment instance against a mocked SDK.

    Returns a factory; keyword args mirror SpritesEnvironment.__init__.
    The factory accepts an optional ``get_side_effect`` to control what
    ``client.get_sprite()`` does (e.g. raise NotFoundError to force create).
    """
    monkeypatch.setenv("SPRITES_TOKEN", "test-token")
    # Don't try to lazy-install the SDK during tests
    monkeypatch.setattr(
        "tools.lazy_deps.ensure", lambda *a, **k: None, raising=False
    )
    # Skip credential-file enumeration so init doesn't bring in real ~/.hermes state
    monkeypatch.setattr(
        "tools.credential_files.get_credential_file_mounts", lambda: []
    )
    monkeypatch.setattr(
        "tools.credential_files.iter_skills_files", lambda **kw: []
    )
    monkeypatch.setattr(
        "tools.credential_files.iter_cache_files", lambda **kw: []
    )
    # Keep the base class from blocking forever on interrupt polling
    monkeypatch.setattr("tools.environments.base.is_interrupted", lambda: False)
    # Pin the active profile to "default" so Sprite names are deterministic
    # regardless of the test runner's HERMES_HOME. Profile-scoping itself is
    # covered explicitly in TestSpriteNaming.
    monkeypatch.setattr(
        "agent.file_safety._resolve_active_profile_name",
        lambda: "default",
        raising=False,
    )

    def _factory(get_side_effect=None, sprite=None, **kwargs):
        sprite = sprite or _make_sprite()

        mock_client = MagicMock()
        mock_client.create_sprite.return_value = sprite
        if get_side_effect is not None:
            mock_client.get_sprite.side_effect = get_side_effect
        else:
            mock_client.get_sprite.side_effect = _NotFoundError("not found")

        sprites_mod, _ = sprites_sdk
        sprites_mod.SpritesClient = MagicMock(return_value=mock_client)

        from tools.environments.sprites import SpritesEnvironment

        env = SpritesEnvironment(**kwargs)
        env._mock_client = mock_client
        env._mock_sprite = sprite
        return env

    return _factory


# ---------------------------------------------------------------------------
# Construction / token handling
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_missing_token_raises(self, sprites_sdk, monkeypatch):
        monkeypatch.delenv("SPRITES_TOKEN", raising=False)
        monkeypatch.delenv("SPRITE_TOKEN", raising=False)
        monkeypatch.setattr(
            "tools.lazy_deps.ensure", lambda *a, **k: None, raising=False
        )
        from tools.environments.sprites import SpritesEnvironment

        with pytest.raises(ValueError, match="SPRITES_TOKEN"):
            SpritesEnvironment(task_id="x")

    def test_persistent_uses_get_first(self, make_env):
        existing = _make_sprite(name="hermes-mine")
        env = make_env(
            get_side_effect=lambda name: existing,
            sprite=existing,
            task_id="mine",
            persistent_filesystem=True,
        )
        env._mock_client.get_sprite.assert_called_once_with("hermes-mine")
        env._mock_client.create_sprite.assert_not_called()
        # The instance records the resolved name it will resume under.
        assert env._sprite_name == "hermes-mine"

    def test_creates_when_not_found(self, make_env):
        env = make_env(task_id="fresh", persistent_filesystem=True)
        env._mock_client.get_sprite.assert_called_once_with("hermes-fresh")
        env._mock_client.create_sprite.assert_called_once_with("hermes-fresh")
        assert env._sprite_name == "hermes-fresh"

    def test_no_size_kwargs_passed_to_create(self, make_env):
        """Compute sizing isn't honored yet — make sure we don't sneak it back in."""
        env = make_env(task_id="sizing")
        args, kwargs = env._mock_client.create_sprite.call_args
        assert args == ("hermes-sizing",)
        assert kwargs == {}


class TestSpriteNaming:
    """`_resolve_sprite_name`: deterministic, profile-scoped Sprite identity.

    A Sprite is resumed *by name*, so the name is the durable identity of a
    session's live sandbox. The name must (a) stay stable so resume works and
    (b) differ across independent Hermes profiles so they never resume into
    one another's live Sprite.
    """

    @staticmethod
    def _set_profile(monkeypatch, name):
        monkeypatch.setattr(
            "agent.file_safety._resolve_active_profile_name",
            lambda: name,
            raising=False,
        )

    def test_default_profile_keeps_legacy_name(self, monkeypatch):
        from tools.environments.sprites import _resolve_sprite_name
        self._set_profile(monkeypatch, "default")
        # Backward compatible with Sprites created before profile scoping.
        assert _resolve_sprite_name("default") == "hermes-default"
        assert _resolve_sprite_name("mytask") == "hermes-mytask"

    def test_named_profile_is_scoped(self, monkeypatch):
        from tools.environments.sprites import _resolve_sprite_name
        self._set_profile(monkeypatch, "work")
        assert _resolve_sprite_name("default") == "hermes-work-default"
        assert _resolve_sprite_name("mytask") == "hermes-work-mytask"

    def test_independent_profiles_do_not_collide(self, monkeypatch):
        """Same task_id under two different profiles → distinct Sprites."""
        from tools.environments.sprites import _resolve_sprite_name
        self._set_profile(monkeypatch, "alpha")
        a = _resolve_sprite_name("default")
        self._set_profile(monkeypatch, "beta")
        b = _resolve_sprite_name("default")
        assert a == "hermes-alpha-default"
        assert b == "hermes-beta-default"
        assert a != b

    def test_same_identity_resumes(self, monkeypatch):
        """Same (profile, task_id) is stable across calls → resume works."""
        from tools.environments.sprites import _resolve_sprite_name
        self._set_profile(monkeypatch, "work")
        assert _resolve_sprite_name("t") == _resolve_sprite_name("t") == "hermes-work-t"

    def test_names_are_sanitized(self, monkeypatch):
        """Messy profile/task components collapse to a Sprite-safe slug."""
        import re
        from tools.environments.sprites import _resolve_sprite_name
        self._set_profile(monkeypatch, "Team/Prod.01")
        name = _resolve_sprite_name("sub agent_42")
        assert name == "hermes-team-prod-01-sub-agent-42"
        # Only lowercase alnum + single interior hyphens (Fly/DNS-safe).
        assert re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name)

    def test_empty_task_id_falls_back(self, monkeypatch):
        from tools.environments.sprites import _resolve_sprite_name
        self._set_profile(monkeypatch, "default")
        assert _resolve_sprite_name("") == "hermes-default"

    def test_profile_resolution_failure_is_non_fatal(self, monkeypatch):
        """A broken profile resolver must not break Sprite naming."""
        from tools.environments import sprites as sprites_mod

        def _boom():
            raise RuntimeError("no home")

        monkeypatch.setattr(
            "agent.file_safety._resolve_active_profile_name", _boom, raising=False
        )
        assert sprites_mod._resolve_sprite_name("x") == "hermes-x"


class TestDispatchWiring:
    """Wiring pins: config → terminal_tool dispatch → SpritesEnvironment kwargs.

    The class-level tests above prove SpritesEnvironment honors
    ``persistent_filesystem``; these prove the dispatch actually delivers it.
    A backend missing from terminal_tool's container_config builder gets
    ``container_config=None``, silently re-defaulting ``container_persistent:
    false`` back to persistent — i.e. ephemeral mode can never engage.
    """

    def test_terminal_tool_builds_container_config_for_sprites(self, monkeypatch):
        import tools.terminal_tool as tt

        captured = {}

        config = {
            "env_type": "sprites",
            "docker_image": "unused",
            "singularity_image": "unused",
            "modal_image": "unused",
            "daytona_image": "unused",
            "cwd": "/root",
            "host_cwd": None,
            "timeout": 180,
            "lifetime_seconds": 300,
            "container_cpu": 1,
            "container_memory": 5120,
            "container_disk": 51200,
            "container_persistent": False,
            "docker_volumes": [],
            "docker_env": {},
            "docker_extra_args": [],
            "docker_mount_cwd_to_workspace": False,
            "docker_run_as_host_user": False,
            "docker_forward_env": [],
            "modal_mode": "auto",
        }

        class _DummyEnv:
            cwd = "/root"

            def execute(self, *a, **k):
                return {"output": "", "exit_code": 0}

        def fake_create_environment(env_type, image, cwd, timeout, **kwargs):
            captured["env_type"] = env_type
            captured["container_config"] = kwargs.get("container_config")
            return _DummyEnv()

        monkeypatch.setattr(tt, "_get_env_config", lambda: config)
        monkeypatch.setattr(tt, "_start_cleanup_thread", lambda: None)
        monkeypatch.setattr(tt, "_check_all_guards", lambda *a, **k: {"approved": True})
        monkeypatch.setattr(tt, "_create_environment", fake_create_environment)
        monkeypatch.setattr(tt, "_active_environments", {})
        monkeypatch.setattr(tt, "_last_activity", {})

        tt.terminal_tool(command="pwd")

        assert captured["env_type"] == "sprites"
        cc = captured["container_config"]
        assert cc is not None, (
            "sprites must be in terminal_tool's container_config builder set; "
            "container_config=None silently discards container_persistent"
        )
        assert cc["container_persistent"] is False

    def test_create_environment_passes_persistence_and_task_id(self, monkeypatch):
        import tools.terminal_tool as tt
        import tools.environments.sprites as sprites_mod

        captured = {}

        class _FakeSpritesEnv:
            def __init__(self, cwd, timeout, persistent_filesystem, task_id):
                captured.update(
                    cwd=cwd,
                    timeout=timeout,
                    persistent_filesystem=persistent_filesystem,
                    task_id=task_id,
                )

        monkeypatch.setattr(sprites_mod, "SpritesEnvironment", _FakeSpritesEnv)

        tt._create_environment(
            env_type="sprites",
            image="ignored",
            cwd="/root",
            timeout=60,
            container_config={"container_persistent": False},
            task_id="tid-ephemeral",
        )

        assert captured["persistent_filesystem"] is False
        assert captured["task_id"] == "tid-ephemeral"
        assert captured["cwd"] == "/root"

    def test_no_base_url_kwarg(self, make_env, sprites_sdk):
        """SpritesClient is constructed without a base_url override (endpoint is fixed)."""
        env = make_env(task_id="urlcheck")
        sprites_mod, _ = sprites_sdk
        _, kwargs = sprites_mod.SpritesClient.call_args
        assert "base_url" not in kwargs


# ---------------------------------------------------------------------------
# CWD / home detection
# ---------------------------------------------------------------------------

class TestCwdResolution:
    def test_default_cwd_rewrites_to_detected_home(self, make_env):
        env = make_env(task_id="cwd1")  # default cwd="/root"
        assert env.cwd == "/home/sprite"  # rewritten from "/root" → detected home

    def test_tilde_cwd_rewrites_to_detected_home(self, make_env):
        env = make_env(cwd="~", task_id="cwd2")
        assert env.cwd == "/home/sprite"

    def test_explicit_cwd_not_overridden(self, make_env):
        sprite = _make_sprite()
        env = make_env(sprite=sprite, cwd="/workspace", task_id="cwd3")
        assert env.cwd == "/workspace"


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

class TestCleanup:
    def test_persistent_cleanup_leaves_sprite_alive(self, make_env):
        env = make_env(task_id="persist", persistent_filesystem=True)
        sprite = env._mock_sprite
        env.cleanup()
        sprite.delete.assert_not_called()

    def test_non_persistent_cleanup_deletes_sprite(self, make_env):
        env = make_env(task_id="ephem", persistent_filesystem=False)
        sprite = env._mock_sprite
        env.cleanup()
        sprite.delete.assert_called_once()

    def test_cleanup_idempotent(self, make_env):
        env = make_env(task_id="idem", persistent_filesystem=True)
        env.cleanup()
        env.cleanup()  # second call must not raise

    def test_cleanup_closes_client(self, make_env):
        env = make_env(task_id="closeit", persistent_filesystem=True)
        env.cleanup()
        env._mock_client.close.assert_called_once()


# ---------------------------------------------------------------------------
# _run_bash exit-code surfacing
# ---------------------------------------------------------------------------

class TestRunBashExitCodes:
    def test_zero_exit_returns_output(self, make_env):
        env = make_env(task_id="rb0")
        # Reset side_effect; new sprite.command() call should return a fresh Cmd.
        cmd = MagicMock()
        cmd.combined_output.return_value = b"hi\n"
        env._mock_sprite.command = MagicMock(return_value=cmd)

        handle = env._run_bash("echo hi", timeout=10)
        handle.wait()
        out = handle.stdout.read()
        assert out == "hi\n"
        assert handle.returncode == 0

    def test_nonzero_exit_surfaces_code_from_ExitError(self, make_env, sprites_sdk):
        env = make_env(task_id="rb7")
        _, exc_mod = sprites_sdk
        cmd = MagicMock()
        cmd.combined_output.side_effect = exc_mod.ExitError(
            "exit status 7", 7, b"before\n", b""
        )
        env._mock_sprite.command = MagicMock(return_value=cmd)

        handle = env._run_bash("exit 7", timeout=10)
        handle.wait()
        out = handle.stdout.read()
        assert "before" in out
        assert handle.returncode == 7

    def test_timeout_surfaces_124(self, make_env, sprites_sdk):
        env = make_env(task_id="rbto")
        _, exc_mod = sprites_sdk
        cmd = MagicMock()
        cmd.combined_output.side_effect = exc_mod.TimeoutError("deadline")
        env._mock_sprite.command = MagicMock(return_value=cmd)

        handle = env._run_bash("sleep 999", timeout=1)
        handle.wait()
        assert handle.returncode == 124


# ---------------------------------------------------------------------------
# File-sync push (upload_fn behavior)
# ---------------------------------------------------------------------------

class TestFileSyncPush:
    def test_upload_writes_via_filesystem_api(self, make_env, tmp_path):
        env = make_env(task_id="fs")
        # Build a fake host file
        host_file = tmp_path / "secret.txt"
        host_file.write_bytes(b"hello")

        # Mock the SpritePath returned by `fs / remote_path`
        remote_path_obj = MagicMock()
        env._fs.__truediv__.return_value = remote_path_obj

        env._sprite_upload(str(host_file), "/home/sprite/.hermes/foo")
        remote_path_obj.parent.mkdir.assert_called_once_with(
            parents=True, exist_ok=True
        )
        remote_path_obj.write_bytes.assert_called_once_with(b"hello")

    def test_delete_invokes_unlink_per_path(self, make_env):
        env = make_env(task_id="fsdel")
        remote_obj = MagicMock()
        env._fs.__truediv__.return_value = remote_obj
        env._sprite_delete(["/home/sprite/.hermes/a", "/home/sprite/.hermes/b"])
        # Each path → one unlink call
        assert remote_obj.unlink.call_count == 2
        remote_obj.unlink.assert_any_call(missing_ok=True)


# ---------------------------------------------------------------------------
# _stdin_mode wiring
# ---------------------------------------------------------------------------

class TestStdinMode:
    def test_stdin_mode_is_heredoc(self):
        """Ensures the base class will embed stdin via heredoc, not pipe.

        SDK calls don't accept a real stdin stream, so the backend declares
        ``_stdin_mode = "heredoc"`` and the base ``execute()`` wraps stdin
        into the command string before calling ``_run_bash``.
        """
        # Inspect the class without constructing — no SDK needed for this check
        import importlib

        # Stub the SDK so the module imports cleanly outside the make_env fixture
        sys.modules.setdefault("sprites", types.ModuleType("sprites"))
        sys.modules.setdefault("sprites.exceptions", types.ModuleType("sprites.exceptions"))

        # Force a clean import (in case earlier tests left it in sys.modules with
        # a different SDK mocked in)
        if "tools.environments.sprites" in sys.modules:
            importlib.reload(sys.modules["tools.environments.sprites"])
        from tools.environments.sprites import SpritesEnvironment

        assert SpritesEnvironment._stdin_mode == "heredoc"
