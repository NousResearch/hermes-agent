"""Tests for the profile-scoped filesystem allowlist guard.

Covers the guarantees the guard is meant to provide:
  * a restricted profile is refused paths outside its allowlisted roots;
  * paths inside an allowlisted root are permitted;
  * a symlink planted inside an allowed root that points at a denied dir is
    refused (realpath resolution);
  * a sibling dir sharing a name prefix with an allowed root is NOT allowed
    (trailing-separator boundary);
  * a profile with no allowlist entry (and the default profile) is
    unrestricted — pure pass-through.

The classes at the bottom exercise the guard WITHOUT mocking the policy
loader, against a real temporary ``HERMES_HOME`` and through the real file
tools, so the loader, the tool dispatch, and the container-path branch are
covered rather than stubbed.
"""

import json
import os
import tempfile
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

import tools.profile_fs_guard as guard


def _with_allowlist(mapping, active_profile):
    """Context helper: force the cached allowlist and the active profile."""
    guard.reset_cache()
    # Patch _load_allowlist to return a pre-resolved map (keys lowercased,
    # roots realpath'd) so tests don't depend on an on-disk config.yaml.
    resolved = {
        p.lower(): [guard._resolve_root(r) for r in roots]
        for p, roots in mapping.items()
    }
    return (
        patch.object(guard, "_load_allowlist", return_value=resolved),
        patch.object(guard, "_active_profile", return_value=active_profile),
    )


class TestProfileFsGuard:
    def test_restricted_profile_denied_outside_root(self):
        p1, p2 = _with_allowlist({"bot": ["/srv/allowed"]}, "bot")
        with p1, p2:
            err = guard.check_path_allowed("/srv/secret/finances.csv")
            assert err is not None
            assert "Access denied" in err

    def test_restricted_profile_allowed_inside_root(self):
        p1, p2 = _with_allowlist({"bot": ["/srv/allowed"]}, "bot")
        with p1, p2:
            assert guard.check_path_allowed("/srv/allowed/notes.md") is None
            # The root itself is allowed.
            assert guard.check_path_allowed("/srv/allowed") is None

    def test_prefix_sibling_not_allowed(self):
        # /srv/allowed must not permit /srv/allowed-secret via startswith.
        p1, p2 = _with_allowlist({"bot": ["/srv/allowed"]}, "bot")
        with p1, p2:
            assert guard.check_path_allowed("/srv/allowed-secret/x") is not None

    def test_symlink_escape_denied(self):
        # A symlink inside an allowed root that points at a denied directory
        # must resolve (realpath) to the denied target and be refused.
        allowed = tempfile.mkdtemp()
        denied = tempfile.mkdtemp()
        try:
            link = os.path.join(allowed, "escape")
            os.symlink(denied, link)
            p1, p2 = _with_allowlist({"bot": [allowed]}, "bot")
            with p1, p2:
                target = os.path.join(link, "secret.csv")
                assert guard.check_path_allowed(target) is not None
        finally:
            os.path.exists(os.path.join(allowed, "escape")) and os.remove(
                os.path.join(allowed, "escape")
            )
            os.rmdir(allowed)
            os.rmdir(denied)

    def test_unlisted_profile_unrestricted(self):
        # 'personal' has no entry -> guard is a pass-through, even for a path
        # that a restricted profile would be denied.
        p1, p2 = _with_allowlist({"bot": ["/srv/allowed"]}, "personal")
        with p1, p2:
            assert guard.check_path_allowed("/srv/secret/finances.csv") is None

    def test_default_profile_unrestricted(self):
        p1, p2 = _with_allowlist({"bot": ["/srv/allowed"]}, "default")
        with p1, p2:
            assert guard.check_path_allowed("/anywhere/at/all") is None

    def test_relative_path_anchored_to_base_dir(self):
        # Relative paths resolve against base_dir before the check.
        p1, p2 = _with_allowlist({"bot": ["/srv/allowed"]}, "bot")
        with p1, p2:
            assert guard.check_path_allowed("sub/x.md", base_dir="/srv/allowed") is None
            assert guard.check_path_allowed("../secret/x", base_dir="/srv/allowed") is not None

    def test_empty_allowlist_is_passthrough(self):
        # No profiles configured at all -> nobody is restricted.
        p1, p2 = _with_allowlist({}, "bot")
        with p1, p2:
            assert guard.check_path_allowed("/anywhere") is None


# ---------------------------------------------------------------------------
# Real-policy tests: no loader mocking. These write an actual config.yaml into
# a temporary HERMES_HOME so the loader itself is under test.
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_home(tmp_path, monkeypatch):
    """A real temporary HERMES_HOME with a profiles/ layout.

    Yields a helper that writes the root ``config.yaml`` and activates a named
    profile, so ``_load_allowlist`` and ``_active_profile`` both resolve for
    real. The guard cache is dropped before and after each test.
    """
    root = tmp_path / "hermes"
    (root / "profiles").mkdir(parents=True)

    class _Home:
        path = root

        def write_config(self, body: str) -> None:
            (root / "config.yaml").write_text(textwrap.dedent(body), encoding="utf-8")
            guard.reset_cache()

        def write_raw_config(self, raw: bytes) -> None:
            (root / "config.yaml").write_bytes(raw)
            guard.reset_cache()

        def activate(self, profile: str) -> Path:
            """Point HERMES_HOME at ``profiles/<profile>`` and return it."""
            home = root / "profiles" / profile
            home.mkdir(parents=True, exist_ok=True)
            monkeypatch.setenv("HERMES_HOME", str(home))
            return home

        def activate_default(self) -> Path:
            monkeypatch.setenv("HERMES_HOME", str(root))
            return root

    # get_default_hermes_root() treats a HERMES_HOME outside ~/.hermes as the
    # root itself, which is exactly the Docker-style layout we build here.
    monkeypatch.setenv("HERMES_HOME", str(root))
    guard.reset_cache()
    yield _Home()
    guard.reset_cache()


class TestRealPolicyLoading:
    """The loader against a real on-disk config.yaml — no mocks."""

    def test_restricted_profile_enforced_from_real_config(self, temp_home):
        allowed = temp_home.path / "work"
        allowed.mkdir()
        temp_home.write_config(f"""
            profile_fs_allowlist:
              clientbot:
                - {allowed}
        """)
        temp_home.activate("clientbot")

        assert guard.check_path_allowed(str(allowed / "notes.md")) is None
        err = guard.check_path_allowed(str(temp_home.path / "private" / "money.csv"))
        assert err is not None and "Access denied" in err

    def test_unlisted_profile_unrestricted_from_real_config(self, temp_home):
        temp_home.write_config("""
            profile_fs_allowlist:
              clientbot:
                - /srv/allowed
        """)
        temp_home.activate("personal")
        assert guard.check_path_allowed("/anywhere/at/all") is None

    def test_absent_key_is_noop(self, temp_home):
        # A config with no profile_fs_allowlist key restricts nobody.
        temp_home.write_config("""
            model: some-model
        """)
        temp_home.activate("clientbot")
        assert guard.check_path_allowed("/anywhere/at/all") is None

    def test_missing_config_file_is_noop(self, temp_home):
        # No config.yaml written at all.
        guard.reset_cache()
        temp_home.activate("clientbot")
        assert guard.check_path_allowed("/anywhere/at/all") is None

    def test_profile_key_is_case_insensitive(self, temp_home):
        allowed = temp_home.path / "work"
        allowed.mkdir()
        temp_home.write_config(f"""
            profile_fs_allowlist:
              ClientBot:
                - {allowed}
        """)
        temp_home.activate("clientbot")
        assert guard.check_path_allowed(str(allowed / "x.md")) is None
        assert guard.check_path_allowed("/etc/passwd") is not None


class TestMalformedPolicyFailsClosed:
    """A declared-but-broken policy must DENY, never silently unrestrict."""

    def test_unparseable_yaml_denies(self, temp_home):
        temp_home.write_config("""
            profile_fs_allowlist:
              clientbot:
                - /srv/allowed
               bad_indent: [
        """)
        temp_home.activate("clientbot")
        err = guard.check_path_allowed("/etc/passwd")
        assert err is not None
        assert "policy could not be loaded" in err

    def test_unreadable_config_denies(self, temp_home):
        temp_home.write_config("""
            profile_fs_allowlist:
              clientbot:
                - /srv/allowed
        """)
        cfg = temp_home.path / "config.yaml"
        cfg.chmod(0o000)
        guard.reset_cache()
        temp_home.activate("clientbot")
        try:
            if os.access(cfg, os.R_OK):
                pytest.skip("cannot make file unreadable (running as root?)")
            err = guard.check_path_allowed("/etc/passwd")
            assert err is not None
            assert "policy could not be loaded" in err
        finally:
            cfg.chmod(0o644)

    def test_scalar_allowlist_block_denies(self, temp_home):
        # profile_fs_allowlist must be a mapping, not a scalar.
        temp_home.write_config("""
            profile_fs_allowlist: "oops"
        """)
        temp_home.activate("clientbot")
        err = guard.check_path_allowed("/etc/passwd")
        assert err is not None and "policy could not be loaded" in err

    def test_bare_string_roots_denies(self, temp_home):
        # A bare string instead of a list is a likely typo; refuse to guess.
        temp_home.write_config("""
            profile_fs_allowlist:
              clientbot: /srv/allowed
        """)
        temp_home.activate("clientbot")
        err = guard.check_path_allowed("/srv/allowed/x")
        assert err is not None and "policy could not be loaded" in err

    def test_empty_root_list_denies_everything(self, temp_home):
        # An explicit empty list means "restricted, with nothing allowed" —
        # distinct from an absent entry, which means "unrestricted".
        temp_home.write_config("""
            profile_fs_allowlist:
              clientbot: []
        """)
        temp_home.activate("clientbot")
        assert guard.check_path_allowed("/anywhere") is not None


class TestContainerPathResolution:
    """Backend-aware resolution: guest paths are not host-realpath'd."""

    def test_container_path_not_host_dereferenced(self, temp_home, monkeypatch):
        # /workspace on the host is a symlink to somewhere else; under a
        # container backend the guard must judge the guest path as written,
        # not the host target it happens to point at.
        host_target = temp_home.path / "host_real"
        host_target.mkdir()
        workspace_link = temp_home.path / "workspace"
        workspace_link.symlink_to(host_target)

        temp_home.write_config(f"""
            profile_fs_allowlist:
              clientbot:
                - {workspace_link}
        """)
        temp_home.activate("clientbot")

        monkeypatch.setattr(guard, "_uses_container_paths", lambda task_id: True)
        # Lexically inside the allowed root -> allowed, despite the host
        # symlink pointing elsewhere.
        assert guard.check_path_allowed(f"{workspace_link}/src/main.py") is None

    def test_container_traversal_still_denied(self, temp_home, monkeypatch):
        temp_home.write_config("""
            profile_fs_allowlist:
              clientbot:
                - /workspace
        """)
        temp_home.activate("clientbot")
        monkeypatch.setattr(guard, "_uses_container_paths", lambda task_id: True)

        # ``..`` is collapsed lexically before the boundary test.
        assert guard.check_path_allowed("/workspace/../etc/passwd") is not None
        assert guard.check_path_allowed("/workspace/ok/../ok2") is None

    def test_local_backend_still_dereferences_symlinks(self, temp_home, monkeypatch):
        # The same layout under a LOCAL backend must resolve the symlink and
        # refuse, since the host target is outside the allowed root.
        allowed = temp_home.path / "allowed"
        allowed.mkdir()
        denied = temp_home.path / "denied"
        denied.mkdir()
        (allowed / "escape").symlink_to(denied)

        temp_home.write_config(f"""
            profile_fs_allowlist:
              clientbot:
                - {allowed}
        """)
        temp_home.activate("clientbot")
        monkeypatch.setattr(guard, "_uses_container_paths", lambda task_id: False)

        assert guard.check_path_allowed(str(allowed / "escape" / "secret.csv")) is not None


class TestFileToolDispatch:
    """End-to-end through the real tools, including patch."""

    @pytest.fixture
    def restricted(self, temp_home):
        allowed = temp_home.path / "work"
        allowed.mkdir()
        denied = temp_home.path / "private"
        denied.mkdir()
        (denied / "money.csv").write_text("secret", encoding="utf-8")
        temp_home.write_config(f"""
            profile_fs_allowlist:
              clientbot:
                - {allowed}
        """)
        temp_home.activate("clientbot")
        return allowed, denied

    def test_read_file_denied_outside_root(self, restricted):
        from tools.file_tools import read_file_tool

        _allowed, denied = restricted
        out = read_file_tool(str(denied / "money.csv"))
        assert "Access denied" in out

    def test_read_file_allowed_inside_root(self, restricted):
        from tools.file_tools import read_file_tool

        allowed, _denied = restricted
        target = allowed / "notes.md"
        target.write_text("hello", encoding="utf-8")
        out = read_file_tool(str(target))
        assert "Access denied" not in out

    def test_write_file_denied_outside_root(self, restricted):
        from tools.file_tools import write_file_tool

        _allowed, denied = restricted
        out = write_file_tool(str(denied / "planted.txt"), "x")
        assert "Access denied" in out
        assert not (denied / "planted.txt").exists()

    def test_search_denied_outside_root(self, restricted):
        from tools.file_tools import search_tool

        _allowed, denied = restricted
        out = search_tool("secret", target="content", path=str(denied))
        assert "Access denied" in out

    def test_patch_replace_denied_outside_root(self, restricted):
        """The gap this guard previously had: patch in replace mode."""
        from tools.file_tools import patch_tool

        _allowed, denied = restricted
        victim = denied / "money.csv"
        out = patch_tool(
            mode="replace", path=str(victim),
            old_string="secret", new_string="tampered",
        )
        assert "Access denied" in out
        # The denied file must be untouched.
        assert victim.read_text(encoding="utf-8") == "secret"

    def test_patch_v4a_denied_outside_root(self, restricted):
        from tools.file_tools import patch_tool

        _allowed, denied = restricted
        victim = denied / "money.csv"
        v4a = (
            "*** Begin Patch\n"
            f"*** Update File: {victim}\n"
            "@@\n"
            "-secret\n"
            "+tampered\n"
            "*** End Patch\n"
        )
        out = patch_tool(mode="patch", patch=v4a)
        assert "Access denied" in out
        assert victim.read_text(encoding="utf-8") == "secret"

    def test_patch_v4a_add_file_denied_outside_root(self, restricted):
        from tools.file_tools import patch_tool

        _allowed, denied = restricted
        target = denied / "new_file.txt"
        v4a = (
            "*** Begin Patch\n"
            f"*** Add File: {target}\n"
            "+planted\n"
            "*** End Patch\n"
        )
        out = patch_tool(mode="patch", patch=v4a)
        assert "Access denied" in out
        assert not target.exists()

    def test_patch_v4a_move_destination_denied(self, restricted):
        """Both Move endpoints are guarded, not just the source."""
        from tools.file_tools import patch_tool

        allowed, denied = restricted
        src = allowed / "movable.txt"
        src.write_text("data", encoding="utf-8")
        dst = denied / "exfiltrated.txt"
        v4a = (
            "*** Begin Patch\n"
            f"*** Move File: {src} -> {dst}\n"
            "*** End Patch\n"
        )
        out = patch_tool(mode="patch", patch=v4a)
        assert "Access denied" in out
        assert not dst.exists()

    def test_patch_allowed_inside_root(self, restricted):
        from tools.file_tools import patch_tool

        allowed, _denied = restricted
        target = allowed / "editable.txt"
        target.write_text("before", encoding="utf-8")
        out = patch_tool(
            mode="replace", path=str(target),
            old_string="before", new_string="after",
        )
        assert "Access denied" not in out

    def test_unrestricted_profile_dispatch_unaffected(self, temp_home):
        """A profile with no allowlist entry sees no behavior change."""
        from tools.file_tools import read_file_tool

        secret = temp_home.path / "private"
        secret.mkdir()
        target = secret / "money.csv"
        target.write_text("secret", encoding="utf-8")
        temp_home.write_config("""
            profile_fs_allowlist:
              clientbot:
                - /srv/allowed
        """)
        temp_home.activate("personal")
        out = read_file_tool(str(target))
        assert "Access denied" not in out
