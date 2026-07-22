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
"""

import os
import tempfile
from unittest.mock import patch

import tools.profile_fs_guard as guard


def _with_allowlist(mapping, active_profile):
    """Context helper: force the cached allowlist and the active profile."""
    guard.reset_cache()
    # Patch _load_allowlist to return a pre-resolved map (keys lowercased,
    # roots realpath'd) so tests don't depend on an on-disk config.yaml.
    resolved = {
        p.lower(): [os.path.realpath(os.path.expanduser(r)) for r in roots]
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
