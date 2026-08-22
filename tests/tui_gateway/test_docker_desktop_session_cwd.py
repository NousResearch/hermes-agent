"""Docker session cwd translation for desktop-created sessions (issue #90679).

The desktop app creates sessions with a Windows host path. Under the Docker
terminal backend that path does not exist container-side, so commands fail with
``cd: D:\\projects: No such file or directory``. The gateway translates the cwd
using the configured ``terminal.docker_volumes`` mounts.

These tests exercise the real helpers through ``tui_gateway.server``, which is
where ``register()`` rebinds them, rather than importing the pre-bind copies
from ``methods_session``.
"""

import pytest

import tui_gateway.server as server

_normalize = server._normalize_docker_session_cwd
_pairs = server._docker_volume_host_container_pairs


# Mount set used across the translation tests: a broad drive-root mount plus a
# nested one, so longest-prefix behaviour is observable.
VOLUMES = [
    "D:/:/hostd:rw",
    "D:/projects:/projects:rw",
    "C:/Users/me/work:/work",
]


class TestDockerVolumeParsing:
    def test_parses_windows_host_and_container_pairs(self):
        assert _pairs(["D:/projects:/projects:rw"]) == [("d:/projects", "/projects")]

    def test_drive_colon_is_not_mistaken_for_the_container_separator(self):
        # ``spec.find(":/")`` on a bare drive root finds the drive colon first;
        # the container half must still be recovered.
        assert _pairs(["D:/:/hostd:rw"]) == [("d:", "/hostd")]

    def test_accepts_backslash_hosts_and_normalizes_them(self):
        assert _pairs(["D:\\projects:/projects"]) == [("d:/projects", "/projects")]

    def test_accepts_posix_hosts(self):
        assert _pairs(["/srv/data:/data:ro"]) == [("/srv/data", "/data")]

    def test_skips_named_volumes_and_malformed_entries(self):
        # Named volumes have no host path to compare a cwd against; the rest are
        # simply unusable.
        assert _pairs(["hermes-data:/data", "", "   ", "D:/only-host", 42, None]) == []

    def test_ignores_non_list_config(self):
        assert _pairs(None) == []
        assert _pairs("D:/projects:/projects") == []


class TestNormalizeDockerSessionCwd:
    def test_maps_mount_root_to_its_container_root(self):
        assert _normalize("D:\\projects", "docker", VOLUMES) == "/projects"

    def test_preserves_the_subpath_below_the_mount(self):
        # The whole point of consulting the mounts: a session opened in a
        # subdirectory must land in that subdirectory, not at the mount root.
        assert _normalize("D:\\projects\\api\\src", "docker", VOLUMES) == "/projects/api/src"

    def test_longest_matching_mount_wins(self):
        # ``D:/`` also covers this path, but the nested mount is more specific.
        assert _normalize("D:/projects/api", "docker", VOLUMES) == "/projects/api"
        # Outside the nested mount, the broad one still applies.
        assert _normalize("D:/other/dir", "docker", VOLUMES) == "/hostd/other/dir"

    @pytest.mark.parametrize(
        "cwd",
        [
            "d:\\projects\\api",
            "D:\\PROJECTS\\api",
            "d:/projects/api",
        ],
    )
    def test_matching_is_case_insensitive_like_windows(self, cwd):
        assert _normalize(cwd, "docker", VOLUMES) == "/projects/api"

    @pytest.mark.parametrize("drive", ["G", "Z", "g", "z"])
    def test_drives_outside_c_through_f_are_translated(self, drive):
        # Secondary data disks and removable drives are common; a hardcoded
        # C–F check would leave these as host paths inside the container.
        assert _normalize(f"{drive}:\\data", "docker", VOLUMES) == "/workspace"

    def test_falls_back_to_workspace_when_no_mount_matches(self):
        assert _normalize("E:\\scratch", "docker", VOLUMES) == "/workspace"
        assert _normalize("D:\\projects", "docker", []) == "/workspace"

    def test_trailing_separators_do_not_change_the_result(self):
        assert _normalize("D:\\projects\\", "docker", VOLUMES) == "/projects"
        assert _normalize("D:/projects/api/", "docker", VOLUMES) == "/projects/api"

    @pytest.mark.parametrize(
        "cwd",
        [
            "/home/user/workspace",
            "/workspace",
            "/root/project",
            ".",
            "./src",
            "projects/hermes",
        ],
    )
    def test_non_windows_paths_pass_through_for_docker(self, cwd):
        # POSIX and relative paths are already usable inside the container.
        assert _normalize(cwd, "docker", VOLUMES) == cwd

    @pytest.mark.parametrize("backend", ["local", "ssh", "modal", "e2b"])
    def test_windows_paths_pass_through_for_non_docker_backends(self, backend):
        assert _normalize("D:\\projects", backend, VOLUMES) == "D:\\projects"

    def test_backend_matching_ignores_case_and_surrounding_whitespace(self):
        assert _normalize("D:\\projects", "DOCKER", VOLUMES) == "/projects"
        assert _normalize("D:\\projects", "  Docker  ", VOLUMES) == "/projects"
        assert _normalize("D:\\projects", "\tdocker\n", VOLUMES) == "/projects"
        assert _normalize("D:\\projects", "LOCAL", VOLUMES) == "D:\\projects"

    @pytest.mark.parametrize(
        ("cwd", "backend"),
        [
            ("", "docker"),
            ("", "local"),
            (None, "docker"),
            ("D:\\projects", ""),
            ("D:\\projects", None),
        ],
    )
    def test_missing_cwd_or_backend_returns_the_input_unchanged(self, cwd, backend):
        assert _normalize(cwd, backend, VOLUMES) == cwd

    def test_omitting_volumes_keeps_the_workspace_fallback(self):
        # Callers that cannot supply the config still get a usable container cwd.
        assert _normalize("D:\\projects", "docker") == "/workspace"


class TestSessionCreateWiring:
    """The handler must consult the config and translate the session's cwd."""

    def test_session_create_reads_backend_and_volumes_from_config(self):
        # Guard the wiring the unit tests above cannot see: session.create has
        # to pass terminal.docker_volumes through, not just the backend.
        handler = server._methods["session.create"]
        names = handler.__code__.co_names

        assert "_normalize_docker_session_cwd" in names, (
            "session.create no longer normalizes the session cwd"
        )
        assert "docker_volumes" in handler.__code__.co_varnames, (
            "session.create must read terminal.docker_volumes and pass it to "
            "the normalizer, otherwise every host path collapses to /workspace"
        )

    def test_normalizer_is_bound_on_server_globals(self):
        # Handlers are rebound onto server.py's namespace by register(), so a
        # helper left behind in methods_session would raise NameError at call
        # time even though the module imports fine.
        handler = server._methods["session.create"]

        assert "_normalize_docker_session_cwd" in handler.__globals__
        assert "_docker_volume_host_container_pairs" in handler.__globals__
