"""Unit tests for the pure bwrap argv builder in tools.environments.bubblewrap.

The builder never touches the host: every path is passed in, so these tests
run on any platform, with or without bwrap installed.
"""

import logging
from pathlib import Path

import pytest

from tools.environments.bubblewrap import (
    BindMount,
    BubblewrapConfig,
    PROFILE_NAMES,
    SENSITIVE_HOME_PATHS,
    build_bwrap_args,
    load_bubblewrap_config,
)

BWRAP = "/usr/bin/bwrap"

NAMESPACE_FLAGS = ("--unshare-all", "--die-with-parent", "--new-session", "--unshare-cgroup-try")
BASE_MOUNTS = (
    ("--ro-bind", "/", "/"),
    ("--dev", "/dev"),
    ("--proc", "/proc"),
    ("--tmpfs", "/tmp"),
)


@pytest.fixture
def paths(tmp_path):
    home = tmp_path / "home"
    hermes_home = home / ".hermes"
    return {
        "initial_cwd": str(tmp_path / "work"),
        "state_dir": str(hermes_home / "sandboxes" / "bwrap-abc123"),
        "home": str(home),
        "hermes_home": str(hermes_home),
        "tracked_cwd": str(tmp_path / "work"),
    }


def build(config=None, bwrap_path=BWRAP, **overrides):
    """Build argv with the shared fixture paths, overriding any of them."""
    kwargs = dict(overrides.pop("paths"))
    kwargs.update(overrides)
    return build_bwrap_args(config or BubblewrapConfig(), bwrap_path=bwrap_path, **kwargs)


def triples(argv, flag):
    """Return the (src, dest) pairs that follow every occurrence of *flag*."""
    return [(argv[i + 1], argv[i + 2]) for i, a in enumerate(argv) if a == flag]


def contains_sequence(argv, seq):
    n = len(seq)
    return any(tuple(argv[i:i + n]) == tuple(seq) for i in range(len(argv) - n + 1))


class TestNamespaceAndBaseMounts:
    def test_starts_with_bwrap_path(self, paths):
        argv = build(paths=paths)
        assert argv[0] == BWRAP

    def test_namespace_flags_present(self, paths):
        argv = build(paths=paths)
        for flag in NAMESPACE_FLAGS:
            assert flag in argv

    def test_base_mounts_present(self, paths):
        argv = build(paths=paths)
        for seq in BASE_MOUNTS:
            assert contains_sequence(argv, seq), seq

    def test_root_ro_bind_precedes_every_other_mount(self, paths):
        argv = build(paths=paths)
        root = argv.index("--ro-bind")
        for flag in ("--dev", "--proc", "--tmpfs", "--bind"):
            if flag in argv:
                assert argv.index(flag) > root

    def test_argv_ends_with_option_terminator(self, paths):
        argv = build(paths=paths)
        assert argv[-1] == "--"

    def test_state_dir_bound_read_write(self, paths):
        argv = build(paths=paths)
        assert (paths["state_dir"], paths["state_dir"]) in triples(argv, "--bind")


class TestProfiles:
    def test_default_profile_is_network(self):
        assert BubblewrapConfig().profile == "network"

    @pytest.mark.parametrize("profile,expect_net", [
        ("restricted", False),
        ("workspace", False),
        ("network", True),
    ])
    def test_share_net_only_for_network(self, paths, profile, expect_net):
        argv = build(BubblewrapConfig(profile=profile), paths=paths)
        assert ("--share-net" in argv) is expect_net

    @pytest.mark.parametrize("profile,expect_bind", [
        ("restricted", False),
        ("workspace", True),
        ("network", True),
    ])
    def test_cwd_writable_only_for_workspace_and_network(self, paths, profile, expect_bind):
        argv = build(BubblewrapConfig(profile=profile), paths=paths)
        cwd = paths["initial_cwd"]
        assert ((cwd, cwd) in triples(argv, "--bind")) is expect_bind
        # Restricted still binds the cwd, read-only, so --chdir resolves when
        # the cwd lives under the masked /tmp.
        assert ((cwd, cwd) in triples(argv, "--ro-bind")) is not expect_bind

    def test_unknown_profile_raises_listing_valid_names(self, paths):
        with pytest.raises(ValueError) as excinfo:
            build(BubblewrapConfig(profile="bogus"), paths=paths)
        message = str(excinfo.value)
        assert "bogus" in message
        for name in ("restricted", "workspace", "network"):
            assert name in message

    def test_profile_names_are_the_three_documented(self):
        assert set(PROFILE_NAMES) == {"restricted", "workspace", "network"}


class TestTrackedCwd:
    def test_chdir_uses_tracked_cwd(self, paths):
        tracked = "/usr/share"
        argv = build(paths=paths, tracked_cwd=tracked)
        assert contains_sequence(argv, ("--chdir", tracked))

    def test_tracked_cwd_never_feeds_a_mount(self, paths):
        tracked = str(Path(paths["home"]) / "elsewhere")
        argv = build(paths=paths, tracked_cwd=tracked)
        mount_paths = {p for flag in ("--bind", "--ro-bind") for pair in triples(argv, flag) for p in pair}
        assert tracked not in mount_paths
        assert argv.count("--chdir") == 1

    def test_mount_set_identical_across_tracked_cwd_changes(self, paths):
        first = build(paths=paths)
        second = build(paths=paths, tracked_cwd="/usr/share")
        strip = lambda argv: [a for i, a in enumerate(argv) if a != "--chdir" and argv[i - 1] != "--chdir"]
        assert strip(first) == strip(second)


class TestExtraBinds:
    def test_plain_bind_added_with_requested_mode(self, paths, tmp_path):
        rw = BindMount(src=str(tmp_path / "data"), dest="/data", readonly=False)
        ro = BindMount(src=str(tmp_path / "ref"), dest=str(tmp_path / "ref"), readonly=True)
        argv = build(BubblewrapConfig(binds=(rw, ro)), paths=paths)
        assert (rw.src, "/data") in triples(argv, "--bind")
        assert (ro.src, ro.dest) in triples(argv, "--ro-bind")

    def test_sensitive_bind_dropped_with_warning(self, paths, tmp_path, caplog):
        plain = BindMount(src=str(tmp_path / "data"), dest=str(tmp_path / "data"), readonly=False)
        secret = BindMount(src=str(Path(paths["home"]) / ".ssh" / "keys"), dest="/keys", readonly=True)
        with caplog.at_level(logging.WARNING, logger="tools.environments.bubblewrap"):
            argv = build(BubblewrapConfig(binds=(plain, secret)), paths=paths)
        baseline = build(paths=paths)
        extra_pairs = (len(triples(argv, "--bind")) + len(triples(argv, "--ro-bind"))) - (
            len(triples(baseline, "--bind")) + len(triples(baseline, "--ro-bind"))
        )
        assert extra_pairs == 1
        assert (plain.src, plain.dest) in triples(argv, "--bind")
        assert not any(secret.src in pair for pair in triples(argv, "--ro-bind"))
        assert any(secret.src in rec.getMessage() for rec in caplog.records if rec.levelno >= logging.WARNING)

    @pytest.mark.parametrize("rel", sorted(SENSITIVE_HOME_PATHS))
    def test_every_sensitive_home_path_is_rejected(self, paths, rel):
        src = str(Path(paths["home"]) / rel)
        argv = build(BubblewrapConfig(binds=(BindMount(src=src, dest=src),)), paths=paths)
        assert (src, src) not in triples(argv, "--ro-bind")
        assert (src, src) not in triples(argv, "--bind")

    def test_hermes_home_bind_dropped(self, paths, caplog):
        src = str(Path(paths["hermes_home"]) / "config.yaml")
        with caplog.at_level(logging.WARNING, logger="tools.environments.bubblewrap"):
            argv = build(BubblewrapConfig(binds=(BindMount(src=src, dest="/cfg"),)), paths=paths)
        assert (src, "/cfg") not in triples(argv, "--ro-bind")
        assert any(src in rec.getMessage() for rec in caplog.records)

    def test_sibling_of_sensitive_path_is_kept(self, paths):
        src = str(Path(paths["home"]) / ".sshfs")
        argv = build(BubblewrapConfig(binds=(BindMount(src=src, dest=src),)), paths=paths)
        assert (src, src) in triples(argv, "--ro-bind")

    def test_symlinked_source_into_sensitive_path_is_dropped(self, paths, tmp_path):
        home = Path(paths["home"])
        (home / ".ssh").mkdir(parents=True)
        link = tmp_path / "innocent"
        link.symlink_to(home / ".ssh")
        argv = build(BubblewrapConfig(binds=(BindMount(src=str(link), dest="/x"),)), paths=paths)
        assert (str(link), "/x") not in triples(argv, "--ro-bind")


class TestLoadConfig:
    def test_defaults_when_env_is_empty(self):
        config = load_bubblewrap_config({})
        assert config.profile == "network"
        assert config.binds == ()
        assert config.memory_mb == 256
        assert config.cpu_seconds == 30
        assert config.max_procs == 256

    def test_reads_every_terminal_bubblewrap_env_name(self):
        env = {
            "TERMINAL_BUBBLEWRAP_PROFILE": "restricted",
            "TERMINAL_BUBBLEWRAP_BINDS": '[{"src": "/data", "dest": "/mnt/data", "readonly": false}, {"src": "/ref"}]',
            "TERMINAL_BUBBLEWRAP_MEMORY_MB": "512",
            "TERMINAL_BUBBLEWRAP_CPU_SECONDS": "0",
            "TERMINAL_BUBBLEWRAP_MAX_PROCS": "64",
        }
        config = load_bubblewrap_config(env)
        assert config.profile == "restricted"
        assert config.binds == (
            BindMount(src="/data", dest="/mnt/data", readonly=False),
            BindMount(src="/ref", dest="/ref", readonly=True),
        )
        assert config.memory_mb == 512
        assert config.cpu_seconds == 0
        assert config.max_procs == 64

    def test_profile_is_normalized(self):
        assert load_bubblewrap_config({"TERMINAL_BUBBLEWRAP_PROFILE": " Workspace "}).profile == "workspace"

    def test_blank_values_fall_back_to_defaults(self):
        env = {k: "" for k in (
            "TERMINAL_BUBBLEWRAP_PROFILE", "TERMINAL_BUBBLEWRAP_BINDS",
            "TERMINAL_BUBBLEWRAP_MEMORY_MB", "TERMINAL_BUBBLEWRAP_CPU_SECONDS",
            "TERMINAL_BUBBLEWRAP_MAX_PROCS",
        )}
        assert load_bubblewrap_config(env) == BubblewrapConfig()

    @pytest.mark.parametrize("value", ["not json", "{}", '["/data"]', '[{"dest": "/x"}]'])
    def test_malformed_binds_raise(self, value):
        with pytest.raises(ValueError, match="TERMINAL_BUBBLEWRAP_BINDS"):
            load_bubblewrap_config({"TERMINAL_BUBBLEWRAP_BINDS": value})

    @pytest.mark.parametrize("name", [
        "TERMINAL_BUBBLEWRAP_MEMORY_MB", "TERMINAL_BUBBLEWRAP_CPU_SECONDS", "TERMINAL_BUBBLEWRAP_MAX_PROCS",
    ])
    @pytest.mark.parametrize("value", ["abc", "-1", "1.5"])
    def test_bad_limits_raise(self, name, value):
        with pytest.raises(ValueError, match=name):
            load_bubblewrap_config({name: value})

    def test_loaded_config_drives_the_builder(self, paths):
        config = load_bubblewrap_config({"TERMINAL_BUBBLEWRAP_PROFILE": "restricted"})
        argv = build(config, paths=paths)
        assert "--share-net" not in argv
        assert (paths["initial_cwd"], paths["initial_cwd"]) not in triples(argv, "--bind")
