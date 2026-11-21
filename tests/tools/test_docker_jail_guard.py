"""Jail duplicate-mount guard (t_44fdd37a): a spawn whose task bucket misses label reuse must
adopt, refuse, or proceed based on the host paths running hermes containers already mount."""
import json
import logging
import os
import subprocess
import sys

import pytest

from tools.environments import docker as docker_env
from tools.environments.path_utils import sanitize_task_id_for_path


@pytest.fixture(autouse=True)
def _stable_tempdir(monkeypatch, tmp_path):
    """pytest's tmp_path sits under the system tempdir, whose mounts count as volatile: that would
    hide every per-task sandbox dir from the guard and blank them out of candidate fingerprints."""
    (tmp_path / "proc-tmp").mkdir()
    monkeypatch.setattr(docker_env.tempfile, "gettempdir", lambda: str(tmp_path / "proc-tmp"))


def _jail_candidate(tmp_path, *, profile="bot_1", egress="off", image="python:3.11", net="default",
                    task="default", jail="/srv/jail", extra=()):
    """Inspect answers for a running persistent jail held under *task*'s sandbox bucket
    (``jail=None``: only the per-task sandbox dirs, the RL rollout / per-session shape)."""
    bucket = tmp_path / "sandboxes" / "docker" / sanitize_task_id_for_path(task)
    for sub in ("home", "workspace"):
        (bucket / sub).mkdir(parents=True, exist_ok=True)
    binds = [(str(bucket / "home"), "/root"), (str(bucket / "workspace"), "/workspace"),
             *([(jail, "/home/bot")] if jail else []), *extra]
    # The fingerprint the matching spawn stamps: its sandbox binds, then its volumes, in argv order.
    fingerprint = docker_env._reuse_environment_fingerprint(
        image=image, mount_args=[arg for s, d in binds for arg in ("-v", f"{s}:{d}")],
        hermes_home=str(docker_env.get_hermes_home()))
    return {
        "labels": {"hermes-agent": "1", "hermes-profile": profile, "hermes-task-id": task,
                   "hermes-egress": egress, "hermes-environment": fingerprint},
        "mounts": [{"Type": "bind", "Source": s, "Destination": d, "RW": True} for s, d in binds],
        "image": image, "net": net}


def _mock_jail_guard(monkeypatch, tmp_path, candidates, *, conflict_ps_rc=0, image_available=True,
                     labeled=None, rm_rc=0, rm_stderr=""):
    """Label-reuse probe misses unless *labeled* (``(cid, state)``) names a by-label hit; the conflict
    probe lists *candidates* (cid -> answers, ``"raw"`` overriding the labels+mounts inspect output).
    ``docker rm`` exits *rm_rc* with *rm_stderr*; every ``docker exec`` finds its container gone.
    Returns the captured argv list."""
    monkeypatch.setenv("TERMINAL_SANDBOX_DIR", str(tmp_path / "sandboxes"))
    # The snapshot bootstrap only execs into the container; nothing here depends on it.
    monkeypatch.setattr(docker_env.DockerEnvironment, "init_session", lambda self: None)
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    monkeypatch.setattr(docker_env, "_get_active_profile_name", lambda: "bot_1")
    monkeypatch.setattr(docker_env, "_readonly_skill_mount_args", lambda: [])
    docker_env._cgroup_limits_ok = True
    calls = []

    def _inspect(c, fmt):
        if fmt == "{{.Config.Image}}":
            return c["image"]
        if fmt == "{{.HostConfig.NetworkMode}}":
            return c["net"]
        return c.get("raw") or json.dumps({"labels": c["labels"], "mounts": c["mounts"]})

    def _run(cmd, **kwargs):
        calls.append(list(cmd))
        result = _answer(cmd)
        if kwargs.get("check") and result.returncode:
            raise subprocess.CalledProcessError(result.returncode, cmd, result.stdout, result.stderr)
        return result

    def _answer(cmd):
        done = lambda rc=0, out="", err="": subprocess.CompletedProcess(cmd, rc, stdout=out, stderr=err)
        sub = cmd[1]
        if sub == "version":
            return done(out="Docker version")
        if sub == "ps" and "-a" in cmd:
            return done(out="\t".join(labeled) if labeled else "")  # usually a miss: the bucket differs
        if sub == "ps":
            assert "status=running" in cmd and not any("hermes-task-id=" in a for a in cmd)
            return done(conflict_ps_rc, "\n".join(candidates))
        if sub == "inspect" and cmd[-1] in candidates:
            return done(out=_inspect(candidates[cmd[-1]], cmd[cmd.index("--format") + 1]))
        if sub == "run":
            return done(out="fresh-cid\n")
        if sub == "rm":
            return done(rm_rc, err=rm_stderr)
        if sub == "image" and cmd[2] == "inspect":
            return done(0 if image_available else 1, out="sha256:img\n")
        if sub == "pull":
            return done(0 if image_available else 1)
        return done()

    def _exec(cmd, stdin_data=None, **kwargs):
        calls.append(list(cmd))
        return popen_bash([sys.executable, "-c", "print('Error response from daemon: No such container')\n"
                                                 "raise SystemExit(1)"], stdin_data)

    popen_bash = docker_env._popen_bash
    monkeypatch.setattr(docker_env.subprocess, "run", _run)
    monkeypatch.setattr(docker_env, "_popen_bash", _exec)
    return calls


def _jail_spawn(**kwargs):
    kwargs = {"image": "python:3.11", "task_id": "profile:forge", "persistent_filesystem": True,
              "volumes": ["/srv/jail:/home/bot"], **kwargs}
    return docker_env.DockerEnvironment(**kwargs)


def _docker_runs(calls, *subs):
    return [c for c in calls if c[1] in (subs or ("run",))]


def test_jail_guard_candidate_fixture_mirrors_spawn_labels(monkeypatch, tmp_path):
    """A gate label the fixture omits makes every gate on it pass vacuously — how fingerprint-blind
    default adoption stayed green. The identical twin carries exactly the spawn's labels."""
    # Every label the adoption path in tools/environments/docker.py reads (ps filter, classifier).
    gate_labels = {"hermes-agent", "hermes-profile", "hermes-task-id", "hermes-egress", "hermes-environment"}
    _mock_jail_guard(monkeypatch, tmp_path, {})

    labels = _jail_candidate(tmp_path)["labels"]

    assert labels.keys() >= gate_labels
    assert labels == _jail_spawn(task_id="default")._labels


def test_jail_guard_adopts_when_only_task_bucket_differs(monkeypatch, tmp_path):
    """The live leak: a ``profile:forge`` spawn missed the ``default``-labeled jail and
    double-mounted /srv/jail. Identical mounts modulo the sandbox bucket must adopt it."""
    calls = _mock_jail_guard(monkeypatch, tmp_path, {"jail-cid": _jail_candidate(tmp_path)})

    env = _jail_spawn()

    # Required within one bucket, ignored across buckets: the hash embeds the bucket paths.
    assert "hermes-environment" in env._labels
    assert env._container_id == "jail-cid"
    assert not _docker_runs(calls)


def test_jail_guard_adopts_regardless_of_inspect_mount_order(monkeypatch, tmp_path):
    """``docker inspect`` lists ``.Mounts`` in map order, not argv order: a live jail with the
    skills/cache binds came back shuffled, so an argv-ordered comparison never adopted."""
    candidate = _jail_candidate(tmp_path)
    candidate["mounts"].reverse()
    calls = _mock_jail_guard(monkeypatch, tmp_path, {"jail-cid": candidate})

    assert _jail_spawn()._container_id == "jail-cid"
    assert not _docker_runs(calls)


@pytest.mark.parametrize("candidate_kw, spawn_kw", [
    ({"egress": "eg123"}, {}),
    ({"extra": [("/srv/extra", "/data")]}, {}),
    # tmpfs /root cannot be proven equivalent to a bound /root, so adoption is impossible.
    ({}, {"persistent_filesystem": False}),
    ({"net": "bridge"}, {"network": False}),
    ({"image": "other:tag"}, {"image_pinned": True}),
], ids=["egress-mismatch", "extra-mount", "tmpfs-spawn", "air-gap-vs-bridge", "pinned-image-mismatch"])
def test_jail_guard_same_profile_conflicting_identity_refuses(monkeypatch, tmp_path, candidate_kw, spawn_kw):
    calls = _mock_jail_guard(monkeypatch, tmp_path, {"jail-cid": _jail_candidate(tmp_path, **candidate_kw)})

    with pytest.raises(RuntimeError, match="already bind-mounted read-write") as err:
        _jail_spawn(**spawn_kw)
    assert "/srv/jail" in str(err.value) and "jail-cid" in str(err.value)
    assert not _docker_runs(calls)


@pytest.mark.parametrize("candidate_kw, spawn_kw, mock_kw", [
    # A read-only second reader is a legitimate access pattern, never a conflict.
    ({"egress": "eg123"}, {"volumes": ["/srv/jail:/home/bot:ro"]}, {}),
    ({"egress": "eg123"}, {}, {"conflict_ps_rc": 125}),
    ({"egress": "eg123", "raw": "not-json"}, {}, {}),
    ({"mounts": [{"Type": "bind", "Source": "/srv/other", "Destination": "/data", "RW": True}]}, {}, {}),
], ids=["readonly-spawn", "probe-failure", "unparseable-candidate", "disjoint-jail"])
def test_jail_guard_proceeds_with_fresh_container(monkeypatch, tmp_path, candidate_kw, spawn_kw, mock_kw):
    candidate = _jail_candidate(tmp_path, egress=candidate_kw.pop("egress", "off"))
    calls = _mock_jail_guard(monkeypatch, tmp_path, {"jail-cid": {**candidate, **candidate_kw}}, **mock_kw)

    env = _jail_spawn(**spawn_kw)

    assert env._container_id == "fresh-cid" and _docker_runs(calls)


def test_jail_guard_foreign_profile_overlap_warns_and_proceeds(monkeypatch, tmp_path, caplog):
    calls = _mock_jail_guard(monkeypatch, tmp_path, {"jail-cid": _jail_candidate(tmp_path, profile="other")})

    with caplog.at_level(logging.WARNING, logger="tools.environments.docker"):
        env = _jail_spawn()

    assert env._container_id == "fresh-cid" and _docker_runs(calls)
    assert "/srv/jail" in caplog.text


def test_jail_guard_persist_across_processes_false_skips_probe(monkeypatch, tmp_path):
    calls = _mock_jail_guard(monkeypatch, tmp_path, {"jail-cid": _jail_candidate(tmp_path, egress="eg123")})

    env = _jail_spawn(persist_across_processes=False)

    assert not [c for c in calls if c[1] == "ps" and "status=running" in c]
    assert env._container_id == "fresh-cid"


def test_jail_guard_two_adoptables_take_first_and_warn_second(monkeypatch, tmp_path, caplog):
    candidates = {"cidA": _jail_candidate(tmp_path), "cidB": _jail_candidate(tmp_path)}
    calls = _mock_jail_guard(monkeypatch, tmp_path, candidates)

    with caplog.at_level(logging.WARNING, logger="tools.environments.docker"):
        env = _jail_spawn()

    assert env._container_id == "cidA" and not _docker_runs(calls)
    assert "cidB" in caplog.text


def test_jail_guard_image_mismatch_never_adopts_across_buckets(monkeypatch, tmp_path):
    """Cross-bucket adoption needs the exact image, pinned or not: an RL rollout's per-task
    ``docker_image`` override must not run inside the user's default jail. Over a real jail
    path that is the leak class, so the spawn refuses."""
    calls = _mock_jail_guard(monkeypatch, tmp_path, {"jail-cid": _jail_candidate(tmp_path, image="other:tag")})

    with pytest.raises(RuntimeError, match="already bind-mounted read-write"):
        _jail_spawn(image_pinned=False)
    assert not _docker_runs(calls)


@pytest.mark.parametrize("theirs, ours", [
    ("rollout:one", "rollout:two"), ("session:1", "session:2"), ("rollout:one", "default")])
def test_jail_guard_per_task_buckets_coexist(monkeypatch, tmp_path, theirs, ours):
    """RL/override rollouts and per-session isolation run distinct task buckets under one profile on
    purpose: an overlap of only the (tokenized) per-task sandbox dirs neither adopts nor refuses —
    not even for a default spawn, which would otherwise run inside the rollout's container."""
    calls = _mock_jail_guard(monkeypatch, tmp_path, {"cid": _jail_candidate(tmp_path, task=theirs, jail=None)})

    env = _jail_spawn(task_id=ours, volumes=[])

    assert env._container_id == "fresh-cid" and _docker_runs(calls)


def test_jail_guard_non_default_real_path_overlap_warns_and_proceeds(monkeypatch, tmp_path, caplog):
    """No default side is outside the leak class: two rollout buckets pointed at one real path
    worked before the guard, so it only warns and gets a fresh container."""
    calls = _mock_jail_guard(monkeypatch, tmp_path, {"jail-cid": _jail_candidate(tmp_path, task="rollout:seven")})

    with caplog.at_level(logging.WARNING, logger="tools.environments.docker"):
        env = _jail_spawn()

    assert env._container_id == "fresh-cid" and _docker_runs(calls)
    assert "/srv/jail" in caplog.text


@pytest.mark.parametrize("task", ["rollout:seven", "default"], ids=["vs-bucket", "vs-default"])
def test_jail_guard_default_spawn_divergent_binds_refuses(monkeypatch, tmp_path, task):
    """A default spawn whose real jail path a differently-mounted container already holds refuses.
    vs-bucket: the narrowed refusal still closes the leak class. vs-default: replace needs
    bind-model equality — the sibling may hold a live sandbox, so divergence never deletes it."""
    calls = _mock_jail_guard(monkeypatch, tmp_path, {"jail-cid": _jail_candidate(
        tmp_path, task=task, extra=[("/srv/x", "/x")])})

    with pytest.raises(RuntimeError, match="/srv/jail"):
        _jail_spawn(task_id="default")
    assert not _docker_runs(calls, "rm", "run")


def test_jail_guard_default_spawn_adopts_forge_jail(monkeypatch, tmp_path):
    """The default side may be ours: a default spawn re-joins a leaked ``profile:forge`` jail."""
    calls = _mock_jail_guard(monkeypatch, tmp_path, {"jail-cid": _jail_candidate(tmp_path, task="profile:forge")})

    assert _jail_spawn(task_id="default")._container_id == "jail-cid"
    assert not _docker_runs(calls)


@pytest.mark.parametrize("candidate_kw, spawn_kw", [
    ({"egress": "eg123"}, {}),
    ({"image": "other:tag"}, {"image_pinned": True}),
    # Invisible to the bind model: only the environment fingerprint tells the twin is stale.
    ({"jail": None}, {"volumes": ["cache:/data"]}),
    ({}, {"volumes": ["/srv/jail:/home/bot:z"]}),
], ids=["egress-change", "pinned-image-change", "named-volume", "mount-option"])
def test_jail_guard_default_config_drift_replaces(monkeypatch, tmp_path, candidate_kw, spawn_kw):
    """``hermes egress enable`` / a re-pinned image / an added named volume on a live profile: the
    stale default container is removed and recreated, as label reuse does, instead of being
    adopted (silently dropping the new config) or wedging every terminal call."""
    calls = _mock_jail_guard(monkeypatch, tmp_path, {"jail-cid": _jail_candidate(tmp_path, **candidate_kw)})

    env = _jail_spawn(task_id="default", **spawn_kw)

    assert ["/usr/bin/docker", "rm", "-f", "jail-cid"] in calls
    assert env._container_id == "fresh-cid" and _docker_runs(calls)


def test_jail_guard_egress_drift_refuses_when_image_unavailable(monkeypatch, tmp_path):
    """An unpullable replacement keeps the old container only for pure image drift: attaching a
    pre-egress container would bypass the firewall, so egress drift refuses (nothing removed)."""
    calls = _mock_jail_guard(monkeypatch, tmp_path,
                             {"jail-cid": _jail_candidate(tmp_path, egress="eg123")},
                             image_available=False)

    with pytest.raises(RuntimeError, match="jail-cid"):
        _jail_spawn(task_id="default")
    assert not _docker_runs(calls, "rm", "run")


def test_jail_guard_unpinned_image_flip_keeps_existing_sandbox(monkeypatch, tmp_path, caplog):
    """Bind-equal, egress-equal, only the unpinned default image moved: keep the sandbox someone
    has state in (``_attach_existing_container``'s policy), even with the new image unpullable."""
    calls = _mock_jail_guard(monkeypatch, tmp_path, {"jail-cid": _jail_candidate(tmp_path, image="other:tag")},
                             image_available=False)

    with caplog.at_level(logging.WARNING, logger="tools.environments.docker"):
        env = _jail_spawn(task_id="default", image_pinned=False)

    assert env._container_id == "jail-cid"
    assert not _docker_runs(calls, "rm", "run")
    assert "other:tag" in caplog.text and "python:3.11" in caplog.text


def test_jail_guard_refusal_names_literal_sandbox_path(monkeypatch, tmp_path):
    """With no real path shared, a default↔default refusal names the sandbox dir both buckets
    share as the host path it is, never the internal ``<sandbox>`` token."""
    _mock_jail_guard(monkeypatch, tmp_path, {"jail-cid": _jail_candidate(
        tmp_path, jail=None, extra=[("/srv/extra", "/data")])})

    with pytest.raises(RuntimeError) as err:
        _jail_spawn(task_id="default", volumes=[])
    assert "<sandbox>" not in str(err.value)
    assert os.path.realpath(tmp_path / "sandboxes" / "docker" / "default" / "home") in str(err.value)


def test_jail_guard_failed_removal_refuses(monkeypatch, tmp_path):
    """A drift replacement whose ``docker rm`` fails must not run alongside the survivor."""
    calls = _mock_jail_guard(monkeypatch, tmp_path, {"jail-cid": _jail_candidate(tmp_path, egress="eg123")},
                             rm_rc=1)

    with pytest.raises(RuntimeError, match="Could not remove .*jail-cid"):
        _jail_spawn(task_id="default")
    assert not _docker_runs(calls)


def test_jail_guard_removal_raced_by_another_remover_proceeds(monkeypatch, tmp_path):
    """A concurrent remover beat our ``docker rm -f``: the container is gone, which is the goal."""
    calls = _mock_jail_guard(monkeypatch, tmp_path, {"jail-cid": _jail_candidate(tmp_path, egress="eg123")},
                             rm_rc=1, rm_stderr="Error response from daemon: No such container: jail-cid")

    env = _jail_spawn(task_id="default")

    assert env._container_id == "fresh-cid" and _docker_runs(calls)


def test_jail_guard_default_spawn_adopts_identical_twin_after_failed_probe(monkeypatch, tmp_path):
    """No host volumes, so the default twin shares only the sandbox dirs: when the label-reuse
    probe misses it (timeout, race), the spawn adopts it instead of removing the live shared jail."""
    calls = _mock_jail_guard(monkeypatch, tmp_path, {"twin-cid": _jail_candidate(tmp_path, jail=None)})

    env = _jail_spawn(task_id="default", volumes=[])

    assert env._container_id == "twin-cid"
    assert not _docker_runs(calls, "rm", "run")


def test_jail_guard_stopped_label_hit_adopts_running_twin(monkeypatch, tmp_path):
    """Daemon restart, forge recovered first: starting the stopped default jail would make two
    running holders of one path. The by-label start is mount-gated and adopts the running twin."""
    calls = _mock_jail_guard(monkeypatch, tmp_path, {"forge-cid": _jail_candidate(tmp_path, task="profile:forge")},
                             labeled=("stopped-cid", "exited"))

    env = _jail_spawn(task_id="default")

    assert env._container_id == "forge-cid"
    assert not _docker_runs(calls, "start", "run")


def test_jail_guard_recovery_restarts_adopted_jail(monkeypatch, tmp_path):
    """An adopter's recovery restarts the jail it adopted — whose default labels its own label
    search misses — instead of ``docker run``-ing a duplicate on the jail path."""
    candidates = {"jail-cid": _jail_candidate(tmp_path)}
    calls = _mock_jail_guard(monkeypatch, tmp_path, candidates)
    env = _jail_spawn()
    assert env._container_id == "jail-cid" and env._labels["hermes-task-id"] != "default"
    candidates.clear()  # the daemon restarted: the jail exists, stopped

    assert env._recreate_container() is True
    assert env._container_id == "jail-cid"
    assert ["/usr/bin/docker", "start", "jail-cid"] in calls
    assert not _docker_runs(calls)


def test_jail_guard_drift_replacement_waits_for_refusal(monkeypatch, tmp_path):
    """A leaked same-profile holder of the jail path refuses before the default one is removed:
    replacing would only double-mount the path the refusal names."""
    candidates = {"jail-cid": _jail_candidate(tmp_path, egress="eg123"),
                  "leak-cid": _jail_candidate(tmp_path, task="rollout:seven", extra=[("/srv/x", "/x")])}
    calls = _mock_jail_guard(monkeypatch, tmp_path, candidates)

    with pytest.raises(RuntimeError, match="leak-cid"):
        _jail_spawn(task_id="default")
    assert not _docker_runs(calls, "rm", "run")


@pytest.mark.parametrize("candidate_kw, volumes", [
    ({"egress": "eg123"}, ["/srv/jail:/home/bot"]),
    ({"jail": None}, ["cache:/data"]),
], ids=["egress", "fingerprint"])
def test_jail_guard_recovery_refuses_drifted_twin_and_restores_prev(monkeypatch, tmp_path, candidate_kw, volumes):
    """Exec recovery runs with labels from before the config change: removing the drifted default
    container there would delete the one its successor just recreated, and the two would trade
    removals. Recovery refuses (and fails the exec) instead, keeping the gone id so the next exec
    retries recovery rather than asserting "Container not started"."""
    candidates = {}
    calls = _mock_jail_guard(monkeypatch, tmp_path, candidates)
    env = _jail_spawn(task_id="default", volumes=volumes)
    prev_id = env._container_id
    candidates["new-cid"] = _jail_candidate(tmp_path, **candidate_kw)

    assert env._recreate_container() is False
    assert env._container_id == prev_id
    assert not _docker_runs(calls, "rm")
    assert len(_docker_runs(calls)) == 1


def test_jail_guard_refused_recovery_retries_on_next_exec(monkeypatch, tmp_path):
    """The kept gone id makes the next exec fail as container-gone and run recovery again."""
    candidates = {}
    calls = _mock_jail_guard(monkeypatch, tmp_path, candidates)
    env = _jail_spawn(task_id="default", volumes=["cache:/data"])
    prev_id = env._container_id
    candidates["twin-cid"] = _jail_candidate(tmp_path, jail=None)
    calls.clear()

    env.execute("true")
    assert len(_docker_runs(calls, "ps")) == 1
    result = env.execute("true")

    assert len(_docker_runs(calls, "ps")) == 2
    assert env._container_id == prev_id and result["returncode"] != 0


def test_jail_guard_recovery_skips_gone_container_and_adopts_other_twin(monkeypatch, tmp_path):
    """A gone container can still be listed while it dies: recovery must not re-adopt the id whose
    exec just failed, but the other exact twin."""
    candidates = {}
    calls = _mock_jail_guard(monkeypatch, tmp_path, candidates)
    env = _jail_spawn(task_id="default", volumes=[])
    assert env._container_id == "fresh-cid"
    candidates.update({"fresh-cid": _jail_candidate(tmp_path, jail=None),
                       "twin-cid": _jail_candidate(tmp_path, jail=None)})

    assert env._recreate_container() is True
    assert env._container_id == "twin-cid"
    assert len(_docker_runs(calls)) == 1


def test_jail_guard_recovery_adopts_identical_twin_no_host_volumes(monkeypatch, tmp_path):
    """Two processes share one profile's default jail with no host volumes; after an out-of-band
    ``docker rm`` the first recreates it. The second's recovery must adopt that exact-label twin,
    not refuse it and brick every later exec."""
    candidates = {}
    calls = _mock_jail_guard(monkeypatch, tmp_path, candidates)
    env = _jail_spawn(task_id="default", volumes=[])
    assert env._container_id == "fresh-cid"
    candidates["twin-cid"] = _jail_candidate(tmp_path, jail=None)

    assert env._recreate_container() is True
    assert env._container_id == "twin-cid"
    assert not _docker_runs(calls, "rm")
    assert len(_docker_runs(calls)) == 1


def test_jail_guard_canonical_bind_source_tokenizes_sandbox_buckets(tmp_path):
    root = tmp_path / "sandboxes" / "docker"
    home = root / "profile_forge-abc" / "home"
    default_home = root / "default" / "home"
    home.mkdir(parents=True)
    default_home.mkdir(parents=True)
    resolved_root = os.path.realpath(root)
    canonical = docker_env._canonical_bind_source

    assert canonical(str(home), resolved_root) == "<sandbox>/home"
    assert canonical(f"{home}/", resolved_root) == "<sandbox>/home"
    assert canonical(str(default_home), resolved_root) == canonical(str(home), resolved_root)
    assert canonical("/nonexistent/jail/", resolved_root) == "/nonexistent/jail"
    assert canonical("rel/dir/", resolved_root) == "rel/dir"


@pytest.mark.require_symlinks
def test_jail_guard_canonical_bind_source_resolves_symlinks(tmp_path):
    root = tmp_path / "sandboxes" / "docker"
    (root / "default" / "home").mkdir(parents=True)
    link = tmp_path / "home-link"
    link.symlink_to(root / "default" / "home")

    assert docker_env._canonical_bind_source(str(link), os.path.realpath(root)) == "<sandbox>/home"


def test_jail_guard_parse_mount_pair_args():
    binds, has_tmpfs = docker_env._parse_mount_pair_args([
        "-v", "/a:/x", "-v", "/b:/y:ro,z", "--mount", "type=bind,source=/c,destination=/z,readonly",
        "--mount", "type=bind,src=/d,dst=/w", "--mount", "type=volume,source=named,target=/v",
        "--tmpfs", "/tmp:rw,nosuid,size=512m", "--tmpfs", "/run:rw,noexec,nosuid,size=64m"])

    assert binds == [("/a", "/x", True), ("/b", "/y", False), ("/c", "/z", False), ("/d", "/w", True)]
    assert has_tmpfs is False  # hardening tmpfs every hermes container carries
    assert docker_env._parse_mount_pair_args(["--tmpfs", "/root:rw,exec,size=1g"])[1] is True
    assert docker_env._parse_mount_pair_args(["--mount", "type=tmpfs,destination=/root"])[1] is True


def test_jail_guard_parse_mount_pair_args_joined_flags_and_named_volumes():
    parse = docker_env._parse_mount_pair_args

    assert parse(["--volume=/a:/x", "--mount=type=bind,source=/b,target=/y,ro", "--tmpfs=/root"]) == parse(
        ["--volume", "/a:/x", "--mount", "type=bind,source=/b,target=/y,ro", "--tmpfs", "/root"])
    assert parse(["-v/a:/x"]) == parse(["-v=/a:/x"]) == parse(["-v", "/a:/x"]) == ([("/a", "/x", True)], False)
    # ``docker inspect`` lists only binds; a named volume in the spawn's model would never compare equal.
    assert parse(["-v", "named:/x", "-v", "./rel:/y"])[0] == [("./rel", "/y", True)]
