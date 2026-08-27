"""Persistent-overlay path safety for the Singularity environment.

A raw task id used as an overlay directory name carries the same class of
bug as the Docker persistent-sandbox mount path (#92414): colon-bearing
session ids break bind specs and are unrepresentable on Windows. Overlay
components must go through the shared sanitizer in tools.environments.base.
"""

import subprocess

from tools.environments import singularity as singularity_env
from tools.environments.base import sanitize_task_id_for_path


def test_snapshot_registry_is_scoped_to_active_profile(tmp_path):
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    profile_a = tmp_path / "profile-a"
    profile_b = tmp_path / "profile-b"
    profile_a.mkdir()
    profile_b.mkdir()

    token_a = set_hermes_home_override(profile_a)
    try:
        singularity_env._save_snapshots({"task-a": "/overlay/a"})
        assert singularity_env._load_snapshots() == {"task-a": "/overlay/a"}
    finally:
        reset_hermes_home_override(token_a)

    token_b = set_hermes_home_override(profile_b)
    try:
        assert singularity_env._load_snapshots() == {}
        singularity_env._save_snapshots({"task-b": "/overlay/b"})
    finally:
        reset_hermes_home_override(token_b)

    token_a2 = set_hermes_home_override(profile_a)
    try:
        assert singularity_env._load_snapshots() == {"task-a": "/overlay/a"}
    finally:
        reset_hermes_home_override(token_a2)


def _stub_singularity(monkeypatch, tmp_path):
    monkeypatch.setattr(
        singularity_env, "_ensure_singularity_available", lambda **_kwargs: "/usr/bin/apptainer"
    )
    monkeypatch.setattr(
        singularity_env,
        "_get_or_build_sif",
        lambda image, executable="apptainer", **_kwargs: str(
            tmp_path / "image.sif"
        ),
    )
    monkeypatch.setattr(singularity_env, "_get_scratch_dir", lambda: tmp_path / "scratch")
    monkeypatch.setattr(
        singularity_env.SingularityEnvironment, "_start_instance", lambda self: None
    )
    monkeypatch.setattr(
        singularity_env.SingularityEnvironment, "init_session", lambda self: None
    )


def test_persistent_overlay_dir_sanitizes_colon_in_task_id(monkeypatch, tmp_path):
    _stub_singularity(monkeypatch, tmp_path)

    task_id = "session:agent:main:telegram:dm:12345"
    env = singularity_env.SingularityEnvironment(
        image="python:3.11",
        persistent_filesystem=True,
        task_id=task_id,
    )

    assert env._overlay_dir is not None
    assert env._overlay_dir.name == f"overlay-{sanitize_task_id_for_path(task_id)}"
    assert ":" not in env._overlay_dir.name
    assert env._overlay_dir.is_dir()


def test_persistent_overlay_dir_keeps_safe_task_ids_verbatim(monkeypatch, tmp_path):
    """Existing overlays for already-safe ids must keep resolving to the same
    directory — renaming would strand a user's persistent overlay state."""
    _stub_singularity(monkeypatch, tmp_path)

    env = singularity_env.SingularityEnvironment(
        image="python:3.11",
        persistent_filesystem=True,
        task_id="default",
    )

    assert env._overlay_dir is not None
    assert env._overlay_dir.name == "overlay-default"


def test_distinct_session_keys_get_distinct_overlay_dirs(monkeypatch, tmp_path):
    """Sanitization must stay injective across backends: two ids that differ
    only in colon placement must not share one overlay."""
    _stub_singularity(monkeypatch, tmp_path)

    names = set()
    for task_id in (
        "session:agent:main:telegram:dm:111",
        "session_agent_main_telegram_dm_111",
    ):
        env = singularity_env.SingularityEnvironment(
            image="python:3.11",
            persistent_filesystem=True,
            task_id=task_id,
        )
        assert env._overlay_dir is not None
        names.add(env._overlay_dir.name)

    assert len(names) == 2, f"overlay dirs collided: {names}"


def test_same_task_uses_distinct_overlay_authority_for_distinct_images(
    monkeypatch, tmp_path
):
    _stub_singularity(monkeypatch, tmp_path)

    first = singularity_env.SingularityEnvironment(
        image="docker://example/image:one",
        persistent_filesystem=True,
        task_id="same-task",
    )
    second = singularity_env.SingularityEnvironment(
        image="docker://example/image:two",
        persistent_filesystem=True,
        task_id="same-task",
    )

    assert first._overlay_dir is not None
    assert second._overlay_dir is not None
    assert first._overlay_dir != second._overlay_dir
    assert first._overlay_dir.name == second._overlay_dir.name == "overlay-same-task"


def test_same_mutable_tag_cannot_reuse_overlay_under_multiplex(
    monkeypatch, tmp_path
):
    from agent.secret_scope import (
        build_profile_secret_scope,
        reset_secret_scope,
        set_multiplex_active,
        set_secret_scope,
    )
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    _stub_singularity(monkeypatch, tmp_path)
    source = tmp_path / "source"
    target = tmp_path / "target"
    for home in (source, target):
        home.mkdir()
        (home / ".env").write_text("", encoding="utf-8")
        (home / "config.yaml").write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(source))
    monkeypatch.setattr("hermes_constants.get_process_hermes_home", lambda: source)

    set_multiplex_active(True)
    scope_token = set_secret_scope(build_profile_secret_scope(target))
    home_token = set_hermes_home_override(target)
    try:
        first = singularity_env.SingularityEnvironment(
            image="docker://example/image:mutable",
            persistent_filesystem=True,
            task_id="same-task",
        )
        second = singularity_env.SingularityEnvironment(
            image="docker://example/image:mutable",
            persistent_filesystem=True,
            task_id="same-task",
        )
    finally:
        reset_hermes_home_override(home_token)
        reset_secret_scope(scope_token)
        set_multiplex_active(False)

    assert first._image_authority_id != second._image_authority_id
    assert first._overlay_dir is not None
    assert second._overlay_dir is not None
    assert first._overlay_dir != second._overlay_dir


def test_multiplex_singularity_quarantines_unbound_credential_mounts(
    monkeypatch, tmp_path
):
    from agent.secret_scope import ProfileEnvBoundary
    from tools import credential_files

    source_home = tmp_path / "source"
    target_home = tmp_path / "target"
    env = object.__new__(singularity_env.SingularityEnvironment)
    env.executable = "/usr/bin/apptainer"
    env.image = "docker://example/image@sha256:" + ("a" * 64)
    env.instance_id = "hermes_test"
    env._persistent = False
    env._overlay_dir = None
    env._memory = 0
    env._cpu = 0
    env._profile_env_boundary = ProfileEnvBoundary(
        source_home=source_home,
        target_home=target_home,
        source_owned_names=frozenset(),
        target_values={},
    )
    env._owner_home = target_home
    env._source_home = source_home

    monkeypatch.setattr(
        credential_files,
        "get_credential_file_mounts",
        lambda: [{"host_path": "/foreign/token", "container_path": "/token"}],
    )
    monkeypatch.setattr(credential_files, "get_skills_directory_mount", lambda: [])
    monkeypatch.setattr(singularity_env, "_singularity_subprocess_env", lambda **_kwargs: {})
    seen = {}

    def _fake_run(cmd, **_kwargs):
        seen["cmd"] = list(cmd)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(singularity_env.subprocess, "run", _fake_run)

    singularity_env.SingularityEnvironment._start_instance(env)

    assert "/foreign/token:/token:ro" not in seen["cmd"]


def test_environment_captures_owner_for_sif_and_contextless_cleanup(
    monkeypatch, tmp_path
):
    """The production object, not caller ContextVars, owns its artifacts."""
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    profile_a = tmp_path / "profile-a"
    profile_b = tmp_path / "profile-b"
    profile_a.mkdir()
    profile_b.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(profile_a))
    seen_owner = {}

    monkeypatch.setattr(
        singularity_env, "_ensure_singularity_available", lambda **_kwargs: "/usr/bin/apptainer"
    )

    def _capture_sif(image, executable="apptainer", **kwargs):
        seen_owner["home"] = kwargs.get("owner_home")
        return str(tmp_path / "image.sif")

    monkeypatch.setattr(singularity_env, "_get_or_build_sif", _capture_sif)
    monkeypatch.setattr(singularity_env, "_get_scratch_dir", lambda: tmp_path / "scratch")
    monkeypatch.setattr(
        singularity_env.SingularityEnvironment, "_start_instance", lambda self: None
    )
    monkeypatch.setattr(
        singularity_env.SingularityEnvironment, "init_session", lambda self: None
    )

    env = singularity_env.SingularityEnvironment(
        image="docker://example/image:tag",
        persistent_filesystem=True,
        task_id="same-task",
    )
    assert seen_owner["home"] == profile_a.resolve()
    assert env._owner_home == profile_a.resolve()

    token = set_hermes_home_override(profile_b)
    try:
        env.cleanup()
    finally:
        reset_hermes_home_override(token)

    snapshot_key = f"{env._image_authority_id}:{env._artifact_epoch}:same-task"
    assert singularity_env._load_snapshots(profile_a) == {
        snapshot_key: str(env._overlay_dir)
    }
    assert singularity_env._load_snapshots(profile_b) == {}


def test_sif_build_is_published_once_and_reused(monkeypatch, tmp_path):
    """A successful build must resolve to the same lookup path on reuse."""
    cache_dir = tmp_path / "cache"
    owner = tmp_path / "profile"
    owner.mkdir()
    calls = []

    monkeypatch.setattr(singularity_env, "_get_apptainer_cache_dir", lambda: cache_dir)
    monkeypatch.setattr(
        singularity_env,
        "_singularity_subprocess_env",
        lambda **_kwargs: {"PATH": "/usr/bin"},
    )

    def _fake_run(cmd, **_kwargs):
        calls.append(list(cmd))
        build_path = singularity_env.Path(cmd[2])
        build_path.parent.mkdir(parents=True, exist_ok=True)
        build_path.write_bytes(b"fake-sif")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(singularity_env.subprocess, "run", _fake_run)

    image = "docker://example/image:tag"
    first = singularity_env._get_or_build_sif(
        image, "/usr/bin/apptainer", owner_home=owner
    )
    second = singularity_env._get_or_build_sif(
        image, "/usr/bin/apptainer", owner_home=owner
    )

    assert first == second
    assert singularity_env.Path(first).read_bytes() == b"fake-sif"
    assert len(calls) == 1


def test_mutable_sif_tag_is_not_reused_across_multiplex_policy_actions(
    monkeypatch, tmp_path
):
    cache_dir = tmp_path / "cache"
    owner = tmp_path / "profile"
    owner.mkdir()
    calls = []

    monkeypatch.setattr(singularity_env, "_get_apptainer_cache_dir", lambda: cache_dir)
    monkeypatch.setattr(
        singularity_env,
        "_singularity_subprocess_env",
        lambda **_kwargs: {"PATH": "/usr/bin"},
    )

    def _fake_run(cmd, **_kwargs):
        calls.append(list(cmd))
        build_path = singularity_env.Path(cmd[2])
        build_path.parent.mkdir(parents=True, exist_ok=True)
        build_path.write_bytes(b"fake-sif")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(singularity_env.subprocess, "run", _fake_run)

    image = "docker://example/image:mutable"
    first = singularity_env._get_or_build_sif(
        image,
        "/usr/bin/apptainer",
        owner_home=owner,
        policy_generation="profile-scope-v1:generation",
    )
    second = singularity_env._get_or_build_sif(
        image,
        "/usr/bin/apptainer",
        owner_home=owner,
        policy_generation="profile-scope-v1:generation",
    )

    assert first != second
    assert len(calls) == 2


def test_digest_pinned_sif_is_reused_within_one_policy_generation(
    monkeypatch, tmp_path
):
    cache_dir = tmp_path / "cache"
    owner = tmp_path / "profile"
    owner.mkdir()
    calls = []

    monkeypatch.setattr(singularity_env, "_get_apptainer_cache_dir", lambda: cache_dir)
    monkeypatch.setattr(
        singularity_env,
        "_singularity_subprocess_env",
        lambda **_kwargs: {"PATH": "/usr/bin"},
    )

    def _fake_run(cmd, **_kwargs):
        calls.append(list(cmd))
        build_path = singularity_env.Path(cmd[2])
        build_path.parent.mkdir(parents=True, exist_ok=True)
        build_path.write_bytes(b"fake-sif")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(singularity_env.subprocess, "run", _fake_run)

    image = "docker://example/image@sha256:" + ("a" * 64)
    first = singularity_env._get_or_build_sif(
        image,
        "/usr/bin/apptainer",
        owner_home=owner,
        policy_generation="profile-scope-v1:generation",
    )
    second = singularity_env._get_or_build_sif(
        image,
        "/usr/bin/apptainer",
        owner_home=owner,
        policy_generation="profile-scope-v1:generation",
    )

    assert first == second
    assert len(calls) == 1
