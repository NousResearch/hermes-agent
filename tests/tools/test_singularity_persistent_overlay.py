"""Persistent-overlay path safety for the Singularity environment.

A raw task id used as an overlay directory name carries the same class of
bug as the Docker persistent-sandbox mount path (#92414): colon-bearing
session ids break bind specs and are unrepresentable on Windows. Overlay
components must go through the shared sanitizer in tools.environments.base.
"""

from tools.environments import singularity as singularity_env
from tools.environments.path_utils import sanitize_task_id_for_path


def _stub_singularity(monkeypatch, tmp_path):
    monkeypatch.setattr(
        singularity_env, "_ensure_singularity_available", lambda: "/usr/bin/apptainer"
    )
    monkeypatch.setattr(
        singularity_env,
        "_get_or_build_sif",
        lambda image, executable="apptainer": str(tmp_path / "image.sif"),
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
