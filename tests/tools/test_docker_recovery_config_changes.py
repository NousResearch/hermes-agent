"""Recovery honors changed or disappearing env-file inputs (#85342)."""

from tests.tools.test_docker_environment import _make_dummy_env
from tests.tools.test_docker_environment_reuse_candidates import _stateful_docker
from tests.tools.test_docker_reuse_lifecycle import isolated_environments


def test_recovery_retires_the_replaced_configuration(monkeypatch, tmp_path):
    containers, _ = _stateful_docker(monkeypatch)
    path = tmp_path / "recovery.env"
    path.write_text("RECOVERY_VALUE=before\n", encoding="utf-8")
    env = _make_dummy_env(persistent_filesystem=True, extra_args=["--env-file", str(path)])
    original_id = env._container_id
    containers[original_id]["state"] = "exited"
    path.write_text("RECOVERY_VALUE=after\n", encoding="utf-8")

    assert env._recreate_container()
    replacement_id = env._container_id
    assert replacement_id != original_id
    assert original_id in containers
    assert env._reuse_owners.reap(env._docker_exe, 0, None) == 1
    assert original_id not in containers
    assert containers[replacement_id]["state"] == "running"


def test_disappearing_env_file_does_not_escape_recovery(monkeypatch, tmp_path):
    _stateful_docker(monkeypatch)
    path = tmp_path / "recovery.env"
    path.write_text("RECOVERY_VALUE=before\n", encoding="utf-8")
    env = _make_dummy_env(persistent_filesystem=True, extra_args=["--env-file", str(path)])
    original_id, lease = env._container_id, env._reuse_lease

    def candidate_inspection(*_args):
        path.unlink()
        return False

    monkeypatch.setattr(env, "_attach_existing_container", candidate_inspection)
    assert env._recreate_container() is False
    assert env._container_id == original_id
    assert env._reuse_lease is lease
