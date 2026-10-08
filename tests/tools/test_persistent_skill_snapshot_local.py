"""Real-filesystem contracts for the local Docker skill snapshot port."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from tools.environments import skill_snapshot as ss
from tools.environments import docker


@pytest.fixture
def roots(tmp_path, monkeypatch):
    data = tmp_path / "docker-data"
    monkeypatch.setenv("HERMES_DOCKER_DATA_ROOT", str(data))
    source = tmp_path / "skills"
    script = source / "one" / "scripts" / "main.py"
    script.parent.mkdir(parents=True)
    script.write_text("print('v1')\n")
    return source, data


def combined(destination, digest):
    return hashlib.sha256(destination.encode() + b"\0" + digest.encode() + b"\0").hexdigest()


def test_persistent_content_addressed_and_secret_filtered(roots):
    source, _ = roots
    (source / ".env").write_text("secret")
    (source / "innocent").symlink_to(source / ".env")
    (source / "outside").symlink_to(source.parent / "outside-secret")
    (source / "cycle").symlink_to(source)
    dest, digest = ss.stage_skills(source, "default", "/root/.hermes/skills")
    assert not (dest / ".env").exists()
    assert not (dest / "innocent").exists()
    assert not (dest / "outside").exists()
    assert not (dest / "cycle").exists()
    assert ss.stage_skills(source, "default", "/root/.hermes/skills") == (dest, digest)
    assert ss.snapshot_usable(dest, digest) is None
    (source / "one/scripts/main.py").write_text("print('v2')\n")
    new, new_digest = ss.stage_skills(source, "default", "/root/.hermes/skills")
    assert new != dest and new_digest != digest and dest.exists()


def test_corrupt_existing_tree_is_never_replaced(roots):
    source, _ = roots
    dest, digest = ss.stage_skills(source, "default", "/root/.hermes/skills")
    script = dest / "one/scripts/main.py"
    script.write_text("corrupt")
    with pytest.raises(RuntimeError, match="refusing replacement"):
        ss.stage_skills(source, "default", "/root/.hermes/skills")
    assert script.read_text() == "corrupt"
    assert ss.snapshot_usable(dest, digest) == "required_mismatch"


def test_cleanup_preserves_mounted_and_unknown_trees(roots, monkeypatch):
    source, _ = roots
    dest, _ = ss.stage_skills(source, "default", "/root/.hermes/skills")
    unknown = dest.parent / "legacy"
    unknown.mkdir()
    assert ss.cleanup_unreferenced("default", mounted={str(dest.parent)}) == []
    monkeypatch.setattr(ss, "mounted_sources", lambda: None)
    assert not ss.safe_to_delete(dest)
    assert ss.cleanup_unreferenced("default") == []
    assert ss.cleanup_unreferenced("default", mounted=set()) == [dest.name]
    assert unknown.exists()


@pytest.mark.parametrize("bad", ["writable", "missing", "changed", "other_tree", "bad_json"])
def test_reuse_rejects_bad_mount(roots, monkeypatch, bad):
    source, _ = roots
    destination = "/root/.hermes/skills"
    dest, digest = ss.stage_skills(source, "default", destination)
    mount = {"Source": str(dest), "Destination": destination, "RW": False}
    expected = combined(destination, digest)
    if bad == "writable":
        mount["RW"] = True
    elif bad == "missing":
        mount["Source"] = str(dest.parent / "tmp-old")
    elif bad == "changed":
        (dest / "one/scripts/main.py").write_text("broken")
    elif bad == "other_tree":
        expected = "0" * 64
    output = "invalid" if bad == "bad_json" else json.dumps([mount])
    monkeypatch.setattr(docker, "_docker_query", lambda *a, **kw: SimpleNamespace(stdout=output))
    env = docker.DockerEnvironment.__new__(docker.DockerEnvironment)
    env._docker_exe = "docker"
    assert not env._skills_mount_ok("cid", expected)


def test_reuse_accepts_exact_readonly_snapshot_and_filters_identity(roots, monkeypatch):
    source, _ = roots
    destination = "/root/.hermes/skills"
    dest, digest = ss.stage_skills(source, "default", destination)
    expected = combined(destination, digest)
    env = docker.DockerEnvironment.__new__(docker.DockerEnvironment)
    env._docker_exe = "docker"
    env._labels = {docker._SKILLS_LABEL_KEY: expected, docker._ENVIRONMENT_LABEL_KEY: "official-env"}
    calls = []
    def query(args, **kwargs):
        calls.append(args)
        if "ps" in args:
            return SimpleNamespace(stdout="cid\trunning\n")
        return SimpleNamespace(stdout=json.dumps([{"Source": str(dest), "Destination": destination, "RW": False}]))
    monkeypatch.setattr(docker, "_docker_query", query)
    assert env._find_reusable_container("task", "default", "off") == ("cid", "running")
    assert f"label={docker._SKILLS_LABEL_KEY}={expected}" in calls[0]
    assert "label=hermes-environment=official-env" in calls[0]


def test_mount_inspection_malformed_is_unknown(monkeypatch):
    outputs = iter([SimpleNamespace(returncode=0, stdout="cid"), SimpleNamespace(returncode=0, stdout='[42]')])
    monkeypatch.setattr(ss.subprocess, "run", lambda *a, **kw: next(outputs))
    assert ss.mounted_sources() is None
