"""Private artifact publication used by proactive tool-result pruning."""

from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess

import pytest

from tools.tool_result_storage import (
    persist_tool_artifact,
    redact_sensitive_text,
    remove_created_tool_artifact,
)


class LocalArtifactEnv:
    def __init__(self, temp_dir: Path):
        self.temp_dir = temp_dir

    def get_temp_dir(self) -> str:
        return str(self.temp_dir)

    def execute(self, command, *, timeout=30, stdin_data=None):
        process = subprocess.run(
            ["bash", "-c", command],
            input=stdin_data,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        return {
            "output": process.stdout + process.stderr,
            "returncode": process.returncode,
        }


def _persist(content: str, env, *, scope="task-a", call_id="call-1"):
    return persist_tool_artifact(
        content,
        kind="output",
        tool_name="terminal",
        tool_use_id=call_id,
        task_scope=scope,
        env=env,
        extension=".output.txt",
    )


def test_persists_redacted_verified_private_bytes(tmp_path):
    env = LocalArtifactEnv(tmp_path)
    secret = "sk-" + "a" * 24
    original = f"API_KEY={secret}\nresult"

    artifact = _persist(original, env)

    assert artifact is not None
    path = Path(artifact.path)
    persisted = path.read_bytes()
    expected = redact_sensitive_text(original, force=True).encode("utf-8")
    assert persisted == expected
    assert secret.encode("utf-8") not in persisted
    assert artifact.sha256 == hashlib.sha256(expected).hexdigest()
    assert artifact.chars == len(expected.decode("utf-8"))
    assert artifact.redacted is True
    assert artifact.created is True
    assert path.stat().st_mode & 0o777 == 0o600
    assert path.parent.stat().st_mode & 0o777 == 0o700
    assert path.parent.name == hashlib.sha256(b"task-a").hexdigest()
    assert artifact.sha256 in path.name


def test_attempt_paths_make_rollback_concurrency_safe(tmp_path):
    env = LocalArtifactEnv(tmp_path)

    rejected = _persist("same bytes", env)
    committed = _persist("same bytes", env)

    assert rejected is not None and committed is not None
    assert rejected.path != committed.path
    assert Path(rejected.path).read_bytes() == b"same bytes"
    assert Path(committed.path).read_bytes() == b"same bytes"
    assert remove_created_tool_artifact(rejected, env) is True
    assert not Path(rejected.path).exists()
    assert Path(committed.path).read_bytes() == b"same bytes"


def test_same_call_id_across_scope_or_content_cannot_clobber(tmp_path):
    env = LocalArtifactEnv(tmp_path)

    first = _persist("first", env, scope="task-a", call_id="same")
    other_content = _persist("second", env, scope="task-a", call_id="same")
    other_scope = _persist("first", env, scope="task-b", call_id="same")

    assert first is not None and other_content is not None and other_scope is not None
    assert len({first.path, other_content.path, other_scope.path}) == 3
    assert Path(first.path).read_text() == "first"
    assert Path(other_content.path).read_text() == "second"
    assert Path(other_scope.path).read_text() == "first"


@pytest.mark.parametrize(
    ("scope", "env"),
    [("", object()), ("   ", object()), ("task-a", None)],
)
def test_missing_scope_or_environment_fails_closed(scope, env):
    assert _persist("content", env, scope=scope) is None


def test_unsafe_extension_is_rejected(tmp_path):
    env = LocalArtifactEnv(tmp_path)
    with pytest.raises(ValueError, match="Unsafe artifact extension"):
        persist_tool_artifact(
            "content",
            kind="output",
            tool_name="terminal",
            tool_use_id="call",
            task_scope="task",
            env=env,
            extension="/../../escape",
        )


def test_cleanup_refuses_modified_bytes(tmp_path):
    env = LocalArtifactEnv(tmp_path)
    artifact = _persist("expected", env)
    assert artifact is not None
    path = Path(artifact.path)
    path.write_text("changed")

    assert remove_created_tool_artifact(artifact, env) is False
    assert path.read_text() == "changed"
