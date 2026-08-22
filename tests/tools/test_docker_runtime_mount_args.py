from __future__ import annotations

import pytest

from tools.environments.docker import _normalize_runtime_mounts, _runtime_mount_args


def test_runtime_mount_args_use_strict_bind_mount_syntax():
    mounts = _normalize_runtime_mounts([
        {
            "source": "/host/task",
            "target": "/workspace",
            "read_only": False,
            "purpose": "workspace",
        },
        {
            "source": "/host/repo/.git",
            "target": "/host/repo/.git",
            "read_only": False,
            "purpose": "git-common-dir",
        },
    ])
    args, targets = _runtime_mount_args(mounts)
    assert targets == {"/workspace", "/host/repo/.git"}
    assert args == [
        "--mount", "type=bind,src=/host/task,dst=/workspace",
        "--mount", "type=bind,src=/host/repo/.git,dst=/host/repo/.git",
    ]


def test_runtime_mount_args_reject_duplicate_targets():
    with pytest.raises(ValueError, match="duplicate"):
        _normalize_runtime_mounts([
            {"source": "/a", "target": "/workspace"},
            {"source": "/b", "target": "/workspace"},
        ])
