"""Verified source reads for group sharing, over the existing terminal exec channel."""
import base64
import binascii
import json
import posixpath
import shlex
import uuid
from pathlib import Path


def _source_policy(home: str | None) -> dict:
    from agent.file_safety import _BLOCKED_PROJECT_ENV_BASENAMES
    from gateway.media_fetch import _DENIED_HOME_RELATIVE, _DENIED_PREFIXES
    from gateway.platforms.base import _ROOT_CREDENTIAL_PATHS
    from tools.group_file_tool import _PRIVATE_STORAGE, _PRIVATE_TREES

    home = posixpath.normpath(home) if home else None
    prefixes = [str(p) for p in _DENIED_PREFIXES if str(p) != home]
    if home:
        prefixes.extend(posixpath.join(home, str(p)) for p in _DENIED_HOME_RELATIVE)
    return dict(prefixes=prefixes, anywhere=[] if home else [p.parts for p in _DENIED_HOME_RELATIVE],
                  private_names=sorted(_BLOCKED_PROJECT_ENV_BASENAMES), private_storage=_PRIVATE_STORAGE,
                  private_trees=[*_PRIVATE_TREES, ("profiles",),
                                 *(tuple(p.replace("\\", "/").split("/")) for p in _ROOT_CREDENTIAL_PATHS)])


def read_remote_file(path: str, limit: int) -> bytes:
    from gateway.media_fetch import _active_remote_env
    from gateway.media_policy import media_delivery_strict
    if media_delivery_strict() or (env := _active_remote_env()) is None:
        raise ValueError("That file cannot be shared. Share a file from your workspace or a Hermes media folder.")
    home = getattr(env, "_remote_home", None)
    if path.startswith("~/") and home:
        path = posixpath.join(home, path[2:])
    policy = _source_policy(home)
    marker = f"__HERMES_SHARE_{uuid.uuid4().hex}__"
    request = dict(path=path, limit=limit, policy=policy, marker=marker)
    # Ship a standalone stdlib script: do not depend on a Hermes install in the sandbox.
    script = Path(__file__).with_name("group_file_remote_read.py").read_text(encoding="utf-8-sig")
    command = "python3 -c " + shlex.quote(script) + " " + shlex.quote(json.dumps(request))
    refusal = "This backend cannot provide verified remote file reads (Python 3 and descriptor paths are required)."
    try:
        response = env.execute(command, timeout=300, rewrite_compound_background=False)
        output = response.get("output") or ""
        if response.get("returncode") != 0 or output.count(marker) != 2:
            raise ValueError(refusal)
        result = json.loads(output.split(marker)[1])
        if "error" in result:
            raise ValueError(result["error"])
        data = base64.b64decode(result["data"], validate=True)
    except (AttributeError, KeyError, TypeError, binascii.Error, json.JSONDecodeError) as exc:
        raise ValueError(refusal) from exc
    if not 0 < len(data) <= limit:
        raise ValueError("The verified remote file transfer was incomplete or oversized. Try again.")
    return data
