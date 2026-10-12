"""Publish complete new files without ever replacing an existing directory entry."""

import json


def atomic_create(ops, path, content):
    """Write a same-directory temporary file, then exclusively link it into place.

    os.link fails for every existing entry, including dangling symlinks and
    directories. Other processes see the complete file once publication succeeds.
    Backends lacking hard links fail safely; no overwrite fallback is allowed.
    """
    snippet = (
        "import os, sys, tempfile\n"
        f"path = {json.dumps(path)}\n"
        "parent = os.path.dirname(path) or '.'\n"
        "os.makedirs(parent, exist_ok=True)\n"
        "fd, tmp = tempfile.mkstemp(prefix='.hermes-create-', dir=parent)\n"
        "published = False\n"
        "try:\n"
        "    with os.fdopen(fd, 'wb') as stream:\n"
        "        stream.write(sys.stdin.buffer.read())\n"
        "    mask = os.umask(0); os.umask(mask)\n"
        "    os.chmod(tmp, 0o666 & ~mask)\n"
        "    try:\n"
        "        os.link(tmp, path)\n"
        "        published = True\n"
        "    except FileExistsError:\n"
        "        print('already_exists'); sys.exit(17)\n"
        "finally:\n"
        "    try:\n"
        "        os.unlink(tmp)\n"
        "    except OSError as exc:\n"
        "        if not published: raise\n"
        "        print('File created; temporary-file cleanup failed: ' + str(exc))\n"
    )
    return ops._exec_python_snippet(snippet, stdin_data=content)
