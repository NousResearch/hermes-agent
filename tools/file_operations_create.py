"""Publish complete new files without ever replacing an existing directory entry."""

import base64
import json

# POSIX shell twin of the Python helper for backends without python3: link(1)
# makes the same link(2) call as os.link, so every existing entry (dangling
# symlinks and directories included) is refused and nothing is overwritten.
_SHELL_CREATE = (
    'd="$(dirname -- "$t")"; mkdir -p -- "$d" || exit 1; '
    'command -v link >/dev/null 2>&1 || { cat >/dev/null; '
    'echo "create_only needs python3 or link(1) on the terminal backend"; exit 1; }; '
    'tmp="$(mktemp "$d/.hermes-create-XXXXXX")" || exit 1; '
    "trap 'rm -f \"$tmp\"' EXIT; "
    'cat > "$tmp" || exit 1; chmod "=rw" "$tmp" 2>/dev/null; '
    'link "$tmp" "$t" 2>/dev/null && exit 0; '
    'if [ -e "$t" ] || [ -L "$t" ]; then echo already_exists; exit 17; fi; '
    'echo "could not link the new file into place"; exit 1'
)


def atomic_create(ops, path, content):
    """Write a same-directory temporary file, then exclusively link it into place.

    os.link fails for every existing entry, including dangling symlinks and
    directories. Other processes see the complete file once publication succeeds.
    Backends lacking hard links fail safely; no overwrite fallback is allowed.
    Remote backends without python3 run the equivalent ``link(1)`` shell script.
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
    if ops._lsp_local_only():
        # The local backend runs the agent's own interpreter (always present).
        return ops._exec_python_snippet(snippet, stdin_data=content)
    py = ops._python_interpreter_cmd()
    encoded = base64.b64encode(snippet.encode("utf-8")).decode("ascii")
    return ops._exec(
        f"if command -v {py} >/dev/null 2>&1; then "
        f"{py} -c \"import base64; exec(base64.b64decode('{encoded}').decode())\"; "
        f"else (t={ops._escape_shell_arg(path)}; {_SHELL_CREATE}); fi", stdin_data=content)
