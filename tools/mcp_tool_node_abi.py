"""Stdio MCP servers whose native addon was built for a different Node.js (#124264).

Hermes runs every stdio MCP server on its own managed Node, never the user's. A server whose
``.node`` addon was compiled under another Node (typically through the ``~/.npm/_npx`` cache the
user's ``npx`` shares with ours) dies at startup with ``NODE_MODULE_VERSION`` /
``ERR_DLOPEN_FAILED`` and only its stderr says so. This module turns that stderr into an error that
names the remedy: rebuild the server under Hermes's Node.
"""

import os
import re
import shlex
from pathlib import Path
from typing import Optional

_ABI_MARKERS = ("NODE_MODULE_VERSION", "ERR_DLOPEN_FAILED")
_MODULE_PATH = re.compile(r"'([^'\n]+?\.node)'|(\S+?\.node):")
_ABI_VERSIONS = re.compile(r"using\s+NODE_MODULE_VERSION (\d+)\..*?requires\s+NODE_MODULE_VERSION (\d+)", re.S)


class NodeAbiMismatchError(RuntimeError):
    """A stdio server's native addon does not load under Hermes's Node. Permanent: every retry
    loads the same binary, so the server parks until the user rebuilds it."""


def _managed(name: str) -> tuple[Optional[Path], str]:
    """``(binary, version)`` of Hermes's own node/npm, or ``(None, "")`` when PM has none."""
    try:
        from pm import installed_package
        installed = installed_package(name)
    except Exception:
        return None, ""
    return (installed.binary, installed.version) if installed and installed.binary else (None, "")


def _package_of(module_path: str) -> tuple[Optional[Path], str]:
    """``(project root, package name)`` owning *module_path*: the directory above its innermost
    ``node_modules`` and the (possibly scoped) package under it."""
    parts = Path(module_path).parts
    marks = [i for i, part in enumerate(parts) if part == "node_modules"]
    if not marks or marks[-1] + 1 >= len(parts):
        return None, ""
    at = marks[-1]
    package = parts[at + 1]
    if package.startswith("@") and at + 2 < len(parts):
        package = f"{package}/{parts[at + 2]}"
    return Path(*parts[:at]), package


def _remedy(root: Optional[Path], package: str, node: Path, npm: Path) -> str:
    """The fix in the host shell's syntax: delete an npx cache entry (Hermes's npx reinstalls it on the
    next start) or rebuild with Hermes's npm, Hermes's Node first on PATH because npm's shebang and its
    lifecycle scripts both run the first ``node`` there."""
    windows = os.name == "nt"
    q = (lambda p: "'" + str(p).replace("'", "''") + "'") if windows else (lambda p: shlex.quote(str(p)))
    target = f" {q(package)} --prefix {q(root)}" if root is not None else ""
    if windows:
        rebuild = f"$env:Path = {q(str(node.parent) + ';')} + $env:Path; & {q(npm)} rebuild{target}"
    else:
        rebuild = f"PATH={q(node.parent)}:\"$PATH\" {q(npm)} rebuild{target}"
    if root is None:
        return f"in the server's package directory run {rebuild}"
    if root.parent.name != "_npx":
        return rebuild
    delete = f"Remove-Item -Recurse -Force {q(root)}" if windows else f"rm -rf {q(root)}"
    return f"{delete} (Hermes reinstalls it on the next start), or {rebuild}"


def node_abi_error(server_name: str, stderr_text: str) -> Optional[NodeAbiMismatchError]:
    """The remedy-naming error for a server whose stderr shows a native-addon load failure, else None."""
    if not stderr_text or not any(marker in stderr_text for marker in _ABI_MARKERS):
        return None
    found = _MODULE_PATH.search(stderr_text)
    module = (found.group(1) or found.group(2)) if found else ""
    versions = _ABI_VERSIONS.search(stderr_text)
    node, node_version = _managed("node")
    npm, _ = _managed("npm")
    ours = f"Hermes's Node {node_version}".strip() if node_version else "Hermes's Node"
    built = (f"NODE_MODULE_VERSION {versions.group(1)}; {ours} needs {versions.group(2)}" if versions
             else f"it does not load under {ours}")
    # Diagnosis first: the startup banner keeps only the first ~120 characters.
    head = (f"native addon built for a different Node.js ({built}): {module or 'see logs/mcp-stderr.log'}. "
            "Hermes runs MCP servers on its own Node; rebuild the server under it")
    if node is None or npm is None:
        return NodeAbiMismatchError(f"{head}: install Hermes's Node with `hermes pm install npm`, then run "
                                    f"`hermes mcp test {server_name}`")
    root, package = _package_of(module) if module else (None, "")
    return NodeAbiMismatchError(f"{head}: {_remedy(root, package, node, npm)}")
