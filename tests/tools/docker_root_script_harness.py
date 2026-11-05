"""Run the image's root-context shell scripts on the host against fake privileged tools.

The scripts pin ``PATH`` to the system directories and call the tools that decide or perform
the privilege drop by absolute path. To run one without a container, a copy of its text is
pointed at a fake-tool directory: the pinned ``PATH`` gets that directory in front, and each
absolute drop tool is redirected into it. Everything else the script resolves still comes from
the system directories, never from the ``PATH`` the script inherited.
"""
from __future__ import annotations

import re
from pathlib import Path

SYSTEM_PATH = "/command:/usr/sbin:/usr/bin:/sbin:/bin"
ABSOLUTE_TOOLS = ("/usr/bin/id", "/command/s6-setuidgid", "/command/s6-svscan", "/bin/sleep")
HERMES_LOCAL_BIN = "/opt/data/.local/bin"


def redirect_root_tools(text: str, bindir: Path) -> str:
    """``text`` with the pinned root PATH and the absolute drop tools pointed at ``bindir``."""
    text = text.replace(f"PATH={SYSTEM_PATH}\n", f"PATH={bindir}:{SYSTEM_PATH}\n")
    for tool in ABSOLUTE_TOOLS:
        name = tool.rsplit("/", 1)[1]
        text = re.sub(rf"(?<![\w/.-]){re.escape(tool)}(?![\w.-])", f"{bindir}/{name}", text)
    return text


def write_recorders(bindir: Path, log: Path, names: dict[str, str]) -> Path:
    """Executables that append ``<name> <args>`` to ``log`` and print a canned stdout."""
    bindir.mkdir(parents=True, exist_ok=True)
    for name, stdout in names.items():
        script = bindir / name
        script.write_text(
            "#!/bin/sh\n"
            f'printf "%s\\n" "{name} $*" >> "{log}"\n'
            + (f"printf '%s\\n' '{stdout}'\n" if stdout else "")
            + "exit 0\n",
            encoding="utf-8",
        )
        script.chmod(0o755)
    return bindir


def write_fake_id(bindir: Path) -> None:
    """``id``: ``-u`` alone prints ``$FAKE_UID`` (default 0, root); any query about the hermes
    user prints its build UID/GID."""
    bindir.mkdir(parents=True, exist_ok=True)
    fake = bindir / "id"
    fake.write_text(
        '#!/bin/sh\nif [ "$#" -eq 1 ] && [ "$1" = -u ]; then echo "${FAKE_UID:-0}"; else echo 10000; fi\n',
        encoding="utf-8",
    )
    fake.chmod(0o755)


def calls(log: Path, name: str) -> list[list[str]]:
    if not log.exists():
        return []
    return [line.split()[1:] for line in log.read_text().splitlines() if line.split()[:1] == [name]]


def dropped_command(args: list[str]) -> tuple[str | None, list[str]]:
    """Split ``s6-setuidgid`` argv after the user into (PATH handed to the child, child argv).

    ``/usr/bin/env PATH=<p> cmd ...`` hands ``<p>``; a bare ``cmd ...`` hands nothing."""
    rest = args[1:]
    child_path = None
    if rest[:1] == ["/usr/bin/env"]:
        rest = rest[1:]
        while rest and "=" in rest[0] and not rest[0].startswith("/"):
            key, _, value = rest[0].partition("=")
            if key == "PATH":
                child_path = value
            rest = rest[1:]
    return child_path, rest
