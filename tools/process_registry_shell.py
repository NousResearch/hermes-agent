"""Post-startup cwd handling for local background shells."""

import os


def background_shell_command(cwd: str | None, safe_command: str) -> str:
    """Re-pin cwd after rc files, bypassing shell functions on POSIX and Git Bash."""
    from tools.environments.local import _quote_bash_path

    # Popen/PTY chdir before the login shell sources rc files. Convert native
    # Windows paths before quoting, and refuse to run in an unintended directory.
    target = _quote_bash_path(cwd or os.getcwd())
    return f"command cd -- {target} || exit 126; set +m; {safe_command}"
