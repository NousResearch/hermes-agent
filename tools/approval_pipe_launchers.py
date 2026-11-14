"""Launcher option table for the pipe-to-shell sink in :mod:`tools.approval_detection` (moved from there)."""


# Launchers that run their trailing words as a command, with the short/long options that take a
# SEPARATE value word. One shared table made a bare flag (`time -p`, `exec -a x`) swallow the shell
# name as its value; per-launcher rows let `-p` be bare for time/command and valued for sudo.
_PIPE_LAUNCHER_VALUE_FLAGS = {
    "sudo": ("ugUCDhpRrtT", ("user", "group", "other-user", "close-from", "chdir", "host", "prompt",
                             "role", "type", "command-timeout", "chroot")),
    "doas": ("uC", ()),
    # `-S STRING` is the command itself (split into words), never a swallowed value.
    "env": ("uCa", ("unset", "chdir", "argv0")),
    "exec": ("a", ()),
    "command": ("", ()),
    "nohup": ("", ()),
    "nice": ("n", ("adjustment",)),
    "time": ("fo", ("format", "output")),
    "timeout": ("sk", ("signal", "kill-after")),
    "stdbuf": ("ioe", ("input", "output", "error")),
    "busybox": ("", ()),
    "setsid": ("", ()),
    "ionice": ("cnpPu", ("class", "classdata", "pid", "pgid", "uid")),
}


def _pipe_launcher_re(name: str, short: str, long_opts: tuple) -> str:
    valued = [rf"(?-i:-[^\W{short}\d-]*+[{short}])\s++[^\s|;&<>]++"] if short else []
    valued += [rf"--(?:{'|'.join(long_opts)})\s++[^\s|;&<>]++"] if long_opts else []
    duration = r"(?:infinity|\d++(?:\.\d++)?(?:e\d++)?|\.\d++)[smhd]?(?=\s)"
    args = "|".join([*valued, r"-[^\s|;&<>]++", r"\w++=[^\s|;&<>]*+", duration])
    return rf"{name}(?:\s++(?>{args}))*+"
