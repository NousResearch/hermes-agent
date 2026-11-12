"""Exec the current Hermes runtime with our wrapper module instead of hermes_cli.main.

`hermes --print-runtime-command` prints the exact argv the launcher would use (right
python, right pm environment generation, right bootstrap). We reuse that argv verbatim
and only swap the module it runs, so nothing here hardcodes a venv path.
"""

import json
import os
import subprocess
import sys

MODULE = "hermes_cost_cli"
TARGET = "runpy.run_module('hermes_cli.main', run_name='__main__', alter_sys=True)"


def main() -> int:
    tool_dir = os.path.dirname(os.path.abspath(__file__))
    out = subprocess.run(
        ["hermes", "--print-runtime-command"], capture_output=True, text=True, check=True
    ).stdout.strip()
    argv = json.loads(out)
    if len(argv) < 4 or TARGET not in argv[3]:
        sys.stderr.write("hermes-cost: unexpected runtime command from hermes; aborting\n")
        return 1
    argv[3] = argv[3].replace(
        TARGET,
        f"sys.path.insert(0, {tool_dir!r}); runpy.run_module({MODULE!r}, run_name='__main__', alter_sys=True)",
    )
    os.execv(argv[0], argv + sys.argv[1:])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
