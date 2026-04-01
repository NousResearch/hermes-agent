"""Helper: parent process for /proc/environ isolation test.

Spawned by TestProcEnvironIsolation with HERMES_TEST_SECRET in its
exec-time environment. Spawns a child via _pidns_wrap with a clean
env, then prints whatever the child outputs.
"""
import os
import subprocess
import sys

sys.path.insert(0, os.environ["HERMES_AGENT_ROOT"])
from tools.environments.local import _pidns_wrap

child_script = sys.argv[1]
child_env = {"PATH": os.environ.get("PATH", "/usr/bin"), "HOME": "/tmp"}

result = subprocess.run(
    _pidns_wrap([sys.executable, child_script]),
    env=child_env,
    capture_output=True,
    text=True,
)
print(result.stdout.strip())
