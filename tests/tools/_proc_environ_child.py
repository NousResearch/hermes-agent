"""Helper: child process for /proc/environ isolation test.

Spawned inside a PID namespace by _proc_environ_parent.py with a
clean environment. Attempts to read the parent's /proc/<ppid>/environ
and print the value of HERMES_TEST_SECRET.
"""
import os

ppid = os.getppid()
try:
    with open(f"/proc/{ppid}/environ", "rb") as f:
        raw = f.read()
    env = dict(
        e.decode("utf-8", "replace").split("=", 1)
        for e in raw.split(b"\x00")
        if b"=" in e
    )
    print(env.get("HERMES_TEST_SECRET", "NOT_FOUND"))
except Exception as e:
    print(f"ERROR:{e}")
