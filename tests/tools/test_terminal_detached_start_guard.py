"""The foreground long-lived-process guard must not block detached starts.

``docker compose up -d`` / ``--detach`` / ``--wait`` and a plain ``systemd-run`` (a transient
unit, detached by default) return as soon as the start is done; none of them keeps the foreground
busy. Blocking them only cost the agent a turn. Attached starts, including an attached start
chained after a detached one, must still be redirected to ``background=true``.
"""

import pytest

from tools.terminal_tool_guards import _foreground_background_guidance as guidance


@pytest.mark.parametrize("cmd", [
    "docker compose up -d",
    "docker compose up -d 2>&1 | tail -5; sleep 6; docker compose ps",
    "cd /srv/app && docker compose up --detach 2>&1 | tail -5",
    "docker compose up --quiet-pull -d db",
    "docker compose up -d --quiet-pull db 2>&1 | tail -3",
    "docker compose pull -q && docker compose up -d --wait --wait-timeout 180 2>&1 | tail -5",
    "timeout 200 docker compose up --wait 2>&1 | tail -5",
    "docker compose up -dV",
    "systemd-run --user --unit=demo-web --property=Restart=on-failure /usr/bin/python3 -m http.server 8766",
])
def test_detached_starts_are_not_blocked(cmd):
    assert guidance(cmd) is None


@pytest.mark.parametrize("cmd", [
    "docker compose up",
    "docker compose up db",
    "docker compose up --build web",
    "docker compose up --no-deps web",
    "docker compose up -d db; docker compose up web",
    "docker compose up -d && npm run dev",
    "python3 -m http.server 8000",
    "systemd-run --user --scope python3 -m http.server 8000",
    "systemd-run --user --wait python3 -m http.server 8000",
    "systemd-run --user -t python3 -m http.server 8000",
    "uvicorn app:app",
])
def test_attached_starts_are_still_redirected(cmd):
    msg = guidance(cmd)
    assert msg is not None and "background=true" in msg
