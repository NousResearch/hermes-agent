"""The long-lived-process guard must match a server BINARY in command position,
never the same word used as an argument, a path component or a grep target.

Regression for #42620, #108593, #137323: the guard matched server names anywhere in
the command, so a read-only command or a package install that merely mentioned one
was refused as if it started a watcher: ``ls -d node_modules/vite``,
``pip show uvicorn``, ``npm install nodemon``. Both directions matter: over-matching
blocks ordinary work, under-matching lets a real dev server hang the call.
"""

import pytest

from tools.terminal_tool_guards import _foreground_background_guidance as guidance


FALSE_POSITIVES = [
    # bare name as an argument / path component / list target
    "ls -d node_modules/react node_modules/vite",
    "du -sh node_modules/vite",
    "ls apps/desktop/vite.config.ts",
    "cat apps/desktop/vite.config.ts",
    'find . -name "vite.config.ts"',
    "pip show uvicorn",
    "du -sh venv/lib/python3.11/site-packages/gunicorn",
    "ls node_modules/.bin/nodemon",
    "grep -rn nodemon package.json",
    "grep -rn vite apps/desktop/vite.config.ts",
    "grep -n 'npm run dev' docs/deploy.md",
    "echo vite nodemon uvicorn gunicorn",
    "python3 -c 'import vite'",
    # Windows read forms from the package-path report
    r"type C:\app\.venv\Lib\site-packages\gunicorn\arbiter.py",
    r"dir C:\app\.venv\Lib\site-packages\uvicorn",
    r"cat C:\app\node_modules\nodemon\package.json",
    # installing or querying a package of that name launches nothing
    "npm install nodemon",
    "npm update vite lodash axios",
    "pnpm add vite",
    "pip install uvicorn",
    "uv add gunicorn",
    "which uvicorn",
    # bounded invocations of a server binary
    "vite build",
    "npx vite build",
    "gunicorn --check-config app:app",
    "docker compose up -d",
    "docker compose up --detach web",
    # quoted mentions and informational flags
    "grep -rn 'vite' .",
    "echo 'uvicorn gunicorn nodemon'",
    "vite --version",
    "uvicorn --help",
]

TRUE_POSITIVES = [
    # bare binary at the start of the command
    "vite",
    "vite --host 0.0.0.0",
    "vite --port 3000",
    # behind a runner
    "npx vite",
    "npx vite --port 3000",
    "pnpm exec vite",
    "pnpm dlx vite",
    "yarn vite",
    "bunx vite",
    # behind a path prefix
    "./node_modules/.bin/vite",
    "apps/desktop/node_modules/.bin/vite",
    # behind sudo, including options that take a value
    "sudo -u www vite",
    "sudo -g www-data -u node vite",
    "sudo -E npx vite",
    # launched as a python module
    "python -m uvicorn app:app",
    "python3 -m uvicorn app:app",
    "python3 -m gunicorn app:app",
    # not the first command in the line
    "cd apps/desktop && npx vite",
    "cd apps/desktop && vite",
    "echo hi; vite",
    "cd app\nvite --port 3000",
    # the sibling patterns keep working
    "npm run dev",
    "pnpm dev",
    "yarn watch",
    "cd x && npm run start",
    "docker compose up",
    "docker compose -f dev.yml up web",
    "next dev",
    "npx next dev",
    "nodemon server.js",
    "uvicorn app:main --reload",
    "/app/.venv/bin/uvicorn app:app",
    "gunicorn -w 4 app:app",
    "python3 -m http.server 8000",
    "python -m http.server",
    # Windows launcher suffix is the same command
    "uvicorn.exe app:app",
    "gunicorn.exe -w 4 app:app",
    "npm.cmd run dev",
    "npx.cmd vite",
    "pnpm.cmd exec vite",
    "python.exe -m uvicorn app:app",
    r"C:\app\.venv\Scripts\uvicorn.exe app:app",
    r"C:\app\node_modules\.bin\vite.cmd --port 3000",
    r".\node_modules\.bin\nodemon.cmd server.js",
    # an install earlier in the line does not excuse the launch after it
    "npm install vite && vite",
    "npm install vite; vite",
    "pip install uvicorn && uvicorn app:app",
    # behind env assignments, wrappers, runners, keywords and subshells
    "PORT=3000 npm run dev",
    "env PORT=3000 vite",
    "timeout 60 uvicorn app:app",
    "poetry run uvicorn app:app",
    "uv run gunicorn app:app",
    "npx --yes vite",
    "npm --prefix web run dev",
    "(cd web && npm run dev)",
    "if true; then npm run dev; fi",
    "doppler run -- npm run dev",
    "ssh -p 2222 host uvicorn app:app",
    "docker exec -it app uvicorn app:app",
]


@pytest.mark.parametrize("command", FALSE_POSITIVES)
def test_command_that_only_names_a_server_runs(command):
    msg = guidance(command)
    assert msg is None, f"{command!r} was flagged as long-lived: {msg}"


@pytest.mark.parametrize("command", TRUE_POSITIVES)
def test_command_that_starts_a_server_is_flagged(command):
    msg = guidance(command)
    assert msg is not None, f"{command!r} should have been flagged"
    assert "long-lived server/watch process" in msg
