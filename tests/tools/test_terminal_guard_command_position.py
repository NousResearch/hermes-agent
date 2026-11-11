"""The long-lived-process guard refuses commands that start a server, never a command
that only names one as an argument, a path component or a grep target.

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
    "pip show uvicorn",
    "du -sh venv/lib/python3.11/site-packages/gunicorn",
    "ls node_modules/.bin/nodemon",
    "grep -rn nodemon package.json",
    "grep -rn vite apps/desktop/vite.config.ts",
    "echo vite nodemon uvicorn gunicorn",
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
    "git checkout -- vite",
    "sudo grep -rn gunicorn /etc",
    "cd web && ls node_modules/vite",
    "env LANG=C grep uvicorn requirements.txt",
    "timeout 10 grep uvicorn requirements.txt",
    "find . -name uvicorn",
    "tail -n 50 logs/uvicorn.log",
    "command -v nodemon",
    "/usr/bin/grep -rn uvicorn .",
    "( cd web && ls node_modules/vite )",
    "source .venv/bin/activate && pip install uvicorn",
    "grep -rl uvicorn . | wc -l",
    "rg --pretty vite",
    ".venv/bin/gunicorn --check-config app:app",
    "sudo -u www grep -rn gunicorn /etc",
    # bounded invocations of a server binary
    "vite build",
    "npx vite build",
    "gunicorn --check-config app:app",
    "docker compose up -d",
    "docker compose up --detach web",
    "docker compose up -d && curl -s localhost:3000/health",
    "docker compose up -d && sleep 5 && docker compose ps",
    ".venv/bin/python -m pip install uvicorn",
    "git switch -c bump-vite",
    "grep -q vite package.json 2>/dev/null && npm run build",
    "pip list | column -t | grep uvicorn",
    "grep -n uvicorn requirements.txt && tail -n 20 logs/app.log",
    "git -P grep uvicorn",
    "grep -c vite package.json" + " && true" * 250,
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
    "npx.cmd vite",
    "pnpm.cmd exec vite",
    "python.exe -m uvicorn app:app",
    r"C:\app\.venv\Scripts\uvicorn.exe app:app",
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
    "(cd web && npm run dev)",
    "if true; then npm run dev; fi",
    "doppler run -- npm run dev",
    "direnv exec . uvicorn app:app",
    "npm run start:dev",
    # a data command still runs what it substitutes, and a watched build never exits
    "echo $(npm run dev)",
    "vite build --watch",
    "nodemon --exec vite build",
    "docker compose up -d && docker compose logs --tail=100 -f",
    "docker compose up web -- -d",
    "watch docker compose up -d",
    "nodemon --exec gunicorn --check-config app:app",
    "vite build -wm dev",
    "vite build --watch=true",
    # a data command stops being data when a flag makes it run a command or never exit
    "find . -name '*.py' -exec gunicorn app:app +",
    "tail -fn 100 logs/uvicorn.log",
    "tail --follow=name uvicorn.log",
    "tail --fol uvicorn.log",
    "git bisect run npm run dev",
    "fd -e ts -x nodemon",
    "command uvicorn app:app",
    "sudo -p echo uvicorn app:app",
    "echo npm run dev | sh",
    "echo npm run dev | ssh host sh",
    "/opt/app/echo npm run dev",
    "rg --pre cat uvicorn",
    "docker compose up -d && tail -f logs/app.log",
    "docker compose up -d && kubectl logs -f deploy/web",
    "echo npm run dev | tee run.sh | zsh",
    "echo npm run dev > run.sh; sh run.sh",
    "echo npm run dev &> run.sh; sh run.sh",
    "echo npm run dev >/dev/null.sh; sh /dev/null.sh",
    "grep -n uvicorn requirements.txt && tail -f logs/app.log",
    "grep -n uvicorn requirements.txt && command tail -f logs/app.log",
    "tail -f logs/app.log; grep -n uvicorn requirements.txt",
    "vite build && docker compose logs -f",
    "npm --prefix install run dev:uvicorn",
    "uv --directory sync run uvicorn app:app",
    "echo npm run dev >/dev/null+x; sh /dev/null+x",
    "git --git-dir .git bisect run npm run dev",
    "docker compose up -d && npm test",
    "git -c core.pager=nodemon log",
    "git rebase -ix nodemon main",
    "./echo npm run dev",
    "case $1 in\n  test ) npm run dev ;;\nesac",
    "docker compose run web npm run dev up -d",
    "docker compose up -d && docker compose --profile dev logs -f",
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
