"""Docker lifecycle rules treat quoted documentation as prose, never executed text."""

import time

import pytest

from tools.approval import detect_dangerous_command


class TestDockerLifecycleQuotedProse:
    """Docker lifecycle text is safe documentation unless a shell executes it.

    Prose that leaves its command (a file write, a pipe into a non-display
    command) or rides a hermes subcommand that stores or runs its arguments
    keeps the scan, and shell-carrier payloads remain code.
    """

    @pytest.mark.parametrize("command", [
        "echo doc-only: to restart the stack later, run: docker restart web-1",
        "echo doc-only: docker stop web-1 stops one container",
        "echo doc-only: docker kill web-1 kills one container",
        "echo doc-only: docker compose restart restarts the stack",
        "echo doc-only: docker compose stop stops the stack",
        "echo 'README: docker compose kill kills the stack'",
        "echo 'README: docker compose down stops the stack'",
        "cat <<'README'\ndocker-compose stop app\ndocker-compose kill app\nREADME",
        'echo "include docker restart findings"',
        'hermes kanban create "Nightly report" --body "Include docker restart/OOM counts from the logs."',
        'git commit -m "note docker restart counts"',
        'grep "docker restart" log.txt',
    ])
    def test_documentation_mentions_do_not_require_approval(self, command):
        assert detect_dangerous_command(command) == (False, None, None), command

    @pytest.mark.parametrize("command,expected", [
        ("docker restart app", "docker restart/stop/kill (container lifecycle)"),
        ("docker compose down", "docker compose restart/stop/kill/down (container lifecycle)"),
        ("docker-compose stop app", "docker compose restart/stop/kill/down (container lifecycle)"),
        ("cat <<'README'\ndocker restart docs-only\nREADME\ndocker restart app",
         "docker restart/stop/kill (container lifecycle)"),
        # Commands that execute their arguments or stdin are never prose.
        ("docker ps -q | xargs docker stop", "docker restart/stop/kill (container lifecycle)"),
        ("ssh host <<EOF\ndocker restart app\nEOF", "docker restart/stop/kill (container lifecycle)"),
        ("ssh host docker restart app", "docker restart/stop/kill (container lifecycle)"),
        ('bash -c "docker restart app"', "docker restart/stop/kill (container lifecycle)"),
        ("eval 'docker stop app'", "docker restart/stop/kill (container lifecycle)"),
        ("echo done; docker kill app", "docker restart/stop/kill (container lifecycle)"),
        ("git log; docker stop app", "docker restart/stop/kill (container lifecycle)"),
        ("printf '%s' \"$(docker compose down)\"",
         "docker compose restart/stop/kill/down (container lifecycle)"),
        ("git -c alias.x='!docker stop app' x", "docker restart/stop/kill (container lifecycle)"),
        ('echo "$(docker kill app)"', "docker restart/stop/kill (container lifecycle)"),
        # An unquoted here-doc delimiter expands $(...) and backticks in the body, even for cat.
        ("cat <<EOF\n$(docker stop app)\nEOF", "docker restart/stop/kill (container lifecycle)"),
        ("cat <<EOF\n`docker kill app`\nEOF", "docker restart/stop/kill (container lifecycle)"),
    ])
    def test_real_lifecycle_commands_remain_flagged(self, command, expected):
        assert detect_dangerous_command(command) == (True, expected, expected)

    def test_large_prose_heredoc_stays_fast(self):
        body = "\n".join(f'"line {i}" docker restart app' for i in range(600))
        command = f"hermes kanban create notes --body-file - <<'EOF'\n{body}\nEOF"
        started = time.perf_counter()
        assert detect_dangerous_command(command) == (False, None, None)
        assert time.perf_counter() - started < 10

    @pytest.mark.parametrize("command", [
        "grep 'docker restart' log.txt 2>&1 | wc -l",
        "echo 'docker restart app' >&2",
        "git log --grep='docker restart' | grep -v merge | head -5",
        'hermes kanban comment t1 "ran docker restart app"',
    ])
    def test_prose_into_fd_or_display_sink_stays_prose(self, command):
        assert detect_dangerous_command(command) == (False, None, None), command

    @pytest.mark.parametrize("command", [
        # hermes is prose only for kanban text subcommands; config/cron/plugins store or run args.
        "hermes config set command_allowlist '[\"docker restart app\"]'",
        "hermes config set mcp_allowlist '[\"docker restart app\"]'",
        "hermes cron create 'every 1h' 'docker restart app'",
        "echo hi; hermes run 'docker compose down'",
        "hermes -p work kanban create t --body 'docker restart app'",
        # Prose written to a file or fed to another command may be persisted or executed.
        "echo 'docker restart app' > deploy.sh",
        "echo 'docker restart app' &> deploy.sh",
        "echo 'docker restart app' | ssh prod",
        "echo 'docker restart app' 2>&1 | ssh prod",
        "echo 'docker restart app' | cat | ssh prod",
        "printf 'docker stop app\\n' | sudo tee run.sh",
        "echo 'docker restart app' |",
        "cat <<'EOF' > deploy.sh\ndocker restart app\nEOF",
        "cat <<'EOF' | ssh prod\ndocker restart app\nEOF",
    ])
    def test_prose_that_leaves_the_command_remains_flagged(self, command):
        assert detect_dangerous_command(command)[0] is True, command

    def test_shell_carrier_payload_remains_flagged(self):
        command = "bash -c 'echo \"docker restart app\"'"
        assert detect_dangerous_command(command) == (
            True,
            "docker restart/stop/kill (container lifecycle)",
            "docker restart/stop/kill (container lifecycle)",
        )
