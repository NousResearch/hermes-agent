"""Docker lifecycle exemptions require a quoted message in one inert command."""

import time

import pytest

from tools.approval import detect_dangerous_command


class TestDockerLifecycleQuotedProse:
    """Anything outside the strict quoted-message subset retains the base scan."""

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
    def test_documentation_contract(self, command):
        expected_flag = command.startswith(("echo doc-only:", "cat <<", "grep "))
        assert detect_dangerous_command(command)[0] is expected_flag, command

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
        assert detect_dangerous_command(command)[0] is True
        assert time.perf_counter() - started < 10

    @pytest.mark.parametrize("command", [
        "grep 'docker restart' log.txt 2>&1 | wc -l",
        "echo 'docker restart app' >&2",
        "git log --grep='docker restart' | grep -v merge | head -5",
        'hermes kanban comment t1 "ran docker restart app"',
    ])
    def test_redirection_and_pipeline_do_not_qualify(self, command):
        assert detect_dangerous_command(command)[0] is (not command.startswith("hermes ")), command

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

    @pytest.mark.parametrize("command", [
        # git grep -O / --open-files-in-pager runs a pager. A detached word is scanned too:
        # the synopsis allows a separate pager, and stuck/equals forms execute it.
        "git grep -O 'docker restart app'",
        "git grep -O'docker restart app' .",
        'git grep -O"docker restart app" .',
        "git grep -nO'docker restart app' .",
        "git grep --open-files-in-pager='docker restart app' .",
        "git grep --open-files-in-pager 'docker restart app' .",
        "git grep --open='docker stop app' .",
        "git grep --op='docker kill app' .",
        "git grep $'--open-files-in-pager=docker stop victim' needle",
        "git grep $'--open=docker stop victim' needle",
        "git grep $'--op=docker stop victim' needle",
        "echo $'x\\\''; docker stop victim",
        "printf $'x\\\''; docker kill victim",
        "git commit -m $'x\\\''; docker stop victim",
        "hermes kanban comment t1 $'x\\\''; docker stop victim",
        "(echo 'docker restart victim') | busybox sh",
        "(echo 'docker restart victim') > deploy.sh",
        "(echo 'docker stop victim' | head) | busybox ash",
        "(cat <<'EOF'\ndocker kill victim\nEOF\n) | busybox sh",
        "busybox sh < <(cat <<'EOF'\ndocker restart victim\nEOF\n)",
        "{ cat <<'EOF'\ndocker stop victim\nEOF\n} | busybox sh",
        "/usr/bin/git grep -O'docker restart app' .",
        # Helpers the following command executes, including prefix and persisted assignments.
        "GIT_SSH_COMMAND='docker stop app' git fetch",
        "GIT_SSH='docker stop app' git fetch",
        "GIT_ASKPASS='docker stop app' git fetch",
        "GIT_PROXY_COMMAND='docker stop app' git fetch",
        "GIT_EXTERNAL_DIFF='docker restart app' git diff",
        "GIT_EDITOR='docker restart app' git commit",
        "GIT_EDITOR='docker restart app' git commit -m 'msg'",
        "GIT_EDITOR='docker restart app' command git commit",
        "GIT_EDITOR='docker restart app' timeout 5 git commit",
        "command git grep -O 'docker restart app'",
        "GIT_SEQUENCE_EDITOR='docker restart app' git rebase -i",
        "EDITOR='docker restart app' git commit",
        "VISUAL='docker restart app' git commit",
        "GIT_PAGER='docker restart app' git --paginate log",
        "GIT_PAGER='docker restart app' git log",
        "PAGER='docker restart app' git --paginate log",
        "MANPAGER='docker restart app' man ls",
        "PAGER='docker restart app' man ls",
        "LESSOPEN='|docker stop app %s' less README",
        "LESSOPEN='docker stop app %s' less README",
        "LESSCLOSE='docker kill app %s' less README",
        "FOO=1 GIT_SSH_COMMAND='docker stop app' git fetch",
        "GIT_EDITOR='docker restart app' sudo git commit",
        "GIT_EDITOR='doc''ker restart app' git commit",
        "GIT_EDITOR=docker\\ restart\\ app git commit",
        "GIT_EDITOR='docker restart app'; git commit",
        "GIT_EDITOR='docker restart app'\ngit commit",
        "echo safe | LESSOPEN='|docker stop app %s' less",
        "GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=core.editor GIT_CONFIG_VALUE_0='docker restart app' git commit",
        "GIT_CONFIG_KEY_0=alias.x GIT_CONFIG_VALUE_0='!docker stop app' git status",
        "GIT_CONFIG_VALUE_0='docker restart app' git commit",
        "GIT_CONFIG_PARAMETERS=\"'core.editor'='docker restart app'\" git commit",
        "GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=credential.helper GIT_CONFIG_VALUE_0='!docker stop app' git credential fill",
        "GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=core.fsmonitor GIT_CONFIG_VALUE_0='docker stop app' git status",
        "GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=gpg.ssh.defaultKeyCommand GIT_CONFIG_VALUE_0='docker stop app' git commit -S",
        "EVIL='!docker stop app' git --config-env=credential.helper=EVIL credential fill",
        "FOO='docker restart app' echo \"$FOO\"",
        "FOO='docker restart app'; echo \"$FOO\"",
        "GIT_EDITOR='docker restart app' echo \"$GIT_EDITOR\"",
        "GIT_EDITOR='docker restart app'; echo hi",
        "GIT_EDITOR='docker restart app' echo hi; git status",
        "GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=user.name GIT_CONFIG_VALUE_0='docker restart app' git status",
    ])
    def test_execution_carriers_remain_flagged(self, command):
        assert detect_dangerous_command(command) == (
            True,
            "docker restart/stop/kill (container lifecycle)",
            "docker restart/stop/kill (container lifecycle)",
        ), command

    @pytest.mark.parametrize("command", [
        "echo \"git grep -O 'docker restart app'\"",
        "git grep 'docker restart' log.txt",
        "git grep --open-files-in-pager=less 'docker restart'",
        "git grep --no-open-files-in-pager 'docker restart' file",
        "git grep -e '-O' 'docker restart' file",
        "git commit -m 'note docker restart counts'",
        "echo 'docker restart app' | less",
        "GIT_CONFIG_KEY_0=alias.x GIT_CONFIG_VALUE_0='status' git status",
        "MANPAGER=less man ls",
    ])
    def test_carrier_allowlist_is_narrow(self, command):
        assert detect_dangerous_command(command)[0] is (command.startswith("git grep") or " | " in command), command
