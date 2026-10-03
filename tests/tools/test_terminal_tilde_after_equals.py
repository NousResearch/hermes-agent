"""Regression tests for _expand_tilde_after_equals.

Context: bash expands ``~`` only at the start of a word or after ``=`` in a
word shaped like a variable assignment. ``--output=~/chart.png`` therefore
reaches the program as the literal ``~/chart.png``, which Node, Python and
most CLIs resolve against the cwd: the file lands in a directory named ``~``
and the command still exits 0. Agents write this form often (a renderer's
``--output=~/...`` in live use).

The rewriter applies zsh's MAGIC_EQUAL_SUBST rule, narrowed to what bash
leaves literal: an unquoted ``~`` or ``~/`` right after the first unquoted
``=`` of a flag or key word becomes ``"$HOME"``.
"""

import json
import shutil
import subprocess
import time
from unittest.mock import MagicMock, patch

import pytest

from tools.terminal_tool_sudo import _expand_tilde_after_equals as rewrite


class TestRewrites:
    """Flag and key values bash leaves literal MUST expand."""

    def test_long_flag(self):
        assert rewrite("capture-website a.html --output=~/chart.png") == (
            'capture-website a.html --output="$HOME"/chart.png'
        )

    def test_short_flag(self):
        assert rewrite("tool -o=~/out.txt") == 'tool -o="$HOME"/out.txt'

    def test_bare_home(self):
        assert rewrite("pip install --target=~ pkg") == 'pip install --target="$HOME" pkg'

    def test_dotted_key(self):
        assert rewrite("git -c core.hooksPath=~/hooks status") == (
            'git -c core.hooksPath="$HOME"/hooks status'
        )

    def test_dashed_key(self):
        assert rewrite("run cache-dir=~/cache") == 'run cache-dir="$HOME"/cache'

    def test_quoted_rest_of_value(self):
        assert rewrite('cp a --target-directory=~/"my files"') == (
            'cp a --target-directory="$HOME"/"my files"'
        )

    def test_every_command_of_a_list(self):
        assert rewrite("a --o=~/1; b --o=~/2 && c --o=~/3 | d --o=~/4") == (
            'a --o="$HOME"/1; b --o="$HOME"/2 && c --o="$HOME"/3 | d --o="$HOME"/4'
        )

    def test_inside_command_substitution(self):
        assert rewrite("echo $(tool --o=~/x)") == 'echo $(tool --o="$HOME"/x)'

    def test_inside_subshell(self):
        assert rewrite("(cd /tmp && tool --o=~/x)") == '(cd /tmp && tool --o="$HOME"/x)'

    def test_after_newline(self):
        assert rewrite("cd /tmp\ntool --o=~/x") == 'cd /tmp\ntool --o="$HOME"/x'

    def test_command_after_heredoc_body(self):
        assert rewrite("cat <<EOF\ntext\nEOF\ntool --o=~/x") == (
            'cat <<EOF\ntext\nEOF\ntool --o="$HOME"/x'
        )


class TestPreserved:
    """Everything bash already expands, or that is not a flag value, is untouched."""

    @pytest.mark.parametrize(
        "command",
        [
            "ls ~/x",
            "tool --o ~/x",
            "FOO=~/x tool",
            "export PREFIX=~/.local",
            "make PREFIX=~/.local install",
            "PATH+=:~/bin",
            "arr[0]=~/x",
            "tool --o='~/x'",
            'tool --o="~/x"',
            "tool '--o=~/x'",
            'tool "--o=~/x"',
            "tool --o=\\~/x",
            "tool --o=~user/x",
            "tool --o=~+",
            "tool --o=~x",
            "tool --o=x~/y",
            "tool --o=a=~/b",
            "tool =~/x",
            "curl http://host/?a=~/b",
            "curl -d key=~/x http://host",
            "[[ $a =~ ^x ]]",
            "[[ $a =~ ~/x ]]",
            "echo $((x=~5))",
            "echo $a=~/x",
            "echo '--o=~/x'",
            "echo hi # tool --o=~/x",
        ],
    )
    def test_unchanged(self, command):
        assert rewrite(command) == command

    def test_word_before_a_comment(self):
        assert rewrite("tool --o=~/x # note --o=~/y") == 'tool --o="$HOME"/x # note --o=~/y'

    def test_comment_line(self):
        assert rewrite("# tool --o=~/x\necho hi") == "# tool --o=~/x\necho hi"

    def test_heredoc_body(self):
        command = "cat <<EOF > f\n--o=~/x\nEOF"
        assert rewrite(command) == command

    def test_quoted_heredoc_body(self):
        command = "cat <<'EOF' > f\n--o=~/x\nEOF"
        assert rewrite(command) == command

    def test_double_quoted_heredoc_body(self):
        command = 'cat <<"EOF" > f\n--o=~/x\nEOF'
        assert rewrite(command) == command

    def test_spaced_heredoc_delimiter(self):
        command = "cat << EOF > f\n--o=~/x\nEOF"
        assert rewrite(command) == command

    def test_tab_stripped_heredoc_body(self):
        command = "cat <<-EOF > f\n\t--o=~/x\n\tEOF\ntool --o=~/y"
        assert rewrite(command) == "cat <<-EOF > f\n\t--o=~/x\n\tEOF\ntool --o=\"$HOME\"/y"

    def test_two_heredocs_on_one_line(self):
        command = "cmd <<A 3<<B\n--o=~/1\nA\n--o=~/2\nB"
        assert rewrite(command) == command

    def test_here_string_is_not_a_heredoc(self):
        assert rewrite("cat <<< hi\ntool --o=~/x") == 'cat <<< hi\ntool --o="$HOME"/x'

    def test_unterminated_quote(self):
        assert rewrite("tool --o='~/x") == "tool --o='~/x"

    def test_empty(self):
        assert rewrite("") == ""


class TestIdempotence:
    @pytest.mark.parametrize(
        "command",
        ["tool --o=~/x", "a --o=~ && b -p=~/y", "cat <<EOF\n--o=~/x\nEOF\n--o=~/z"],
    )
    def test_second_pass_changes_nothing(self, command):
        once = rewrite(command)
        assert rewrite(once) == once


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
class TestAgainstBash:
    """The rewrite keeps bash's own output wherever it does not target."""

    HOME = "/tmp/hermes home"

    def _run(self, command: str) -> str:
        return subprocess.run(
            ["bash", "-c", command],
            capture_output=True,
            text=True,
            env={"HOME": self.HOME, "PATH": "/usr/bin:/bin"},
            check=False,
        ).stdout

    @pytest.mark.parametrize(
        "command",
        [
            "printf '%s\\n' ~/x",
            "printf '%s\\n' --o ~/x",
            "printf '%s\\n' FOO=~/x",
            "FOO=~/x; printf '%s\\n' \"$FOO\"",
            "printf '%s\\n' --o='~/x' --o=\"~/x\" '--o=~/x' --o=\\~/x",
            "printf '%s\\n' --o=~root/x --o=x~/y --o=a=~/b =~/x",
            "printf '%s\\n' http://host/?a=~/b",
            "a=~/x; [[ $a =~ x$ ]] && printf '%s\\n' match",
            "printf '%s\\n' $((x=~5))",
            "cat <<EOF\n--o=~/x\nEOF",
            "cat <<-EOF\n\t--o=~/x\n\tEOF",
        ],
    )
    def test_same_output_as_the_original(self, command):
        assert self._run(rewrite(command)) == self._run(command)

    @pytest.mark.parametrize(
        ("command", "expected"),
        [
            ("printf '%s\\n' --o=~/x", "--o=/tmp/hermes home/x\n"),
            ("printf '%s\\n' --o=~", "--o=/tmp/hermes home\n"),
            ("printf '%s\\n' core.hooksPath=~/h", "core.hooksPath=/tmp/hermes home/h\n"),
            ("printf '%s\\n' --o=~/\"a b\"", "--o=/tmp/hermes home/a b\n"),
            ("printf '%s\\n' $(printf '%s' --o=~/x)", "--o=/tmp/hermes\nhome/x\n"),
        ],
    )
    def test_expands_like_an_assignment(self, command, expected):
        assert self._run(rewrite(command)) == expected

    def test_home_with_a_space_stays_one_argument(self):
        assert self._run(rewrite("printf '[%s]' --o=~/x")) == "[--o=/tmp/hermes home/x]"


class TestTerminalTool:
    """Real local runs: the executed copy expands; guards see the command as written."""

    WRITER = "python3 -c 'import pathlib,sys; pathlib.Path(sys.argv[1].split(\"=\", 1)[1]).write_text(\"hi\")'"

    @staticmethod
    def _config(cwd):
        return {
            "env_type": "local",
            "timeout": 180,
            "cwd": str(cwd),
            "host_cwd": None,
            "modal_mode": "auto",
            "docker_image": "",
            "singularity_image": "",
            "modal_image": "",
            "daytona_image": "",
        }

    def _run(self, tmp_path, monkeypatch, **kwargs):
        from tools.terminal_tool import terminal_tool

        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hh"))
        home = tmp_path / "home dir"
        home.mkdir()
        work = tmp_path / "work"
        work.mkdir()
        command = f"export HOME='{home}'; {self.WRITER} --out=~/made.txt"
        with patch("tools.terminal_tool._get_env_config", return_value=self._config(work)), \
             patch("tools.terminal_tool._start_cleanup_thread"), \
             patch("tools.terminal_tool._check_all_guards", return_value={"approved": True}):
            result = json.loads(terminal_tool(command=command, **kwargs))
        return result, home, work

    def test_foreground_writes_under_home(self, tmp_path, monkeypatch):
        result, home, work = self._run(tmp_path, monkeypatch)

        assert (result.get("exit_code"), (home / "made.txt").read_text(), (work / "~").exists()) == (
            0,
            "hi",
            False,
        )

    def test_background_writes_under_home(self, tmp_path, monkeypatch):
        result, home, work = self._run(tmp_path, monkeypatch, background=True)
        deadline = time.time() + 10
        while not (home / "made.txt").exists() and time.time() < deadline:
            time.sleep(0.05)

        assert (result.get("error"), (home / "made.txt").exists(), (work / "~").exists()) == (
            None,
            True,
            False,
        )

    def test_guards_see_the_command_as_written(self, tmp_path, monkeypatch):
        from tools.terminal_tool import terminal_tool

        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hh"))
        guards = MagicMock(return_value={"approved": True})
        with patch("tools.terminal_tool._get_env_config", return_value=self._config(tmp_path)), \
             patch("tools.terminal_tool._start_cleanup_thread"), \
             patch("tools.terminal_tool._check_all_guards", guards):
            terminal_tool(command="true --output=~/x.png")

        assert guards.call_args[0][0] == "true --output=~/x.png"
