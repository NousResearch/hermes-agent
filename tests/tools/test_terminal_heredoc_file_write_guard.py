"""Regression tests for the unquoted-heredoc-file-write guardrail.

Real-world trigger: a session wrote a Next.js page via
``cat > page.tsx <<EOF ... EOF`` (unquoted delimiter) instead of
``write_file``. The body contained a TypeScript template literal
(`` `${something}` ``); bash expanded the backtick-quoted span as command
substitution before the content ever reached the file, silently replacing
it with the (empty) output of that "command." The write reported success --
there was no error anywhere in the chain, only a wrong file on disk.

``detect_unquoted_heredoc_file_write`` (``tools/shell_heredoc.py``) blocks
this specific combination -- a heredoc feeding a file-write consumer
(``cat >``/``cat >>``/a bare redirect) with an unquoted delimiter -- while
leaving heredocs that feed an interpreter (python, psql, docker exec -i)
untouched, since those are a common and legitimate pattern this guard must
not break.
"""

import json

from tools.shell_heredoc import detect_unquoted_heredoc_file_write
from tools.terminal_tool import detect_unquoted_heredoc_file_write as guard

BT = chr(96)
NL = chr(10)


class TestUnquotedFileWriteBlocked:
    """The exact dangerous combination: unquoted delimiter + file-write consumer."""

    def test_cat_redirect_unquoted_delimiter_with_backtick_body(self):
        cmd = (
            "cat > page.tsx <<EOF" + NL
            + "title: " + BT + "${something}" + BT + "," + NL
            + "EOF"
        )
        result = detect_unquoted_heredoc_file_write(cmd)
        assert result is not None
        assert "write_file" in result
        assert "EOF" in result

    def test_cat_append_redirect_unquoted_delimiter(self):
        cmd = "cat >> notes.txt <<EOF" + NL + "some text" + NL + "EOF"
        assert detect_unquoted_heredoc_file_write(cmd) is not None

    def test_bare_redirect_no_cat_prefix(self):
        cmd = "> out.txt <<EOF" + NL + "content" + NL + "EOF"
        assert detect_unquoted_heredoc_file_write(cmd) is not None

    def test_unquoted_delimiter_with_dollar_expansion_body(self):
        cmd = "cat > deploy.sh <<EOF" + NL + "echo $HOME" + NL + "EOF"
        assert detect_unquoted_heredoc_file_write(cmd) is not None

    def test_env_prefix_before_cat(self):
        cmd = "LC_ALL=C cat > out.txt <<EOF" + NL + "text" + NL + "EOF"
        assert detect_unquoted_heredoc_file_write(cmd) is not None

    def test_redirect_after_heredoc_operator(self):
        # `cat <<EOF > file` -- the file redirect comes AFTER the heredoc
        # operator on the opener line, not before it. Must be caught the
        # same as `cat > file <<EOF`; the danger doesn't depend on order.
        cmd = "cat <<EOF > file.txt" + NL + "content" + NL + "EOF"
        assert detect_unquoted_heredoc_file_write(cmd) is not None


class TestNonFileRedirectTargetsAllowed:
    """A `>` that doesn't point at a plain file is not a file-write consumer."""

    def test_fd_duplication_not_a_file(self):
        # `>&2` duplicates stdout onto fd 2 (stderr) -- not a file target.
        cmd = "cat >&2 <<EOF" + NL + "content" + NL + "EOF"
        assert detect_unquoted_heredoc_file_write(cmd) is None

    def test_process_substitution_not_a_file(self):
        # `>(cmd)` is process substitution -- writes to a pipe feeding a
        # subprocess, not a plain file.
        cmd = "cat > >(tee log.txt) <<EOF" + NL + "content" + NL + "EOF"
        assert detect_unquoted_heredoc_file_write(cmd) is None

    def test_append_fd_duplication_not_a_file(self):
        cmd = "cat >>&2 <<EOF" + NL + "content" + NL + "EOF"
        assert detect_unquoted_heredoc_file_write(cmd) is None


class TestQuotedFileWriteAllowed:
    """A quoted delimiter is inert -- no shell expansion, safe as written."""

    def test_single_quoted_delimiter(self):
        cmd = (
            "cat > page.tsx <<'EOF'" + NL
            + "title: " + BT + "${something}" + BT + "," + NL
            + "EOF"
        )
        assert detect_unquoted_heredoc_file_write(cmd) is None

    def test_double_quoted_delimiter(self):
        cmd = 'cat > page.tsx <<"EOF"' + NL + "content" + NL + "EOF"
        assert detect_unquoted_heredoc_file_write(cmd) is None

    def test_backslash_escaped_delimiter(self):
        # `<<\EOF` -- a leading backslash marks the whole delimiter word
        # quoted per bash heredoc rules, same as `<<'EOF'`.
        cmd = "cat > page.tsx <<\\EOF" + NL + "content" + NL + "EOF"
        assert detect_unquoted_heredoc_file_write(cmd) is None


class TestNonFileWriteConsumersUntouched:
    """Heredocs feeding an interpreter are a common pattern -- must not block them."""

    def test_python_interpreter_unquoted(self):
        cmd = "python3 <<EOF" + NL + "print(1)" + NL + "EOF"
        assert detect_unquoted_heredoc_file_write(cmd) is None

    def test_docker_exec_unquoted(self):
        cmd = "docker exec -i mycontainer bash <<EOF" + NL + "ls" + NL + "EOF"
        assert detect_unquoted_heredoc_file_write(cmd) is None

    def test_psql_unquoted(self):
        cmd = "psql -h localhost <<EOF" + NL + "SELECT 1;" + NL + "EOF"
        assert detect_unquoted_heredoc_file_write(cmd) is None

    def test_no_heredoc_at_all(self):
        assert detect_unquoted_heredoc_file_write("cat file.txt") is None

    def test_plain_redirect_no_heredoc(self):
        assert detect_unquoted_heredoc_file_write("echo hello > file.txt") is None

    def test_cat_reading_not_writing(self):
        assert detect_unquoted_heredoc_file_write("cat file.txt | grep x") is None

    def test_interpreter_with_own_unrelated_output_redirect(self):
        # `python3 <<EOF > out.log` -- the heredoc feeds python as its
        # script (safe, same as any other interpreter case); the `>` on
        # this line redirects PYTHON's own stdout, not the heredoc body.
        # A file-write redirect being present somewhere on the opener must
        # not be enough to flag this on its own.
        cmd = "python3 <<EOF > out.log" + NL + "print(1)" + NL + "EOF"
        assert detect_unquoted_heredoc_file_write(cmd) is None

    def test_bash_interpreter_with_own_output_redirect(self):
        cmd = "bash <<EOF > out.log" + NL + "echo hi" + NL + "EOF"
        assert detect_unquoted_heredoc_file_write(cmd) is None


class TestTerminalToolReexport:
    """terminal_tool.py imports and wires this in -- confirm the same function."""

    def test_same_function_object(self):
        assert guard is detect_unquoted_heredoc_file_write


class TestTerminalToolActuallyBlocks:
    """Integration-level: terminal_tool() itself hard-blocks before executing.

    The unit tests above only exercise the detector function directly --
    they protect the detection logic but not the wiring. A future refactor
    could silently drop or reorder the guard inside terminal_tool() without
    any of those tests catching it. These call terminal_tool() itself and
    assert on its actual response, the same JSON contract an agent sees.
    """

    def test_dangerous_command_is_blocked_with_no_execution(self):
        from tools.terminal_tool import terminal_tool

        cmd = "cat > page.tsx <<EOF" + NL + "title: " + BT + "${x}" + BT + "," + NL + "EOF"
        result = json.loads(terminal_tool(command=cmd))

        assert result["status"] == "error"
        assert result["exit_code"] == -1
        assert "write_file" in result["error"]
        assert result["output"] == ""

    def test_safe_quoted_variant_is_not_blocked_by_this_guard(self):
        # Deliberately NOT calling terminal_tool() here: the quoted variant is
        # allowed, so a real call would execute the command and write a file
        # as a test side effect. The guard's own decision is the thing under
        # test, and the block above already proves terminal_tool() acts on it.
        cmd = "cat > page.tsx <<'EOF'" + NL + "title: " + BT + "${x}" + BT + "," + NL + "EOF"
        assert guard(cmd) is None
