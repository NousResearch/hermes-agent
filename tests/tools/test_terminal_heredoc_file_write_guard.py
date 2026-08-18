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


class TestTerminalToolReexport:
    """terminal_tool.py imports and wires this in -- confirm the same function."""

    def test_same_function_object(self):
        assert guard is detect_unquoted_heredoc_file_write
