"""HOME-path classification invariants (regression for #84639 / #98868).

Command strings are classified only; none of the displayed paths are accessed.
Windows and UNC are path data, not claims about native shell execution.
"""

import pytest

from tools.approval import detect_dangerous_command


_HOMES = [
    "/root", "/tester", "/home/tester", "/home/test user",
    "C:/Accounts/tester", r"C:\Accounts\tester", "//server/share", r"\\server\share",
]
_WRITES = [
    "echo placeholder > {path}", "echo placeholder >> {path}",
    "echo placeholder | tee -a {path}", "cp placeholder {path}",
    "mv placeholder {path}", "install placeholder {path}",
    "sed -i 's/a/b/' {path}", "perl -pi {path}", "ruby -pi {path}",
]
_SPELLINGS = [(path, ".ssh/authorized_keys") for path in [
    "{home}/.ssh/authorized_keys", '"{home}/.ssh/key with spaces"',
    "{home}//.ssh/authorized_keys", "{home}/./.ssh/authorized_keys",
    "{home}/../{name}/.ssh/authorized_keys", r"{home}\/\.ssh/authorized_keys",
    '"{home}/.ssh/authorized_keys"', "'{home}/.ssh/authorized_keys'",
    '{home}/".ssh"/authorized_keys', "~/./.ssh/authorized_keys",
    "~//.ssh/authorized_keys", '"${HOME}/./.ssh/authorized_keys"',
]] + [("{home}/" + target, target) for target in [
    ".ssh/id_rsa", ".netrc", ".pgpass", ".npmrc", ".pypirc", ".bashrc",
    ".zshrc", ".profile", ".bash_profile", ".zprofile", ".hermes/.env", ".hermes/config.yaml",
]]


@pytest.mark.parametrize("home", _HOMES)
@pytest.mark.parametrize("spelling,target", _SPELLINGS)
@pytest.mark.parametrize("operation", _WRITES + [
    "rm -rf {path}",
    pytest.param('echo "$(echo placeholder >> {path})"', id="nested-substitution"),
])
def test_sensitive_home_spellings_agree(home, spelling, target, operation, monkeypatch):
    monkeypatch.setenv("HOME", home)
    name = home.replace("\\", "/").rsplit("/", 1)[-1]
    if " " in name:
        name = '"' + name + '"'
    home_word = home
    if " " in home and not spelling.startswith(("'", '"')):
        home_word = '"' + home + '"'
    path = spelling.replace("{home}", home_word).replace("{name}", name)
    command = operation.format(path=path)
    control = detect_dangerous_command(operation.format(path="~/" + target))
    assert control[0] and control[1] is not None
    assert detect_dangerous_command(command) == control, command


@pytest.mark.parametrize("home,path", [
    ("/root", "/srv/root/.bashrc"), ("/root", "root/.bashrc"),
    ("/home/tester", "/srv/home/tester/.bashrc"), ("/home/tester", "home/tester/.bashrc"),
    ("/root", "/root-other/.bashrc"), ("/root", "/root/notes.txt"),
    ("/root", "/root/reports/out.txt"), ("/root", "/root/.ssh-backup/key"),
    ("/root", "/root/.bashrc_backup"), ("/root", "/root/.netrc_backup"),
    ("/root", "/root/.ssh/../notes.txt"),
    ("", "/.bashrc"), ("/", "/.bashrc"), ("//", "//.bashrc"),
    ("//host", "//host/.bashrc"), ("/C:", "/C:/.bashrc"),
    ("C:/", "C:/.bashrc"), ("C:\\", "C:\\.bashrc"), (r"\\host", r"\\host\.bashrc"),
    ("C:/Accounts/tester", "C:/Accounts/tester/notes.txt"),
    (r"C:\Accounts\tester", r"C:\Accounts\tester\notes.txt"),
    ("//server/share", "//server/share/notes.txt"),
    (r"\\server\share", r"\\server\share\notes.txt"),
])
@pytest.mark.parametrize("operation", _WRITES + [
    "cat {path}", "cp {path} ordinary-backup", "sed -n '1p' {path}",
])
def test_non_home_paths_stay_allowed(home, path, operation, monkeypatch):
    monkeypatch.setenv("HOME", home)
    assert detect_dangerous_command(operation.format(path=path)) == (False, None, None)
    # Read/copy-out are not sensitive writes, even for true HOME credential paths.
    for control in (
        "cat ~/.bashrc", "cat ~/.ssh/id_rsa", "cp ~/.netrc ordinary-backup",
        "cp '~/.ssh/key with spaces' 'ordinary backup'",
        "echo 'echo placeholder >> /root/.bashrc'",
    ):
        assert detect_dangerous_command(control) == (False, None, None)

@pytest.mark.parametrize("home", ["/home/quality-user", "/root", "C:/Accounts/quality-user", "//server/share/quality-user"])
@pytest.mark.parametrize("literal", [
    "key$literal", "key?literal", "key*literal", "key[literal]", "key{literal}", "key`literal`",
])
@pytest.mark.parametrize("quoting", ["single", "double", "escaped"])
@pytest.mark.parametrize("operation", ["echo placeholder > {path}", "cp placeholder {path}", "sed -i 's/a/b/' {path}"])
def test_literal_sensitive_descendants_keep_protection(home, literal, quoting, operation, monkeypatch):
    """A literal special character does not undo an already known credential prefix."""
    monkeypatch.setenv("HOME", home)

    def render(path):
        if quoting == "single":
            return "'" + path + "'"
        if quoting == "double":
            return '"' + path.replace("$", "\\$").replace("`", "\\`") + '"'
        return "".join("\\" + ch if ch in "$`*?[]{}" else ch for ch in path)

    control = detect_dangerous_command(operation.format(path=render("~/.ssh/" + literal)))
    assert control[0] and control[1] is not None
    assert detect_dangerous_command(operation.format(path=render(home + "/.ssh/" + literal))) == control
    for path in (
        home + "/notes/" + literal,
        home + "/.ssh-backup/" + literal,
        home + "-other/.ssh/" + literal,
        "/srv" + home + "/.ssh/" + literal,
        home + "/.ssh/../notes/" + literal,
    ):
        assert detect_dangerous_command(operation.format(path=render(path))) == (False, None, None), path
    for read in ("cat {path}", "cp {path} ordinary-backup", "sed -n '1p' {path}"):
        assert detect_dangerous_command(read.format(path=render(home + "/.ssh/" + literal))) == (False, None, None)


@pytest.mark.parametrize("home", ["/home/quality-user", "/root", "C:/Users/quality-user", "//server/share/quality-user"])
@pytest.mark.parametrize("dynamic", [
    "$UNKNOWN", "`unknown`", "*", "?", "[ab]", "{a,b}",
    '"$UNKNOWN"', '"`unknown`"', '$(unknown)', '"$(unknown)"',
])
@pytest.mark.parametrize("operation", [
    "echo placeholder > {path}", "cp placeholder {path}", "sed -i 's/a/b/' {path}",
])
def test_unknown_expansions_are_not_lexically_resolved(home, dynamic, operation, monkeypatch):
    """Do not collapse .. across an unknown shell expansion into a protected path."""
    monkeypatch.setenv("HOME", home)
    path = home + "/notes/" + dynamic + "/../../.ssh/id_rsa"
    assert detect_dangerous_command(operation.format(path=path)) == (False, None, None)
    ordinary = home + "/notes/" + dynamic + "/../../out.txt"
    assert detect_dangerous_command(operation.format(path=ordinary)) == (False, None, None)
    for read in ("cat {path}", "cp {path} ordinary-backup", "sed -n '1p' {path}"):
        assert detect_dangerous_command(read.format(path=path)) == (False, None, None)


@pytest.mark.parametrize("home", [r"C:\Users\quality-user", r"\\server\share\quality-user"])
@pytest.mark.parametrize("dynamic", ["*", "$UNKNOWN", "?", "key?literal", "[ab]", "{a,b}", "`unknown`"])
@pytest.mark.parametrize("operation", ["echo placeholder > {path}", "cp input {path}", "sed -i 's/a/b/' {path}"])
def test_native_separator_unknown_expansions_stay_unresolved(home, dynamic, operation, monkeypatch):
    """Native separators must not hide unknown syntax before lexical traversal."""
    monkeypatch.setenv("HOME", home)
    path = home + "\\notes\\" + dynamic + r"\..\..\.ssh\id_rsa"
    for spelling in (path, path.replace("\\", "/")):
        assert detect_dangerous_command(operation.format(path=spelling)) == (False, None, None)
        for read in ("cat {path}", "cp {path} ordinary-backup", "sed -n '1p' {path}"):
            assert detect_dangerous_command(read.format(path=spelling)) == (False, None, None)
    for relative in (r"\notes\out.txt", r"\.ssh-backup\id_rsa", r"\.ssh\..\notes.txt"):
        assert detect_dangerous_command(operation.format(path=home + relative)) == (False, None, None)


@pytest.mark.parametrize("home", [r"C:\Users\quality-user", r"\\server\share\quality-user"])
@pytest.mark.parametrize("special,quote,resolved", [
    ("*", "'", True), ("$UNKNOWN", "'", True), ("`unknown`", "'", True),
    ("?", "'", True), ("[ab]", "'", True), ("{a,b}", "'", True),
    ("*", '"', True), ("?", '"', True), ("[ab]", '"', True), ("{a,b}", '"', True),
    ("$UNKNOWN", '"', False), ("`unknown`", '"', False),
])
@pytest.mark.parametrize("operation", ["echo placeholder > {path}", "cp input {path}", "sed -i 's/a/b/' {path}"])
def test_native_separator_quote_provenance(home, special, quote, resolved, operation, monkeypatch):
    """Keep native single-quoted syntax and double-quoted globs literal."""
    monkeypatch.setenv("HOME", home)
    path = home + "\\notes\\" + special + r"\..\..\.ssh\id_rsa"
    control = detect_dangerous_command(operation.format(path="~/.ssh/id_rsa"))
    assert control[0] and control[1] is not None
    expected = control if resolved else (False, None, None)
    for spelling in (path, path.replace("\\", "/")):
        assert detect_dangerous_command(operation.format(path=quote + spelling + quote)) == expected
        for read in ("cat {path}", "cp {path} ordinary-backup", "sed -n '1p' {path}"):
            assert detect_dangerous_command(read.format(path=quote + spelling + quote)) == (False, None, None)
    ordinary = home + "\\notes\\" + special + r"\..\..\out.txt"
    assert detect_dangerous_command(operation.format(path=quote + ordinary + quote)) == (False, None, None)


@pytest.mark.parametrize("home", ["/home/quality-user", "/root"])
@pytest.mark.parametrize("literal", [r"\*", r"\$UNKNOWN", r"\?", r"\[ab\]", r"\{a,b\}", r"\`unknown\`"])
@pytest.mark.parametrize("operation", ["echo placeholder > {path}", "cp input {path}", "sed -i 's/a/b/' {path}"])
def test_posix_escaped_literals_keep_provenance(home, literal, operation, monkeypatch):
    """Native separator projection must not consume POSIX literal escapes."""
    monkeypatch.setenv("HOME", home)
    path = home + "/notes/" + literal + "/../../.ssh/id_rsa"
    control = detect_dangerous_command(operation.format(path="~/.ssh/id_rsa"))
    assert control[0] and control[1] is not None
    assert detect_dangerous_command(operation.format(path=path)) == control
    for read in ("cat {path}", "cp {path} ordinary-backup", "sed -n '1p' {path}"):
        assert detect_dangerous_command(read.format(path=path)) == (False, None, None)
    ordinary = home + "/notes/" + literal + "/../../out.txt"
    assert detect_dangerous_command(operation.format(path=ordinary)) == (False, None, None)


@pytest.mark.parametrize("path", [
    pytest.param('"C:/Users/quality-user/.ssh/key\\$literal"', id="double-quoted-dollar"),
    pytest.param(r"C:/Users/quality-user/.ssh/key\$literal", id="escaped-dollar"),
    pytest.param(r"C:/Users/quality-user/.ssh/key\?literal", id="escaped-question"),
])
@pytest.mark.parametrize("operation", [
    "cat {path}", "cp {path} ordinary-backup", "sed -n '1p' {path}",
])
def test_forward_drive_escaped_literals_preserve_read_boundary(path, operation, monkeypatch):
    """Literal escapes must not turn read operands into Windows SSH access prompts."""
    monkeypatch.setenv("HOME", "C:/Users/quality-user")
    allowed = (False, None, None)
    for control in ('"C:/Users/quality-user/.ssh/id_rsa"',
                    r'"C:\Users\quality-user\.ssh\id_rsa"',
                    path.replace("/.ssh/", "/notes/")):
        assert detect_dangerous_command(operation.format(path=control)) == allowed
    for write in ("echo placeholder > {path}", "cp input {path}", "sed -i 's/a/b/' {path}"):
        protected = detect_dangerous_command(write.format(path=path))
        assert protected[0] and protected[1] is not None
    assert detect_dangerous_command(operation.format(path=path)) == allowed


@pytest.mark.parametrize("path", [r"C:/Accounts/tester\.bashrc", r"C:/Accounts\tester/.bashrc"])
@pytest.mark.parametrize("quote", ["", '"'])
@pytest.mark.parametrize("operation", ["echo placeholder > {path}", "cp placeholder {path}", "sed -i 's/a/b/' {path}"])
def test_mixed_separator_home_writes_keep_protection(path, quote, operation, monkeypatch):
    """The exact twelve lost mixed-separator writes retain operation-specific verdicts."""
    monkeypatch.setenv("HOME", "C:/Accounts/tester")
    control = detect_dangerous_command(operation.format(path="~/.bashrc"))
    assert control[0] and control[1] is not None
    operand = quote + path + quote
    assert detect_dangerous_command(operation.format(path=operand)) == control
    ordinary = quote + path.replace(".bashrc", "notes.txt") + quote
    assert detect_dangerous_command(operation.format(path=ordinary)) == (False, None, None)
    for read in ("cat {path}", "cp {path} ordinary-backup", "sed -n '1p' {path}"):
        assert detect_dangerous_command(read.format(path=operand)) == (False, None, None)


@pytest.mark.parametrize("literal", [
    "key$literal", "key?literal", "key*literal", "key[literal]", "key{literal}", "key`literal`",
])
@pytest.mark.parametrize("quoting", ["single", "double", "escaped"])
@pytest.mark.parametrize("operation", ["echo placeholder > {path}", "cp input {path}", "sed -i 's/a/b/' {path}"])
def test_forward_drive_literal_families_keep_operation_boundary(literal, quoting, operation, monkeypatch):
    """The Windows SSH fallback must not replace the known HOME write boundary."""
    monkeypatch.setenv("HOME", "C:/Users/quality-user")

    def render(path):
        if quoting == "single":
            return "'" + path + "'"
        if quoting == "double":
            return '"' + path.replace("$", "\\$").replace("`", "\\`") + '"'
        return "".join("\\" + ch if ch in "$`*?[]{}" else ch for ch in path)

    path = render("C:/Users/quality-user/.ssh/" + literal)
    control = detect_dangerous_command(operation.format(path=render("~/.ssh/" + literal)))
    assert control[0] and control[1] is not None
    assert detect_dangerous_command(operation.format(path=path)) == control
    assert detect_dangerous_command(operation.format(path=render("C:/Users/quality-user/notes/" + literal))) == (False, None, None)
    for read in ("cat {path}", "cp {path} ordinary-backup", "sed -n '1p' {path}"):
        assert detect_dangerous_command(read.format(path=path)) == (False, None, None)
