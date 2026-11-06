"""Regression: a regex engine over a spillover file must be refused before spawn.

The recorded incident command ran GNU grep with a
``.\\{0,26000\\}`` interval over a ~116 KB one-line spillover file; grep grew past 5 GB and
outlived the 180s terminal timeout. These tests never start grep and never allocate the
quantifier's state: every case is decided statically, and the huge single-line fixture is only
ever read through the bounded read_file path.
"""

import json

import pytest

from tools import terminal_tool as tt
from tools.spillover_guard import (
    MAX_INTERVAL_BOUND,
    SPILLOVER_READ_TIMEOUT_S,
    bounded_spillover_timeout,
    execute_code_spillover_block,
    references_spillover,
    spillover_background_block,
    spillover_regex_block,
    spillover_search_block,
)

SPILL = "C:/Users/alice/.hermes/profiles/alpha/cache/spillover/call_61dc7706a9d84b30a650c744.txt"

# Verbatim from the recorded terminal call (ops/so298/original-call.json).
RECORDED_COMMAND = (
    "grep -o '\"name\":\"Lola De Valence\".\\{0,26000\\}' '" + SPILL + "' "
    "| grep -o '\"odds\":\\[.\\{0,900\\}' | head -c 1400; echo; "
    "grep -o 'Lola De Valence.\\{0,26000\\}' '" + SPILL + "' "
    "| grep -o '\"grade\":\"[^\"]*\"' | sort | uniq -c"
)


def test_recorded_command_is_blocked():
    msg = spillover_regex_block(RECORDED_COMMAND)
    assert msg and msg.startswith("Blocked")
    assert "read_file" in msg and "head -c" in msg  # points at the bounded alternatives


@pytest.mark.parametrize("command", [
    # regex tools over the file, no interval at all
    f"grep -o 'Lola.*' {SPILL}",
    f"grep -E 'a+b' {SPILL}",
    f"grep -P 'x(?=y)' {SPILL}",
    f"rg 'Lola' {SPILL}",
    f"sed -n 's/a/b/p' {SPILL}",
    f"awk '/Lola/' {SPILL}",
    f"perl -ne 'print if /Lola/' {SPILL}",
    f"cat {SPILL} | grep -o 'Lola.*'",
    f"grep Lola < {SPILL}",
    # path spellings
    "grep Lola C:\\Users\\alice\\.hermes\\profiles\\alpha\\cache\\spillover\\call_61dc7706a9d84b30a650c744.txt",
    "grep Lola C:/Users/alice/.hermes/profiles/alpha/cache//spillover/./call_61dc7706a9d84b30a650c744.txt",
    "grep Lola /c/Users/alice/.hermes/profiles/alpha/cache/'spill'over/call_61dc7706a9d84b30a650c744.txt",
    'grep Lola "$HOME/.hermes/profiles/alpha/cache/spillover/call_61dc7706a9d84b30a650c744.txt"',
    "grep Lola ~/.hermes/profiles/*/cache/spillover/*.txt",
    "grep Lola ~/.hermes/profiles/alpha/cache/spill*/*",
    "cd ~/.hermes/profiles/alpha/cache/spillover && grep Lola call_61dc7706a9d84b30a650c744.txt",
    "grep Lola /tmp/hermes-results/call_61dc7706a9d84b30a650c744",
    # wrappers
    f"bash -c \"grep -o 'Lola.*' {SPILL}\"",
    f"sh -c 'grep Lola {SPILL}'",
    f"eval \"grep Lola {SPILL}\"",
    f"C:/Program Files/Git/usr/bin/grep.exe Lola {SPILL}",
    f"/usr/bin/grep Lola {SPILL}",
    f"xargs grep Lola < {SPILL}",
    # regex-mode flag overrides a fixed-string flag
    f"grep -F -E 'a+' {SPILL}",
    # nested commands: $(), backticks, () subshell, <() process substitution (round-1 review)
    f"x=$(grep -o 'Lola.*' {SPILL}); echo $x",
    f"echo `grep -o 'Lola.*' {SPILL}`",
    f"(grep -o 'Lola.*' {SPILL})",
    f"cat <(grep -o 'Lola.*' {SPILL})",
])
def test_regex_over_spillover_blocked(command):
    assert spillover_regex_block(command), command


@pytest.mark.parametrize("command", [
    "grep -o 'a.\\{0,26000\\}' src/app.py",
    "grep -oE 'a.{0,26000}' src/app.py",
    "sed -E 's/a.{5000}//' src/app.py",
    "echo hi | grep -E 'x{1001}'",
    "grep -o 'a.\\{1000,2000\\}' src/app.py",
    "bash -c \"grep -E 'a.{0,26000}' src/app.py\"",
    # intervals inside command substitution count too (round-1 review)
    "x=$(grep -o 'a.\\{0,26000\\}' src/app.py)",
    "echo `grep -E 'a.{0,26000}' src/app.py`",
    # every single bound under the cap, but the nested product exceeds it (round-1 review)
    "grep -oE '(.{0,999}){0,999}' src/app.py",
    "grep -oE '(a{2,500}b{2,500}){2,500}' src/app.py",
    "sed -E 's/(x.{0,800}){0,800}/y/' src/app.py",
    # BRE nested spellings: \)\\{ breaks a bare '){' match (round-2 review)
    r"grep -o '\(.\{0,999\}\)\{0,999\}' src/app.py",
    r"sed 's/\(.\{0,999\}\)\{0,999\}/y/' src/app.py",
    # double-escaped form as it arrives in raw JSON tool-call text
    "grep -o '\\\\(\\\\.\\\\{0,999\\\\}\\\\)\\\\{0,999\\\\}' src/app.py",
])
def test_oversized_interval_blocked_anywhere(command):
    msg = spillover_regex_block(command)
    assert msg and str(MAX_INTERVAL_BOUND) in msg, command


@pytest.mark.parametrize("command", [
    # ordinary code search is untouched
    "grep -rn 'def foo' src/",
    "grep -E 'a.{0,50}b' src/app.py",
    "grep -o 'x.\\{0,900\\}' src/app.py",
    # nested but bounded products stay legal (BRE spelling included)
    "grep -oE '(.{0,30}){0,31}' src/app.py",
    "grep -oE '(a{2,20}b{2,20}){2,20}' src/app.py",
    r"grep -o '\(.\{0,30\}\)\{0,31\}' src/app.py",
    "rg 'TODO' tools/",
    "sed -n '1,20p' src/app.py",
    "awk '{print $1}' data.csv",
    "git log --oneline | grep -E 'fix(es)?'",
    "ls ~/.hermes/profiles/alpha/cache/",
    "ls ~/.hermes/profiles/alpha/cache/spillover",
    "cd ~/.hermes/profiles/alpha/cache/spillover && ls -la",
    # bounded / literal reads of the spillover file are allowed
    f"head -c 20000 {SPILL}",
    f"tail -c +50000 {SPILL} | head -c 20000",
    f"wc -c {SPILL}",
    f"grep -F -b -o -m 1 -- 'Lola De Valence' {SPILL} | head -c 2000",
    f"grep --fixed-strings -c Lola {SPILL}",
    f"rg -F 'Lola' {SPILL}",
    f"rg --fixed-strings 'Lola' {SPILL}",
    f"fgrep Lola {SPILL}",
    f"python scripts/slice.py {SPILL} 0 20000",
    # a pipeline whose regex stage never sees the spillover file's path is judged by that stage
    "echo \"call_61dc7706a9d84b30a650c744.txt\" | wc -c",
])
def test_ordinary_and_bounded_commands_pass(command):
    assert spillover_regex_block(command) is None, command


def test_pipe_inside_quoted_regex_does_not_split_stage():
    assert spillover_regex_block(f"grep -F 'a|b' {SPILL}") is None
    assert spillover_regex_block(f"grep 'a|b' {SPILL}")


def test_non_string_and_empty_are_ignored():
    assert spillover_regex_block("") is None
    assert spillover_regex_block(None) is None  # type: ignore[arg-type]
    assert not references_spillover(None)  # type: ignore[arg-type]


def test_references_spillover_other_profile_and_default_home():
    assert references_spillover("/home/u/.hermes/cache/spillover/call_aaaaaaaaaaaaaaaa.txt")
    assert references_spillover("C:\\Users\\a\\hermes\\profiles\\coder\\cache\\spillover\\x.txt")
    assert not references_spillover("/home/u/project/spillover_notes.md")
    assert not references_spillover("grep foo src/main.py")


# --- timeout clamp ---------------------------------------------------------------------

def test_timeout_clamped_for_spillover_commands_only():
    assert bounded_spillover_timeout(f"wc -c {SPILL}", None) == SPILLOVER_READ_TIMEOUT_S
    assert bounded_spillover_timeout(f"wc -c {SPILL}", 600) == SPILLOVER_READ_TIMEOUT_S
    assert bounded_spillover_timeout(f"wc -c {SPILL}", 5) == 5
    assert bounded_spillover_timeout("make test", 600) == 600
    assert bounded_spillover_timeout("make test", None) is None


def test_plan_execution_clamps_and_never_promotes_spillover_reads(monkeypatch):
    monkeypatch.setenv("TERMINAL_ENV", "local")
    plan = tt._plan_execution(f"wc -c {SPILL}", task_id=None, timeout=900, background=False, _host_local=False)
    assert plan.effective_timeout == SPILLOVER_READ_TIMEOUT_S
    assert plan.promoted_from_foreground_timeout is None
    plan = tt._plan_execution("make test", task_id=None, timeout=900, background=False, _host_local=False)
    assert plan.promoted_from_foreground_timeout == 900


# --- terminal_tool boundary: refused before any environment or subprocess exists ------------

def test_terminal_tool_refuses_recorded_command_before_spawn(monkeypatch):
    def boom(*a, **k):  # any spawn / environment acquisition is a test failure
        raise AssertionError("subprocess or environment reached for a blocked command")

    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.setattr(tt, "_acquire_env", lambda plan, task_id: object())
    monkeypatch.setattr(tt, "_run_foreground", boom)
    monkeypatch.setattr(tt, "spawn_background_process", boom)
    monkeypatch.setattr("subprocess.Popen", boom)
    monkeypatch.setattr("tools.process_registry._is_supervised_gateway_process", lambda: False)
    out = json.loads(tt.terminal_tool(RECORDED_COMMAND, timeout=180))
    assert out["status"] == "blocked"
    assert "read_file" in out["error"]


def test_terminal_tool_refuses_when_backgrounded_too(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("background spawn reached for a blocked command")

    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.setattr(tt, "_acquire_env", lambda plan, task_id: object())
    monkeypatch.setattr(tt, "_run_foreground", boom)
    monkeypatch.setattr(tt, "spawn_background_process", boom)
    monkeypatch.setattr("tools.process_registry._is_supervised_gateway_process", lambda: False)
    out = json.loads(tt.terminal_tool(RECORDED_COMMAND, background=True))
    assert out["status"] == "blocked"


# --- background path: no deadline applies, so a spillover command is refused outright --------

def test_background_spillover_command_refused_before_spawn():
    msg = spillover_background_block(f"wc -c {SPILL}", True)
    assert msg and msg.startswith("Blocked")
    assert "foreground" in msg and "read_file" in msg


def test_background_block_scopes_to_spillover_only():
    assert spillover_background_block(f"wc -c {SPILL}", False) is None
    assert spillover_background_block("sleep 300 && grep -rn foo src/", True) is None
    assert spillover_background_block("", True) is None
    assert spillover_background_block(None, True) is None  # type: ignore[arg-type]


def test_terminal_tool_refuses_background_spillover_before_any_env(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("environment or spawn reached for a blocked background command")

    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.setattr(tt, "_acquire_env", boom)
    monkeypatch.setattr(tt, "_run_foreground", boom)
    monkeypatch.setattr(tt, "spawn_background_process", boom)
    out = json.loads(tt.terminal_tool(f"tail -c +1 {SPILL} | head -c 200", background=True))
    assert out["status"] == "blocked"
    assert "foreground" in out["error"]


def test_terminal_tool_allows_background_for_ordinary_commands(monkeypatch):
    """A plain long build stays a legal background process; only spillover reads are refused."""
    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.setattr(tt, "_acquire_env", lambda plan, task_id: object())
    monkeypatch.setattr(
        tt, "spawn_background_process",
        lambda **k: json.dumps({"session_id": "s1", "pid": 1, "output": "", "exit_code": None}),
    )
    monkeypatch.setattr("tools.process_registry._is_supervised_gateway_process", lambda: False)
    out = json.loads(tt.terminal_tool("sleep 300", background=True))
    assert out.get("session_id") == "s1"


# --- execute_code route --------------------------------------------------------------------

@pytest.mark.parametrize("code", [
    f"import subprocess\nsubprocess.run(['grep', '-o', 'Lola.*', '{SPILL}'])",
    f"import os\nos.system(\"grep -o 'Lola.\\\\{{0,26000\\\\}}' {SPILL}\")",
    f"import subprocess\nsubprocess.check_output('grep -E \"a+\" {SPILL}', shell=True)",
    "import subprocess\nsubprocess.run(['grep', '-E', 'a.{0,26000}', 'src/app.py'])",
    # path and tool in separate literals
    f"import subprocess\np = '{SPILL}'\nsubprocess.run(['grep', '-o', 'Lola.*', p])",
    f"import subprocess\np = '{SPILL}'\ncmd = 'g' + 'rep'\nsubprocess.run([cmd, 'x', p]); x = 'grep'",
])
def test_execute_code_regex_over_spillover_blocked(code):
    assert execute_code_spillover_block(code), code


@pytest.mark.parametrize("code", [
    f"from hermes_tools import read_file\nprint(read_file('{SPILL}', offset=1, limit=50))",
    f"data = open('{SPILL}', 'rb').read(20000)\nprint(len(data))",
    f"import json\ntext = open('{SPILL}', encoding='utf-8').read()\nprint(text.find('Lola De Valence'))",
    f"import subprocess\nsubprocess.run(['wc', '-c', '{SPILL}'])",
    "import subprocess\nsubprocess.run(['grep', '-rn', 'foo', 'src/'])",
    "print('hello')",
    "def broken(:\n",
])
def test_execute_code_bounded_reads_and_ordinary_code_pass(code):
    assert execute_code_spillover_block(code) is None, code


# --- search_files route --------------------------------------------------------------------

def test_search_block_content_over_spillover_dir_or_glob():
    d = "C:/Users/alice/.hermes/profiles/alpha/cache/spillover"
    assert spillover_search_block("Lola", d, None, "content")
    assert spillover_search_block("Lola", SPILL, None, "content")
    assert spillover_search_block("Lola", "C:\\Users\\alice\\.hermes\\profiles\\alpha\\cache\\spillover", "*.txt", "content")
    assert spillover_search_block("Lola", "C:/Users/alice/.hermes/profiles/alpha/cache", "spillover/*.txt", "content")
    # finding files by name is bounded, and other searches are untouched
    assert spillover_search_block("*.txt", d, None, "files") is None
    assert spillover_search_block("def foo", "src", "*.py", "content") is None


def test_search_tool_refuses_spillover_dir_and_leaves_normal_search(tmp_path, monkeypatch):
    from tools import file_tools

    out = json.loads(file_tools.search_tool("Lola", path="C:/x/profiles/p/cache/spillover", task_id="t-spill"))
    assert "Blocked" in out["error"]

    (tmp_path / "a.py").write_text("def foo():\n    pass\n", encoding="utf-8")
    ok = json.loads(file_tools.search_tool("def foo", path=str(tmp_path), task_id="t-so298-ok"))
    assert "error" not in ok or not ok["error"]


# --- huge single-line fixture: bounded read stays bounded --------------------------------------

def test_huge_single_line_spillover_read_is_line_clamped(tmp_path, monkeypatch):
    """The 116 KB one-line payload comes back clamped by read_file, not as 116 KB."""
    from tools import file_tools

    f = tmp_path / "cache" / "spillover" / "call_61dc7706a9d84b30a650c744.txt"
    f.parent.mkdir(parents=True)
    line = '{"result":"' + ("\\\\n" + "x" * 400) * 300 + '"}'  # ~120 KB, one line
    f.write_text(line, encoding="utf-8")
    assert len(line) > 100_000 and "\n" not in line
    out = json.loads(file_tools.read_file_tool(str(f), offset=1, limit=5, task_id="t-so298-read"))
    content = out.get("content", "")
    assert content, out
    assert len(content) < 20_000
