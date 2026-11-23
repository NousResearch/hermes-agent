"""Matchers that stand in for approval rules whose stdlib spelling holds lookarounds match exactly where the rule does.

Each table entry keeps the stdlib pattern; DANGEROUS_PATTERNS_COMPILED swaps in a matcher (RE2 has no lookaround).
The rule compiled with stdlib ``re`` is the reference on a corpus built around each lookaround.
"""
import itertools
import random
import re
import time

import pytest

from tools.approval_detection import _CMD_PATH, _CMDPOS, DANGEROUS_PATTERNS, DANGEROUS_PATTERNS_COMPILED, HARDLINE_PATTERNS_COMPILED
from tools.approval_fetch_exec import _ps_reads
from tools.approval_hermes_store import home_create_target_matcher


def _rule(description):
    original = re.compile(next(p for p, d in DANGEROUS_PATTERNS if d == description), re.IGNORECASE | re.DOTALL)
    return original, next(p for p, d in DANGEROUS_PATTERNS_COMPILED if d == description)


def _sample(parts, size, seed):
    cases = ["".join(c) for c in itertools.product(*parts)]
    return random.Random(seed).sample(cases, min(size, len(cases)))


_BLANKS = [" ", "  ", "\t", "\n", " \n "]

_CREATE = "create a file in a sensitive credential/SSH/shell-rc path"
_CREATE_WORDS = ["", "-r ", "--reference ", "-R ", "-- ", "-- -r ", "--  -r ", "x -- --reference ", "-x ", "a ",
                 "--referencex ", "-rr ", "x-r ", '"-r" ', "a\n-r ", "-r  ", "-r\t", "-p -r ", "-- -- -r ", "a\n"]
_CREATE_TARGETS = ["~/.ssh/x", "'~/.bashrc'", "$home/.netrc", "~/.bashrc.bak", "~/.ssh", "${home}/.hermes/.env", "x"]


def _create_cases():
    return _sample([["", "sudo ", "x; ", "echo `", "$(", "\n  ", "a\n", "touch a;", "touch a\n", "touch a\nb "], ["touch", "mkdir", "touchx"],
                    _BLANKS, _CREATE_WORDS, _CREATE_WORDS, _CREATE_TARGETS, ["", ";", " | y", "\nz", ")", " x"]],
                   40_000, 1)


_RM = "recursive delete (flags after operands)"
_RM_WORDS = ["a", "build/", "--", "-f", "-rf", "-R", "--recursive", "--recursivex", "-x-r", '"q"', "a;b", "a|", "&",
             "--x", "-", "x--", "rm"]


def _rm_cases():
    return _sample([["", "x ", "xrm ", "sudo ", "a;"], ["rm", "RM", "rm2"], _BLANKS, _RM_WORDS, _BLANKS, _RM_WORDS,
                    _BLANKS, _RM_WORDS, ["", " ", "\n", "x"]], 40_000, 2)


_LAUNCHCTL = "stop/restart hermes launchd service (kills running agents)"
_LAUNCHCTL_WORDS = ["launchctl stop", "launchctl  bootout", "launchctl\nkill", "launchctl stopx", "xlaunchctl stop",
                    "launchctl list", "hermes", "ai.hermes", "hermesx", "_hermes", "gui/501/ai.hermes.gateway", "x"]


def _launchctl_cases():
    return _sample([_LAUNCHCTL_WORDS, ["", " ", "\n", "/"], _LAUNCHCTL_WORDS, [" ", ";"], _LAUNCHCTL_WORDS], 5_000, 3)


_LINK = "link a file into a sensitive credential/SSH/shell-rc path"
_LINK_TARGETS = ["~/.ssh/x", "'~/.bashrc'", '"$home/.netrc', "~/.bashrc.bak", "~/.ssh", "${home}/.hermes/.env", "x",
                 "~/.ssh\n", "~/.zshrc'x"]
_LINK_TAILS = ["", ">x", " 2>/dev/null", " >&2", " > ", ">", " 2>&1 >log", "<in", " 1", "\n", " \n", "\nx", ";x", " &",
               ")", "`", " | y", " x", "'", "\"", " >x y", ">>a>b", " 2>", "&>x", " >&", " # c", " >&2 x",
               " 2>&1 y"]


def _link_cases():
    return _sample([["", "sudo ", "x; ", "$(", "a\n", "ln a;"], ["ln", "LN", "lnx"], _BLANKS,
                    ["", "-s ", "-sf a ", "a\n", "a;"], ["", *_BLANKS], _LINK_TARGETS, _LINK_TAILS, ["", " ;z", "\n"]],
                   40_000, 4)


@pytest.mark.parametrize(("description", "cases"), [
    (_CREATE, _create_cases), (_RM, _rm_cases), (_LAUNCHCTL, _launchctl_cases), (_LINK, _link_cases)])
def test_matcher_matches_exactly_where_the_stdlib_rule_does(description, cases):
    original, matcher = _rule(description)
    corpus = cases()
    differ = [c for c in corpus if bool(original.search(c)) != bool(matcher.search(c))]
    assert differ == []
    assert any(original.search(c) for c in corpus) and not all(original.search(c) for c in corpus)


@pytest.mark.parametrize(("description", "command", "expected"), [
    (_CREATE, "touch -r ~/.bashrc /tmp/x", False),
    (_CREATE, "touch -- -r ~/.bashrc", True),
    (_CREATE, "touch -r a ~/.ssh/x", True),
    (_CREATE, "touch a;touch -r ~/.ssh/x", False),
    (_RM, "rm build/ -rf", True),
    (_RM, "rm -- build -rf", False),
    (_RM, "rm build -- -rf", False),
    (_RM, "rm build | grep -r x", False),
    (_LAUNCHCTL, 'for l in ai.hermes.gateway; do launchctl bootout "$l"; done', True),
    (_LAUNCHCTL, "launchctl list | grep hermes", False),
    (_LINK, "ln -sf evil ~/.bashrc 2>/dev/null", True),
    (_LINK, "ln -s ~/.bashrc /tmp/x", False),
    (_LINK, "ln -s a ~/.bashrc.bak", False),
    (_LINK, "ln -s a ~/.bashrc\necho", False),
])
def test_known_commands(description, command, expected):
    original, matcher = _rule(description)
    assert bool(original.search(command)) is expected
    assert bool(matcher.search(command)) is expected


def test_create_target_search_resumes_inside_a_dropped_command():
    # A prefix whose longest parse reaches a later command word: the search must go on one character after the
    # dropped start, where the earlier command word and its target are reached, as the stdlib rule does.
    prefix = r"(?:a[^\n]*)?"
    rule = prefix + next(p for p, d in DANGEROUS_PATTERNS if d == _CREATE).removeprefix(_CMDPOS)
    command = "atouch ~/.ssh/x touch -r b"
    assert re.search(rule, command, re.IGNORECASE | re.DOTALL)
    assert home_create_target_matcher(prefix).search(command)


# The stdlib drive-root rm rule (hardline rules have no approval key, so the table holds the matcher itself).
_STDLIB_COLLAPSE_SEG = r'(?:(?:\.\.?)?/)*+(?:\.\.?)?'
_STDLIB_COLLAPSE_TAIL = (rf'(?:(?:\.\.?)?/|\.\.?(?=["\']/)|"{_STDLIB_COLLAPSE_SEG}"|\'{_STDLIB_COLLAPSE_SEG}\')*+'
                         r'(?:\.\.?)?\**+(?=\s|$|[)`;|&])')
_STDLIB_PS_INCLUDE_NARROWS = (
    r'(?!(?!(?:[^\r\n;&|]*\s)?(?:/s\b|--(?:\s|$)))(?:[^\r\n;&|#"\'`)]|"[^"\n]*"|\'[^\'\n]*\')*[^\S\r\n]-inc[a-z]*(?::|\s+)'
    r'["\']?(?=[^\s,;&|"\'$`(){}@]*[^\s,;&|"\'$`(){}@*?.])[^\s,;&|"\'$`(){}@]+["\']?(?!\s*,)(?:[\s;&|)]|$))')
_STDLIB_DRIVE_PATH = r'[a-z]:/(?:(?:\.\.?)?/)*(?:\.\.?)?\**|[a-z]:/ \*|[a-z]:(?=["\'][\"\']?/)'
_STDLIB_DRIVE_ROOT_RM = (_CMDPOS + r'rm\s+' + _STDLIB_PS_INCLUDE_NARROWS + r'(-[^\s]*\s+)*'
                         + rf'(?:["\'](?:{_STDLIB_DRIVE_PATH})["\']{_STDLIB_COLLAPSE_TAIL}|(?:{_STDLIB_DRIVE_PATH})(?:\s|$|[)`;|&]))')
_INCLUDES = ["", " -Include *.tmp", " -include:x.log", " -Include *", " -Include a,b", " -Include '*.tmp'", " -inc \"$x\"",
             " /s -Include x", " -- -Include x", " # -Include x", " \"a b\" -Include x", " ) -Include x", "\n-Include x",
             " -Include x ,y", " -Include x;", " -Includex y"]
_DRIVES = ["C:/", "c:/*", '"C:/"', "'C:'/", '"C:"\'/\'', '"C:/"./', "C:/x", "C:", '"C:"/x', "d:/ *", '"c:".\'/\'', "C:/.."]


def _drive_root_cases():
    return _sample([["", "sudo ", "x; ", "$(", "rm a\n"], ["rm", "RM", "farm", "./rm"], [" ", "  ", " \n "],
                    ["", "-rf ", "-r -f ", "-Include x "], _DRIVES, _INCLUDES, ["", " y", ")", ";z"]], 40_000, 5)


def test_drive_root_rm_matches_exactly_where_the_stdlib_rule_does():
    original = re.compile(_STDLIB_DRIVE_ROOT_RM, re.IGNORECASE | re.DOTALL)
    rules = [r for r, d, _ in HARDLINE_PATTERNS_COMPILED if d == "recursive delete of drive root"]
    matcher = rules[0]  # the POSIX-path rule; the others are the Windows-path twins
    corpus = _drive_root_cases()
    differ = [c for c in corpus if bool(original.search(c)) != bool(matcher.search(c))]
    assert differ == []
    assert any(original.search(c) for c in corpus) and not all(original.search(c) for c in corpus)


def test_link_destination_work_stays_linear_on_a_long_redirect_run():
    # Every `~/.ssh/"1` word is a candidate target whose redirect run reaches the end of the line; each candidate
    # starts one character off the previous run's redirect boundaries. Walking every run again is quadratic
    # (minutes at this size); the remembered boundaries keep it to one walk.
    matcher = next(p for p, d in DANGEROUS_PATTERNS_COMPILED if d == _LINK)
    command = "ln a " + '~/.ssh/"1> ' * 10_000 + "x y"
    started = time.process_time()
    assert matcher.search(command) is None
    assert time.process_time() - started < 2


# The stdlib reads of the download-then-run PowerShell check (approval_fetch_exec._ps_reads stands in for it).
_STDLIB_PS_READ = r"\b(?:gc|get-content|cat|type)(?:[^\S\n]++-\w++)*+[^\S\n]++['\"]?(?P<{}>[^\s;&|<>'\"`()]++)"
_STDLIB_PS_READ_RUN = re.compile(
    rf"\b(?:iex|invoke-expression)\b(?:[^\S\n]++-command\b)?(?:[($'\"]|[^\S\n])*+{_STDLIB_PS_READ.format('iex')}"
    rf"|\[scriptblock\]::create\((?:[($'\"]|[^\S\n])*+{_STDLIB_PS_READ.format('block')}"
    rf"|{_STDLIB_PS_READ.format('piped')}['\"]?(?:[^\S\n]++-\w++)*+[^\S\n]*+\|[^\S\n]*+(?:iex|invoke-expression)\b"
    rf"|(?<=[\"'{{;(])[^\S\n]*+[.&][^\S\n]++['\"]?(?P<sourced>[^\s;&|<>'\"`()]++)",
    re.IGNORECASE,
)


def _ps_read_cases():
    return _sample([["", "x; ", "iex ", "iex -Command ", "iex (", "Invoke-Expression (\"", "[scriptblock]::Create((", "[scriptblock]::Create(", "iexx ", "iex\u00e9 (", "iex -commandx ",
                     "gc a | ", "\"", "{", "("],
                    ["gc", "Get-Content", "cat", "type", "gcx", "-cat", "", " . ", "& "],
                    ["", " -Raw", " -Raw -Encoding", " -", " -a-b", " --x", " -cat", "\t-Raw"],
                    [" ", "  ", "", "\n"], ["", "\"", "'"], ["f", "-", "-Raw", "./a.ps1", "a;b", "-.x", ")", "gc"],
                    ["", ")", " | iex", "\" | iex", " -Raw | iex", "|iex", " | Invoke-Expression", " | iexx", " & ./f",
                     "; . ./g.ps1", " -Raw|iex x", " | gc y"]], 60_000, 6)


def _ps_reads_found(found):
    return [(read.start, read.end, read.file) for read in found]


def test_ps_reads_are_the_stdlib_reads():
    corpus = _ps_read_cases()
    for pos in (0, 3):
        expected = {c: [(m.start(), m.end(), m.group(m.lastgroup)) for m in _STDLIB_PS_READ_RUN.finditer(c, pos)]
                    for c in corpus}
        differ = [c for c in corpus if _ps_reads_found(_ps_reads(c, pos)) != expected[c]]
        assert differ == []
        assert any(expected.values()) and not all(expected.values())


def test_ps_reads_work_stays_linear_on_a_shared_flag_run():
    # Every `cat` of `-cat` is a read word whose flag run ends where the first one's does, before one long FILE
    # that no `| iex` follows: reading the run and FILE again per word is quadratic.
    command = "gc" + " -cat" * 20_000 + " " + "f" * 100_000 + " | x"
    started = time.process_time()
    assert list(_ps_reads(command, 0)) == []
    assert time.process_time() - started < 2


_NC = "reverse shell via netcat -e/-c"
_SOCAT = "reverse shell via socat EXEC:/SYSTEM:"
_NC_SPANS = ["", " ", " h 1", "&>x", "2>&1", "<&0", " &", " # c", "#x", " x#y", ";", "|", "\n", " >&2", "&", " <x", ">"]


def _reverse_shell_cases(names, flags):
    return _sample([["", "sudo ", "x; ", "$(", "a\n", "env A=1 ", "busybox "], ["", "/usr/bin/", "x/", "./a b/"], names,
                    ["", ".exe", ".", "x"], _NC_SPANS, _NC_SPANS, flags, ["", "x", " y", ";"]], 40_000, 7)


def _nc_cases():
    return _reverse_shell_cases(["nc", "ncat", "netcat", "NC", "ncx", "xnc"],
                                [" -e sh", " -c", "\n-e", ' "-e', " -E", " --exec", " --sh-exec", " -xe", " -ex", "-e", ""])


def _socat_cases():
    return _reverse_shell_cases(["socat", "SOCAT", "socatx"], ["exec:sh", " EXEC:sh", " system:x", "-x:exec:sh", " xexec:",
                                                                 "\nexec:", " ,exec:", ""])


# The exec argument in another command or line than the tool word, before it and after it: only an argument in the
# tool's own span (up to `;`, `|`, a background `&`, a comment or a newline it may still start on) counts.
_OTHER_COMMANDS = ["", "echo -e x", "echo EXEC:x", "x --exec", "nc -e sh", "socat exec:sh", "x -c y", "echo system:a"]
_COMMAND_BREAKS = ["\n", ";", " | ", " & ", " && ", "\n\n", " # ", "\n "]


def _split_reverse_shell_cases(names, flags):
    return _sample([_OTHER_COMMANDS, _COMMAND_BREAKS, ["", "sudo ", "/usr/bin/"], names, ["", " h 1", "&>x", " -z h 80"],
                    ["", *flags[:3]], ["", *_COMMAND_BREAKS], _OTHER_COMMANDS], 20_000, 10)


_MKFS, _MKFS_HL = "format filesystem", "format filesystem (mkfs)"
_SHUTDOWN = "system shutdown/reboot"
_VERB_TAILS = ["", " /dev/sda", "/x", ".ext4", ".ext4/x", ".ext_4", "x", "-x", "-x/y", ".exe", ".exe/x", ".exex", ".sh", "=1",
               ")", ">x", "<x", " | y", ";", "\n", "_", ".", "\u00e9"]


def _verb_cases(verbs):
    return _sample([["", "sudo ", "sudo -x ", "x; ", "$(", "`", "a\n", "env A=1 ", "busybox ", "exec ", "env mkfs=1 ",
                     "env A_mkfs=1 A_halt=1 "],
                    ["", "/sbin/", "./", "x/", "C:/w/", "a=b/"], verbs, _VERB_TAILS, ["", " y", ";z", "/q"]], 30_000, 8)


def _hardline(description):
    return next(r for r, d, _ in HARDLINE_PATTERNS_COMPILED if d == description)


# The stdlib command position (possessive runs): the reference for the rules whose matcher stands in for it.
_STDLIB_CMD_PATH = r'(?:/?+(?:[^\s/)=<>]++/)*+)?+'
_STDLIB_CMDPOS = (r'(?:^|\n(?![^\S\n]*+\n)|`|\$\()' r'\s*+' r'(?:sudo\s++(?:-[^\s]++\s++)*+)?'
                  r'(?:env\s++(?:\w+=\S*+\s++)*+)?' r'(?:(?:exec|nohup|setsid|time)\s++)*+' r'\s*+'
                  r'(?:' + _STDLIB_CMD_PATH + r'(?:busybox|toybox)\b\s++)?+' + _STDLIB_CMD_PATH)
_NC_RULE = (r'(?:\S*/)?(?:nc|ncat|netcat)(?:\.\w+)?(?=[\s<>]|&>)(?:[^\n;&|#]|(?<!\s)#|(?<=[<>])&|(?<![<>])&(?=>))*\s["\']?'
            r'(?:(?-i:-[a-zA-Z]*[ec])\b|--(?:sh-)?exec\b)')
_SOCAT_RULE = (r'(?:\S*/)?socat(?:\.\w+)?(?=[\s<>]|&>)(?:[^\n;&|#]|(?<!\s)#|(?<=[<>])&|(?<![<>])&(?=>))*'
               r'\b(?:exec|system):')


@pytest.mark.parametrize(("description", "rule", "cases", "table"), [
    (_NC, _NC_RULE, _nc_cases, DANGEROUS_PATTERNS),
    (_SOCAT, _SOCAT_RULE, _socat_cases, DANGEROUS_PATTERNS),
    (_NC, _NC_RULE, lambda: _split_reverse_shell_cases(["nc", "ncat"], [" -e sh", " -c", "\n-e"]), DANGEROUS_PATTERNS),
    (_SOCAT, _SOCAT_RULE, lambda: _split_reverse_shell_cases(["socat"], [" EXEC:sh", " system:x", "\nexec:"]),
     DANGEROUS_PATTERNS),
    (_MKFS, r'mkfs\b', lambda: _verb_cases(["mkfs", "MKFS", "mkfsx"]), DANGEROUS_PATTERNS),
    (_MKFS_HL, r'mkfs(\.[a-z0-9]+)?\b', lambda: _verb_cases(["mkfs", "MKFS", "mkfsx"]), None),
    (_SHUTDOWN, r'(shutdown|reboot|halt|poweroff)\b(?!-|\.(?!exe\b))',
     lambda: _verb_cases(["shutdown", "reboot", "halt", "poweroff", "HALT", "rebootx"]), None),
])
def test_gated_rule_matches_exactly_where_the_stdlib_rule_does(description, rule, cases, table):
    # Dangerous rules keep the stdlib spelling in the table (it feeds the legacy approval key); hardline rules have no
    # key, so their table holds the matcher itself.
    if table is not None:
        assert (_CMDPOS + rule, description) in table
        matcher = next(p for p, d in DANGEROUS_PATTERNS_COMPILED if d == description)
    else:
        matcher = _hardline(description)
    # Exact against the stdlib rule on the current command position with the stdlib (possessive) `_CMD_PATH`, the
    # run the gate stands in for. The fully possessive stdlib command position also keeps a sudo flag or env pair
    # that names the verb (`env mkfs=1 x`), which the rewritten `_CMDPOS` may give back: never less than it.
    exact = re.compile(_CMDPOS.replace(_CMD_PATH, _STDLIB_CMD_PATH) + rule, re.IGNORECASE | re.DOTALL)
    original = re.compile(_STDLIB_CMDPOS + rule, re.IGNORECASE | re.DOTALL)
    corpus = cases()
    assert [c for c in corpus if bool(exact.search(c)) != bool(matcher.search(c))] == []
    assert [c for c in corpus if original.search(c) and not matcher.search(c)] == []
    assert any(original.search(c) for c in corpus) and not all(original.search(c) for c in corpus)


@pytest.mark.parametrize(("description", "command"), [
    (_NC, "echo -e hello\nnc -z example.org 80"),
    (_NC, "nc -z example.org 80\necho -e hello"),
    (_SOCAT, "echo EXEC:notes\nsocat TCP-LISTEN:8000 TCP:example.org:80"),
    (_SOCAT, "socat TCP-LISTEN:8000 TCP:example.org:80; echo EXEC:notes"),
])
def test_reverse_shell_rule_ignores_an_exec_argument_of_another_command(description, command):
    original, matcher = _rule(description)
    assert original.search(command) is None
    assert matcher.search(command) is None


def test_gated_matcher_treats_private_use_text_as_the_stdlib_rule_does():
    matcher = next(p for p, d in DANGEROUS_PATTERNS_COMPILED if d == _NC)
    assert matcher.search("nc -e sh \ue000\ue001")
    assert not matcher.search("nc\ue000 -e sh")
    assert not matcher.search("echo nc -e \ue000")
    # The text's own private-use characters never stand for a mark: here one follows a command-position `nc`.
    assert not matcher.search("nc\ue000 -e x\necho nc -e y")
    exact = re.compile(_CMDPOS.replace(_CMD_PATH, _STDLIB_CMD_PATH) + _NC_RULE, re.IGNORECASE | re.DOTALL)
    for mark in ("\ue000", "\ue001"):
        for command in (f"nc{mark} -e sh", f"nc -e {mark}", f"{mark}\nnc -e sh", f"nc {mark}-e sh", f"x{mark}nc -e sh"):
            assert bool(matcher.search(command)) == bool(exact.search(command)), command


def test_gated_matcher_survives_a_text_holding_every_private_use_character():
    # Every character of the private-use planes: no mark can be one the text lacks, and the verdict must still come.
    planes = "".join(map(chr, itertools.chain(range(0xE000, 0xF900), range(0xF0000, 0xFFFFE), range(0x100000, 0x10FFFE))))
    matcher = next(p for p, d in DANGEROUS_PATTERNS_COMPILED if d == _NC)
    assert matcher.search(planes + "\nnc -e sh")
    assert not matcher.search(planes + "\nnc -x sh")


def test_reverse_shell_gate_work_stays_linear_with_many_names():
    # Every `nc` is a candidate end; each reads the next `-e` and the next span break once, by bisection.
    matcher = next(p for p, d in DANGEROUS_PATTERNS_COMPILED if d == _NC)
    command = "x" + " nc" * 100_000 + " ; y -e"
    started = time.process_time()
    assert matcher.search(command) is None
    assert time.process_time() - started < 2


# The Hermes store rules (stdlib table text; verb_matcher + argument_gate in DANGEROUS_PATTERNS_COMPILED).
_STORE_DELETE, _STORE_OVERWRITE = "delete/move/wipe Hermes home or session store", "overwrite Hermes session/credential store"
_STORE_TARGETS = ["~/.hermes/state.db", '"~/.hermes/state.db"', "~/.hermes/state.db.bak", "$HERMES_HOME/sessions/",
                  "~/.hermes/sessions/x.jsonl", "~/.hermes", "~/.hermes/", "'~/.hermes/*'", "~/.hermes/rm", "/tmp/x",
                  "of=~/.hermes/state.db", "of=/dev/sda", "x,of=~/.hermes/state.db"]


def _store_cases(verbs):
    return _sample([["", "sudo ", "x; ", "$(", "a\n", "env A=1 "], ["", "/bin/", "./a/"], verbs,
                    ["", "x", "-x", "-x/y", ".exe", ",", "=1"], [" ", "  ", "\n", " -f ", " a;b "], _STORE_TARGETS,
                    ["", " /tmp/y", " 2>/dev/null", " # c", ";", " | z", "\n", ")"]], 30_000, 9)


@pytest.mark.parametrize(("verbs", "description", "cases"), [
    ("(?:rm|unlink|shred|truncate)", _STORE_DELETE, lambda: _store_cases(["rm", "unlink", "RM", "rmx"])),
    ("mv", _STORE_DELETE, lambda: _store_cases(["mv", "mvx"])),
    ("(?:cp|install|rsync)", _STORE_OVERWRITE, lambda: _store_cases(["cp", "install", "rsync", "cpx"])),
    ("dd", _STORE_OVERWRITE, lambda: _store_cases(["dd", "ddx"])),
    ("tee", _STORE_OVERWRITE, lambda: _store_cases(["tee", "teex"])),
])
def test_store_rule_matches_exactly_where_the_stdlib_rule_does(verbs, description, cases):
    (rule, matcher), = [(p, c) for (p, d), (c, _) in zip(DANGEROUS_PATTERNS, DANGEROUS_PATTERNS_COMPILED)
                        if d == description and p.startswith(_CMDPOS + verbs + r"\b")]
    exact = re.compile(_CMDPOS.replace(_CMD_PATH, _STDLIB_CMD_PATH) + rule.removeprefix(_CMDPOS), re.IGNORECASE | re.DOTALL)
    original = re.compile(_STDLIB_CMDPOS + rule.removeprefix(_CMDPOS), re.IGNORECASE | re.DOTALL)
    corpus = cases()
    assert [c for c in corpus if bool(exact.search(c)) != bool(matcher.search(c))] == []
    assert [c for c in corpus if original.search(c) and not matcher.search(c)] == []
    assert any(original.search(c) for c in corpus) and not all(original.search(c) for c in corpus)


def test_hardline_dd_matches_exactly_where_the_stdlib_rule_does():
    rule = r'dd\b[^\n]*\bof=/dev/(sd|nvme|hd|mmcblk|vd|xvd)[a-z0-9]*'
    exact = re.compile(_CMDPOS.replace(_CMD_PATH, _STDLIB_CMD_PATH) + rule, re.IGNORECASE | re.DOTALL)
    original = re.compile(_STDLIB_CMDPOS + rule, re.IGNORECASE | re.DOTALL)
    matcher = _hardline("dd to raw block device")
    corpus = _store_cases(["dd", "DD", "ddx"]) + [c.replace(";", " ; ") for c in _store_cases(["dd"])]
    assert [c for c in corpus if bool(exact.search(c)) != bool(matcher.search(c))] == []
    assert [c for c in corpus if original.search(c) and not matcher.search(c)] == []
    assert any(original.search(c) for c in corpus) and not all(original.search(c) for c in corpus)


def test_argument_gate_work_stays_linear_with_many_verbs():
    matcher = next(c for (p, d), (c, _) in zip(DANGEROUS_PATTERNS, DANGEROUS_PATTERNS_COMPILED)
                   if d == _STORE_DELETE and p.startswith(_CMDPOS + "(?:rm"))
    command = "rm" + " rm" * 100_000 + " x; y ~/.hermes/state.db"
    started = time.process_time()
    assert matcher.search(command) is None
    assert time.process_time() - started < 2
