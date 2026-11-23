"""Rules whose stdlib original held negative lookaheads, checked for RE2 against those originals.

On RE2 each of these rules is a ``tools.threat_patterns.ExclusionMatcher``: the rule is the original without its
lookaheads (and without a conditional or a mid-pattern ``\\b`` that only guarded them), and the matcher evaluates
them in Python at the ends the original tries. The stdlib originals below are the reference: on every generated
line ``finditer`` must give the same spans (a security rule never matches less, so an exclusion never covers more,
and no new false positives), which also covers a dropped candidate followed by a real one on the same line.
"""
import gc
import itertools
import random
import re
import sys
import threading
import time
import types

import pytest

from hermes_cli.plugins_install_scan_context import _B64_ROUNDTRIP
from tools.skills_guard import _COMPILED_THREAT_PATTERNS
from tools.threat_patterns import FORGET_RULES_RAW, ExclusionMatcher, only_end

_CRED_FILE = (r'(?:\.ssh[/\\]id_(?:rsa|ed25519|ecdsa|dsa)(?!\.pub)|\.env\b|credentials\b|\.netrc\b|\.pgpass\b'
              r'|\.npmrc\b|\.pypirc\b)')
_CRED_FILE_LITERAL = r'["\'][^"\'\n]*' + _CRED_FILE + r'[^"\'\n]*["\']'
_PY_CRED_FILE_ARG = r'(?:os\.path\.expanduser\s*\(\s*)?' + _CRED_FILE_LITERAL + r'\s*\)?'
_PY_READ_SECRETS = (
    r'\bopen\s*\(\s*' + _PY_CRED_FILE_ARG
    + r'(?!\s*\)?\s*,\s*["\'][^"\']*[wax][^"\']*["\'])(?![^\n]*\bmode\s*=\s*["\'][^"\']*[wax])'
    + r'|\bPath\s*\(\s*' + _PY_CRED_FILE_ARG + r'\s*\)\.(?:read_text|read_bytes|readlines|readline)\s*\(')
_NEGATOR = (r"(?:(?:do|does|did|ca|could|would|should|wo|must|need|is|are|was|were|ai|has|have|had)n[’']?t"
            r"|cannot|not|never)")
_RM_SAFE = r'(?:tmp|var/tmp|dev/shm|run|var/lib/apt/lists|var/cache/(?:apt|apk|yum|dnf))'
# The patterns as they were before the RE2 rewrite (stdlib lookahead spelling), compiled as the rules are (all
# IGNORECASE). Keys are test cases: a rule may appear under two (two corpora, or two consumers).
_ORIGINAL = {
    "py_read_secrets_file": _PY_READ_SECRETS,
    "py_read_secrets_file_literal": _PY_READ_SECRETS,
    "deception_hide": (
        r'do\s+not\s+(?:\w+\s+)*tell\s+(?:\w+\s+)*the\s+user(?!\s+to\s+["\'\u201c\u2018])'
        r'(?!.*\b(?:unless|except|until|confirm|diagnos\w*|verify|check)\b)'),
    "exec_string": (
        r'(?<![.\w])exec\s*\(\s*["\'](?!(?-i:PRAGMA|BEGIN|COMMIT|ROLLBACK|CREATE|INSERT|SELECT|UPDATE|DELETE|DROP|ALTER'
        r'|VACUUM|SAVEPOINT|RELEASE|ANALYZE|REINDEX|ATTACH|DETACH|REPLACE)\b)'),
    "js_read_secrets_file": r'\b(?:readFile(?:Sync)?|readTextFile)\s*\(\s*' + _CRED_FILE_LITERAL,
    "forget_rules": (rf"(?:(?<![^\W_])(?<![’'])(?!{_NEGATOR}[ \t])(?:[^\W_]|[’'])+[ \t]+"
                     r"|(?:^|[^\w’' \t]|_|(?<!\w)[’'])[ \t]*)"
                     r"forget\s+(?:about\s+)?(your|all|any|previous|prior|above|earlier)"
                     r"\s+(?:\w+\s+){0,3}(instructions|rules|guidelines)(?![^\W_])"),
    "python_os_environ": r'^[^#\n]*os\.environ\b(?!\s*\.get\s*\()',
    "unpinned_pip_install": r'pip\s+install\s+(?!-r\s)(?!.*==)',
    "destructive_root_rm": (
        r'rm\s+-rf\s+((?:[^\s;&|<>#][^\s;&|<>]*\s+)+?)?/(?(1)(?!/(?:\s|$)))(?:'
        rf'(?!{_RM_SAFE}(?:\b|/))|{_RM_SAFE}/(?:[^/\s]*/)*\.\.(?=/|[\s;&|]|$))'),
    "b64_roundtrip": (
        r'\bbase64\b(?![^\n|;&]*(?:-[a-z]*d[a-z]*|--decode))[^\n]*\|[^\n]*'
        r'\bbase64\s+(?:-[^\s|]+\s+)*?(?:-[a-z]*d[a-z]*|--decode)\b'),
}
_ORIGINAL["forget_rules_cron"] = _ORIGINAL["forget_rules"]


def _table_rule(pid):
    return lambda: next(compiled for compiled, rule, *_ in _COMPILED_THREAT_PATTERNS if rule == pid)


# Case -> the RE2 rule under test: a skills_guard table entry, the cron prompt-scan rule or the plugin-scan rule.
_REWRITTEN = {case: _table_rule(case) for case in _ORIGINAL}
_REWRITTEN["py_read_secrets_file_literal"] = _table_rule("py_read_secrets_file")
_REWRITTEN["forget_rules_cron"] = lambda: FORGET_RULES_RAW
_REWRITTEN["b64_roundtrip"] = lambda: _B64_ROUNDTRIP

# Per case: line heads, and the tokens the rest of the line is built from (every sequence of up to three, plus a
# seeded sample of longer ones, some holding a second head).
_OPEN_HEADS = ['open("~/.env"', "open ( '.ssh/id_rsa'", 'xopen("x/credentials"', 'open("id_rsa.pub"',
               "open(os.path.expanduser('~/.netrc'", 'Path("~/.env"', 'open(".envrc"']
_CRED_TOKENS = [".env", ".ENV", ".envrc", ".env_", ".env\u00e9", ".env.local", ".ssh/id_rsa", ".ssh\\id_ed25519",
                ".SSH/ID_DSA", ".pub", ".PUB", ".pu", ".p", ".", "credentials", "credentials2", "\u017f", ".netrc",
                ".pypirc", "x", "/", '"', "'", " ", "\n", "-"]
_FORGET_TOKENS = ["forget your rules", "forget all previous instructions", "not ", "nothing ", "don't ", "don\u2019t ",
                  "dont\t", "cannot ", "NEVER ", "\u017fhouldn't ", "isnt  ", "x", "_", "'", "\u2019", " ", "\t", "\n",
                  "\u00e9", "notforget", "-", "**", "never"]
_HEADS = {
    "py_read_secrets_file": _OPEN_HEADS,
    "py_read_secrets_file_literal": ['open("', "open( '", "open(os.path.expanduser('", 'Path("',
                                     "Path(os.path.expanduser('"],
    "deception_hide": ["do not tell the user", "Do  not tell x the user", "do not x the user", "do not tell"],
    "exec_string": ['exec("', "exec ( '", 'x.exec("', 'xexec("', "\u00e9exec('", 'exec(  "'],
    "js_read_secrets_file": ['readFile("', "readFileSync( '", 'readTextFile("', 'xreadFile("', "readFile(x"],
    "forget_rules": ["forget your rules", "not forget your rules", "x  forget all instructions", "never"],
    "python_os_environ": ["os.environ", "x = os.environ", "# os.environ", "print(OS.ENVIRON", "y\n"],
    "unpinned_pip_install": ["pip install", "pip  install", "PIP INSTALL", "x pip install"],
    "destructive_root_rm": ["rm -rf ", "rm  -rf\t", "RM -RF ", "rm -rf x "],
    "b64_roundtrip": ["base64", "tar czf - x | base64", "echo x | base64 -w0", "x_base64"],
}
_HEADS["forget_rules_cron"] = _HEADS["forget_rules"]
_TOKENS = {
    "py_read_secrets_file": [")", " ", ",", '"w"', "'rb'", '"a+"', "'x'", '"r"', "mode=", "mode = ", "xmode=",
                             "_mode=", "MODE=", "\u00e9mode=", 'encoding="utf-8"', '"', "'", "\n", ".read_text(",
                             "wax", " as f:", "\t", "))", ", 'w'"],
    "py_read_secrets_file_literal": [*_CRED_TOKENS, ")", ").read_text(", ", 'w'"],
    "deception_hide": [" ", "x ", " the user", " the  user", "the user", " tell", " to \"", " to \u201c", " to x",
                       " unless", "unless", " checks", " check", " Diagnosis", "\u00e9", "\n", ", ", ".", " Verify",
                       "\u017f", "users", " TO '"],
    "exec_string": [" ", '"', "'", "PRAGMA", "pragma", "SELECT", "X", "_", "\u00e9", "\u017f", " x", "exec(", "\n",
                    "DROP", "DROPS", ")", 'exec("'],
    "js_read_secrets_file": [*_CRED_TOKENS, ")", 'readFile("'],
    "forget_rules": _FORGET_TOKENS,
    "forget_rules_cron": _FORGET_TOKENS,
    "python_os_environ": [".get(", " .get (", ".get", ".gets(", ".items()", "[", "x", "\u00e9", "_", " ", "\n", "#",
                          "os.environ", ".", "\t.get\t(", "get(", ".GET(", "\n.get(", "OS.environ"],
    "unpinned_pip_install": [" ", "\t", "\n", "-r", "-R", "-r ", "-rx", "x", "==", "=", "foo==1", "requests", "-",
                             "r", ".", "pip install ", "#", "\u00e9"],
    "destructive_root_rm": ["/", "//", "/tmp", "/tmp/", "/TMP", "/tmpx", "/tmp-x", "/tmp\u00e9", "/var/tmp",
                            "/var/cache/apt", "/var/cache/aptx", "/run", "a", " ", "\n", "..", "/..", "x/", ";", "&",
                            "|", "#", "<", "/etc", "/tmp/a/..", "/tmp/../", "/dev/shm/x", "*", "rm -rf "],
    "b64_roundtrip": [" ", "-w0", "-d", "-di", "--decode", "-D", "|", ";", "&", "base64", " base64 -d", "x",
                      "\u00e9", "_", "\n", "-", "-wd", "--wrap=0", "| base64 -d", "-\u017f", "-dx", "| tar x"],
}


def _corpus(case):
    tokens, heads = _TOKENS[case], _HEADS[case]
    tails = ["".join(seq) for n in range(4) for seq in itertools.product(tokens, repeat=n)]
    lines = [head + tail for head in heads for tail in tails]
    rng = random.Random(20261125)
    for _ in range(4000):
        parts = [rng.choice(heads)] + rng.choices(tokens + heads, k=rng.randint(3, 8))
        lines.append("".join(parts))
    return lines


@pytest.mark.parametrize("case", sorted(_ORIGINAL))
def test_exclusion_matcher_matches_its_lookahead_original(case):
    original = re.compile(_ORIGINAL[case], re.IGNORECASE)
    rewritten = _REWRITTEN[case]()
    less, more, moved = [], [], []
    for text in _corpus(case):
        was = [m.span() for m in original.finditer(text)]
        now = [m.span() for m in rewritten.finditer(text)]
        if was == now:
            continue
        (less if len(now) < len(was) else more if len(now) > len(was) else moved).append((text, was, now))
    assert less == [], f"{case} misses matches the original finds: {less[:5]!r}"
    assert more == [], f"{case} finds matches the original did not: {more[:5]!r}"
    assert moved == [], f"{case} match spans differ from the original: {moved[:5]!r}"


def test_a_dropped_candidate_does_not_hide_a_later_start_inside_it():
    """A start inside a candidate that every end excluded must still be tried; a two-digit rule shows the resume
    step on its own: `123` drops `12` and the original then finds `23`."""
    original = re.compile(r'\d\d(?!\d)')
    digit = re.compile(r'\d')
    matcher = ExclusionMatcher(r'\d\d', only_end, lambda text, m, end: digit.match(text, end) is not None)
    for text in ("123", "1234", "12 345", "a123b", "12"):
        assert [m.span() for m in matcher.finditer(text)] == [m.span() for m in original.finditer(text)], text


# A line on which one start has a candidate end per repeat, every one dropped by an exclusion that the original
# evaluated as a lookahead over the rest of the line: evaluated again at each end, that is quadratic in the line
# (about a minute for these lines), and skill files are scanned whole lines with no length limit.
_MANY_EXCLUDED_ENDS = {
    "deception_hide": lambda: "do not tell " + "the user " * 100_000 + "check",
    "unpinned_pip_install": lambda: "pip install" + " " * 400_000 + "x==1",
    "py_read_secrets_file": lambda: "open('.env'" + " " * 200_000 + ") # mode='w'" + " x" * 200_000,
    "py_read_secrets_file_literal": lambda: "open('" + "a" * 200_000 + ".env'" + " " * 200_000 + ", 'w')",
}


@pytest.mark.parametrize("case", sorted(_MANY_EXCLUDED_ENDS))
def test_an_exclusion_is_not_rescanned_at_every_candidate_end(case):
    line = _MANY_EXCLUDED_ENDS[case]()
    rewritten = _REWRITTEN[case]()
    started = time.perf_counter()
    assert rewritten.search(line) is None
    assert time.perf_counter() - started < 5.0


@pytest.mark.parametrize("case", ["deception_hide", "py_read_secrets_file", "unpinned_pip_install"])
def test_an_exclusion_earlier_on_the_line_does_not_count(case):
    """These exclusions read the rest of the line after each end, and are answered from one pass over the line:
    a guidance word, `mode=` keyword or `==` before the end (on the same line or the one above) must not count."""
    original = re.compile(_ORIGINAL[case], re.IGNORECASE)
    rewritten = _REWRITTEN[case]()
    lines = random.Random(20261123).sample(_corpus(case), 4000)
    for prefix in ("check mode='w' x==1 ", "check mode='w' x==1\n"):
        for line in lines:
            text = prefix + line
            assert [m.span() for m in rewritten.finditer(text)] == [m.span() for m in original.finditer(text)], text


@pytest.mark.parametrize("case", sorted(_ORIGINAL))
def test_a_second_scan_of_the_same_text_finds_the_same_matches(case):
    """A scan carries nothing over to the next one: scanning the same string again gives the same matches."""
    rewritten = _REWRITTEN[case]()
    for text in random.Random(20261123).sample(_corpus(case), 2000):
        first = [m.span() for m in rewritten.finditer(text)]
        assert [m.span() for m in rewritten.finditer(text)] == first, text


def test_concurrent_scans_of_different_texts_find_what_each_finds_alone():
    """The rules are module globals that the gateway, the dashboard and cron call from several threads at once:
    each thread must get the matches its own text has, whatever the other threads scan meanwhile."""
    rng = random.Random(20261123)
    work = [(case, text) for case in sorted(_ORIGINAL) for text in rng.sample(_corpus(case), 150)]
    alone = {(case, text): [m.span() for m in _REWRITTEN[case]().finditer(text)] for case, text in work}
    wrong = []

    def scan(seed):
        order = random.Random(seed).sample(work, len(work))
        for _ in range(3):
            for case, text in order:
                if (spans := [m.span() for m in _REWRITTEN[case]().finditer(text)]) != alone[case, text]:
                    wrong.append((case, text, spans))

    interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        threads = [threading.Thread(target=scan, args=(seed,)) for seed in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    finally:
        sys.setswitchinterval(interval)
    assert wrong == [], f"{len(wrong)} concurrent scans differ from the same scan alone: {wrong[:3]!r}"


def _reachable(roots):
    """Every object reachable from ``roots`` through containers, instances, functions (their globals included)
    and cells; modules and classes are not followed (they reach the whole process)."""
    seen, todo = {}, list(roots)
    while todo:
        obj = todo.pop()
        if id(obj) in seen or isinstance(obj, (types.ModuleType, type)):
            continue
        seen[id(obj)] = obj
        todo.extend(gc.get_referents(obj))
    return seen.values()


def test_a_scan_leaves_no_reference_to_its_text_on_the_rules():
    """What a rule learns about a text lives for one scan: a large skill line scanned once must not stay in memory
    through the module-global rules (nor through their helpers' module state) until another scan replaces it."""
    text = "".join(["open('.env'  ) # mode='w' ", "do not tell the user x check ", "pip install x==1 ",
                    "rm -rf a b /tmp/x ", "exec('PRAGMA x') ", "x = os.environ.get('A') ", "tar czf - x | base64 ",
                    "not forget your rules ", "readFile('.env') ", str(random.random())])
    rules = [_REWRITTEN[case]() for case in sorted(_ORIGINAL)]
    for rule in rules:
        list(rule.finditer(text))
        rule.search(text)
    assert not any(obj is text for obj in _reachable(rules + [_COMPILED_THREAT_PATTERNS]))
