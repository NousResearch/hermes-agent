"""Process-kill floor: refuse kills whose resolved match set includes Hermes' own runtime.

Everything here uses real throwaway child processes carrying unique argv markers; the guard only
ever *resolves* targets (pgrep/ps) and never signals anything.
"""

import os
import re
import subprocess
import sys
import uuid

import pytest

from tools import approval, approval_context, process_kill_guard as guard

pytestmark = pytest.mark.platforms("posix")


@pytest.fixture
def victim():
    """A 'protected' service stand-in and an unprotected bystander, both identifiable by argv."""
    token = uuid.uuid4().hex[:10]
    procs = {}
    for role in ("svc", "other"):
        marker = f"pkg-{role}-{token}"
        procs[role] = (
            subprocess.Popen(
                [sys.executable, "-c", "import sys; sys.stdin.read()", marker],
                start_new_session=True, stdin=subprocess.PIPE,
            ),
            marker,
        )
    yield procs
    for p, _ in procs.values():
        p.stdin.close()
        p.wait()


@pytest.fixture(autouse=True)
def _protect_svc(monkeypatch, victim, tmp_path):
    svc_marker = victim["svc"][1]
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        f"approvals:\n  protected_processes:\n    - '{svc_marker}'\n", encoding="utf-8",
    )


def check(cmd):
    return guard.detect_protected_process_kill(cmd)


def test_pkill_pattern_hitting_service_is_blocked(victim):
    assert check(f"pkill -f {victim['svc'][1]}")


def test_pkill_pattern_only_hitting_bystander_is_allowed(victim):
    assert check(f"pkill -f {victim['other'][1]}") is None


def test_pkill_options_after_pattern_become_patterns_on_bsd_pgrep(victim):
    # The incident: `pkill -f PAT -P 1`. Only bites where the platform stops option parsing at the
    # first operand (macOS); there the extra patterns "-P" and "1" widen the match set. Mirror the host.
    cmd = f"pkill -f {victim['other'][1]} -P 1"
    widened = subprocess.run(["pgrep", "-f", victim["other"][1], "-P", "1"], capture_output=True, text=True, check=False)
    hits = {int(x) for x in widened.stdout.split()}
    if victim["svc"][0].pid in hits:
        assert check(cmd)
    else:
        assert check(cmd) is None  # procps permutes options: -P 1 is a real filter, nothing to protect


def test_options_before_pattern_are_not_widened(victim):
    assert check(f"pkill -f -P 1 {victim['other'][1]}") is None


def test_pgrep_xargs_kill(victim):
    assert check(f"pgrep -f {victim['svc'][1]} | xargs kill")
    assert check(f"pgrep -f {victim['other'][1]} | xargs kill -9") is None


def test_kill_command_substitution_and_backticks(victim):
    assert check(f"kill $(pgrep -f {victim['svc'][1]})")
    assert check(f"kill -9 `pgrep -f {victim['svc'][1]}`")
    assert check(f"kill $(pgrep -f {victim['other'][1]})") is None


def test_variable_capture(victim):
    assert check(f"pids=$(pgrep -f {victim['svc'][1]}); kill -9 $pids")
    assert check(f"pids=$(pgrep -f {victim['other'][1]}); kill $pids") is None


def test_ps_grep_awk_pipeline(victim):
    # The argv of the agent's own shell contains the grep pattern too (classic self-match); the
    # service is in the set either way.
    assert check(f"ps -axo pid,command | grep {victim['svc'][1]} | awk '{{print $1}}' | xargs kill")


def test_plain_pid_and_pgid(victim):
    svc = victim["svc"][0].pid
    assert check(f"kill {svc}")
    assert check(f"kill -9 {svc}")
    assert check(f"kill -TERM -- {svc}")
    assert check(f"kill {victim['other'][0].pid}") is None
    assert check(f"kill -s KILL {svc}")


def test_process_group_kill_expands_members(victim):
    pgid = os.getpgid(victim["svc"][0].pid)
    assert pgid != os.getpgid(0), "fixture child must have its own process group"
    assert check(f"kill -- -{pgid}")


def test_killall_by_name(victim, monkeypatch):
    name = os.path.basename(victim["svc"][0].args[0])
    monkeypatch.setattr(guard, "_protected_regexes", lambda: [re.compile(r"pkg-svc-")])
    assert check(f"killall {name}")
    assert check(f"killall -s {name}") is None  # -s only shows what would be signalled


def test_launchd_and_own_ancestors_are_protected():
    assert check("kill 1")
    assert check("kill 0")
    assert check(f"kill {os.getppid()}")


def test_kill_minus_one_is_a_hardline_rule_for_every_signal_spelling():
    from tools.approval_detection import detect_hardline_command
    for cmd in ("kill -1", "kill -9 -1", "kill -s KILL -1", "kill -n 9 -1", "kill -- -1", "kill -SIGKILL -1"):
        assert detect_hardline_command(cmd)[0], cmd
    assert not detect_hardline_command('echo "kill -1 sends SIGHUP"')[0]
    assert not detect_hardline_command("kill -1234")[0]


def test_unresolvable_and_non_kill_commands_pass():
    assert check("echo kill the process") is None
    assert check("kill $SOME_PID") is None
    assert check("kill 99999999") is None
    assert check("pkill -f $PATTERN") is None


def test_remote_backends_are_not_inspected(victim):
    assert guard.detect_protected_process_kill(f"pkill -f {victim['svc'][1]}", "docker") is None


def test_floor_blocks_even_when_approvals_are_off(victim, monkeypatch):
    """The incident context: mode=off. The floor sits before the off/yolo/cron bypass."""
    from tools import approval_context
    monkeypatch.setattr(approval_context, "_get_approval_mode", lambda: "off")
    result = approval.check_all_command_guards(f"pkill -f {victim['svc'][1]}", "local")
    assert result["approved"] is False
    assert "BLOCKED" in result["message"]
    ok = approval.check_all_command_guards(f"pkill -f {victim['other'][1]}", "local")
    assert ok["approved"] is True
    ok = approval.check_dangerous_command(f"pkill -f {victim['other'][1]}", "local")
    assert ok["approved"] is True
    assert approval.check_dangerous_command(f"pkill -f {victim['svc'][1]}", "local")["approved"] is False


def test_guard_failure_fails_open(monkeypatch, victim):
    monkeypatch.setattr(guard, "_process_table", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    assert check(f"pkill -f {victim['svc'][1]}") is None


def test_default_protection_covers_runtime_processes():
    rx = [re.compile(p) for p in guard._DEFAULT_PROTECTED]
    for argv in (
        "hermes dashboard --no-open",
        "hermes --profile example gateway run",
        "hermes -p example dashboard",
        "python3 -m hermes_cli.main --profile default gateway run --replace",
        "python3 -m hermes_cli.main --profile default relay start --no-ssl",
        "python3 -m agent.managed_worker",
        "python3 -m hermes_cli.kanban_worker_client",
    ):
        assert any(r.search(argv) for r in rx), argv
    assert not any(r.search("python -m pytest tests/ -k hermes_cli") for r in rx)


# ---- regex metacharacters in a pgrep/pkill/killall pattern are not shell globs ----------

def _regex_of(marker):
    """``pkg-svc-abc123`` -> ``pkg-sv.*-abc123``: a regex that only pgrep, not the shell, interprets."""
    role, token = marker.rsplit("-", 1)
    return f"{role[:-1]}.*-{token}"


@pytest.mark.parametrize("quote", ["'", '"', ""])
def test_pkill_regex_pattern_is_resolved_not_treated_as_a_glob(victim, quote):
    svc, other = _regex_of(victim["svc"][1]), _regex_of(victim["other"][1])
    assert check(f"pkill -f {quote}{svc}{quote}")
    assert check(f"pkill -9 -f {quote}{svc}{quote}")
    assert check(f"pkill -f {quote}{other}{quote}") is None


def test_bracket_and_question_mark_regexes_are_resolved(victim):
    token = victim["svc"][1].rsplit("-", 1)[1]
    assert check(f"pkill -f 'pkg-s[v]c-{token}'")
    assert check(f"pkill -f 'pkg-s.c-{token}$'")
    assert check(f"pkill -f 'pkg-sv?c-{token}'")
    assert check(f"pkill -f 'pkg-o[t]her-{token}'") is None


def test_pgrep_regex_producers_are_resolved(victim):
    svc, other = _regex_of(victim["svc"][1]), _regex_of(victim["other"][1])
    assert check(f"kill $(pgrep -f '{svc}')")
    assert check(f'kill -9 `pgrep -f "{svc}"`')
    assert check(f"pids=$(pgrep -f '{svc}'); kill $pids")
    assert check(f"pgrep -f '{svc}' | xargs kill")
    assert check(f"kill $(pgrep -f '{other}')") is None
    assert check(f"pgrep -f '{other}' | xargs kill -9") is None


def test_killall_regex_name_is_resolved(victim, monkeypatch):
    name = os.path.basename(victim["svc"][0].args[0])
    monkeypatch.setattr(guard, "_protected_regexes", lambda: [re.compile(r"pkg-svc-")])
    assert check(f"killall -m '{re.escape(name[:3])}.*'")
    assert check(f"killall -m '^{re.escape(name)}$'")
    assert check("killall -m 'no-such-process-.*zz'") is None


def test_real_shell_expansion_stays_unresolvable(victim):
    svc = victim["svc"][1]
    assert check('pkill -f "$PATTERN"') is None
    assert check("pkill -f ${PATTERN}") is None
    assert check("pkill -f {a,b}") is None
    # Single quotes make ``$`` literal: the regex ``$PATTERN`` matches nothing, nothing to protect.
    assert check("pkill -f '$PATTERN'") is None
    assert check(f"pkill -f '{svc}'")


@pytest.mark.spawns_gateway_lookalike
def test_exact_incident_hermes_dot_star_gateway_is_refused(monkeypatch):
    """``pkill -f 'hermes.*gateway'`` used to pass because ``*`` looked like an unexpanded glob."""
    token = uuid.uuid4().hex[:10]
    gw = subprocess.Popen([sys.executable, "-c", "import sys; sys.stdin.read()",
                           "hermes_cli.main", "--profile", "default", "gateway", "run", token], stdin=subprocess.PIPE)
    try:
        monkeypatch.setattr(guard, "_protected_regexes",
                            lambda: [re.compile(p) for p in guard._DEFAULT_PROTECTED])
        for cmd in (f"pkill -f 'hermes.*gateway run {token}'",
                    f'pkill -f "hermes.*gateway run {token}"',
                    f"pkill -f hermes.*gateway.*{token}",
                    f"kill $(pgrep -f 'hermes.*gateway.*{token}')"):
            reason = check(cmd)
            assert reason, cmd
            assert "a Hermes service/worker process" in reason, reason
        assert check(f"pkill -f 'hermes.*gateway.*{token}-not-running'") is None
    finally:
        gw.stdin.close()
        gw.wait()


def test_shell_words_follows_shell_quoting():
    w = guard._shell_words
    assert w("-f 'hermes.*gateway'") == ["-f", "hermes.*gateway"]
    assert w('-f "a.*b c"') == ["-f", "a.*b c"]
    assert w("-f a\\*b") == ["-f", "a*b"]
    assert w("-f 'it\"s'") == ["-f", 'it"s']
    assert w("-f '$X'") == ["-f", "$X"]
    for unresolvable in ("-f $X", '-f "$X"', "-f `x`", "-f $(x)", "-f {a,b}", "-f ~/x", "-f 'open"):
        assert w(unresolvable) is None, unresolvable


@pytest.mark.spawns_gateway_lookalike
class TestProcessKillFloorRegexPatterns:
    """``pkill -f 'hermes.*gateway'`` reached the gateway because the floor read the
    regex ``*`` as an unresolved shell glob and allowed the kill. The floor must resolve a quoted
    pattern against live argv (read-only) even when approvals are off."""

    @pytest.fixture
    def fake_gateway(self):
        token = uuid.uuid4().hex[:10]
        proc = subprocess.Popen([sys.executable, "-c", "import sys; sys.stdin.read()",
                                 "hermes_cli.main", "--profile", "default", "gateway", "run", token], stdin=subprocess.PIPE)
        yield token
        proc.stdin.close()
        proc.wait()

    @pytest.mark.parametrize("template", [
        "pkill -f 'hermes.*gateway run {t}'",
        'pkill -f "hermes.*gateway.*{t}"',
        "pkill -9 -f 'hermes.*gateway.*{t}'",
        "kill $(pgrep -f 'hermes.*gateway.*{t}')",
        "pgrep -f 'hermes.*gateway.*{t}' | xargs kill",
    ])
    def test_regex_pattern_hitting_gateway_is_blocked_with_approvals_off(self, fake_gateway, template, monkeypatch):
        monkeypatch.setattr(guard, "_protected_regexes",
                            lambda: [re.compile(p) for p in guard._DEFAULT_PROTECTED])
        monkeypatch.setattr(approval_context, "_get_approval_mode", lambda: "off")
        result = approval.check_all_command_guards(template.format(t=fake_gateway), "local")
        assert result["approved"] is False
        assert "BLOCKED" in result["message"]

    def test_regex_pattern_missing_every_protected_process_is_allowed(self, fake_gateway, monkeypatch):
        monkeypatch.setattr(approval_context, "_get_approval_mode", lambda: "off")
        result = approval.check_all_command_guards(
            f"pkill -f 'hermes.*gateway.*{fake_gateway}-absent'", "local")
        assert result["approved"] is True


@pytest.mark.parametrize("filter_command", [
    "sort -o {path}", "uniq /dev/null {path}",
    "awk -f {path}", "awk 'BEGIN {{system(\"echo unsafe\")}}'",
])
def test_pid_pipeline_cannot_write_or_load_uninspected_filters(tmp_path, filter_command):
    output = tmp_path / "output"
    pipeline = f"pgrep -f no-such-process | {filter_command.format(path=output)}"
    assert guard._resolve_pipeline(pipeline) is None
    assert not output.exists()
