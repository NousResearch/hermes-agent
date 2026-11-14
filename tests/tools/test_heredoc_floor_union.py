"""Heredoc-aware scanning may only ADD detections: a body the shell executes, a heredoc whose
output feeds an interpreter, and a `<<` that is not a heredoc stay visible to the hardline floor,
and benign heredoc writes stay linear (#140704 follow-up)."""

import time

import pytest

from tools import approval, approval_context
from tools.approval_detection import (
    DANGEROUS_PATTERNS_COMPILED,
    detect_dangerous_command,
    detect_hardline_command,
)

# Built at runtime so no literal destructive command sits in the source.
RM_ROOT = "r" + "m -r" + "f /"
MKFS = "mk" + "fs.ext4 /dev/sda"


@pytest.fixture(params=["manual", "off"])
def gate_mode(request, monkeypatch):
    mode = request.param
    monkeypatch.setattr(approval_context, "_get_approval_mode", lambda: mode)
    monkeypatch.setattr(approval_context, "_get_approval_config", lambda: {"mode": mode})
    return mode


EXECUTED_BODIES = [
    f"cat <<EOF | sh\n{RM_ROOT}\nEOF",
    f"cat <<'EOF' | sudo sh\n{MKFS}\nEOF",
    f"sh -s <<'EOF'\n{RM_ROOT}\nEOF",
    f"ssh host <<'EOF'\n{RM_ROOT}\nEOF",
    f"bash <<'EOF'\n{RM_ROOT}\nEOF",
    f"sudo bash <<'EOF'\n{RM_ROOT}\nEOF",
    f"docker exec -i c sh <<'EOF'\n{RM_ROOT}\nEOF",
    # `<<` that is an arithmetic shift, not a heredoc: the next line is a real command.
    f"echo $((1<<2))\n{RM_ROOT}",
    f"x=$((a<<b))\n{RM_ROOT}\nb",
    # ANSI-C / locale-quoted delimiters: bash ends the body at ``E``, so the next line runs.
    f"cat <<$'E'\nx\nE\n{RM_ROOT}\n$E",
    f"cat <<$\"E\"\nx\nE\n{RM_ROOT}\n$E",
    f"cat <<'E'$'F'\nx\nEF\n{RM_ROOT}\nE$F",
]


@pytest.mark.parametrize("command", EXECUTED_BODIES)
def test_executed_heredoc_body_stays_on_hardline_floor(command):
    assert detect_hardline_command(command)[0] is True


@pytest.mark.parametrize("command", EXECUTED_BODIES)
def test_executed_heredoc_body_is_blocked_by_the_gate(gate_mode, command):
    result = approval.check_all_command_guards(command, "local")

    assert result["approved"] is False


@pytest.mark.parametrize("command", [
    f"python3 - <<'EOF'\nimport os\nos.system('{RM_ROOT}')\nEOF",
    f"osascript <<'EOF'\ndo shell script \"{RM_ROOT}\"\nEOF",
])
def test_code_heredoc_body_stays_visible_to_dangerous_scan(command):
    assert detect_dangerous_command(command)[0] is True


@pytest.mark.parametrize("command", [
    f"cat <<'EOF' > job.sh\n{RM_ROOT}\nEOF\nbash job.sh",
    f"cat > /tmp/job <<'EOF'\n{RM_ROOT}\nEOF\nchmod +x /tmp/job && /tmp/job",
    f"cat > 'job.sh' <<'EOF'\n{RM_ROOT}\nEOF\nbash job.sh",
    f"cat > /tmp/j <<'EOF'\n{RM_ROOT}\nEOF\ncd /tmp; sh j",
    f"cat > j <<'EOF'\n{RM_ROOT}\nEOF\nsudo -E env A=1 nohup bash 'j'",
])
def test_heredoc_written_then_run_keeps_its_body_on_the_floors(gate_mode, command):
    # The body lands in a file the same string runs: it is code, not data.
    assert detect_hardline_command(command)[0] is True
    assert approval.check_all_command_guards(command, "local")["approved"] is False


def test_quote_in_inert_body_does_not_hide_the_next_command(gate_mode):
    command = f"cat > notes.txt <<'EOF'\nit's data\nEOF\n{RM_ROOT}"

    assert approval.check_all_command_guards(command, "local")["approved"] is False


@pytest.mark.parametrize("opener", ["python3 - <<'EOF'\n# it's code", "osascript <<'EOF'\n-- it's code"])
def test_quote_in_code_heredoc_body_does_not_hide_the_next_command(opener):
    # Code bodies stay raw too; only their dropped copy keeps the quote from hiding the next line.
    assert detect_hardline_command(f"{opener}\nEOF\n{RM_ROOT}")[0] is True


def test_ansi_c_after_inert_heredoc_is_decoded():
    # Heredoc masking runs first, then ANSI-C expansion: `$'\x72m'` in the body is data, after it a command.
    command = "cat <<'EOF'\n$'\\x72m'\nEOF\n$'\\x72m' -rf /"

    assert detect_hardline_command(command)[0] is True


def _elapsed(fn, command):
    start = time.perf_counter()
    fn(command)
    return time.perf_counter() - start


def test_long_benign_heredoc_write_is_linear():
    body = "\n".join(f"line {i}: it's a plain sentence with words" for i in range(300))
    command = f"cat > notes.md <<'EOF'\n{body}\nEOF"

    assert _elapsed(detect_dangerous_command, command) + _elapsed(detect_hardline_command, command) < 2


def test_unterminated_heredoc_spam_is_linear():
    command = "cat <<EOF\n" * 200

    assert _elapsed(detect_dangerous_command, command) + _elapsed(detect_hardline_command, command) < 2


@pytest.mark.parametrize("command", [
    "curl -o f " * 7111 + "\nz",
    "curl -o f;" * 6400,
    "curl -O " + "https://e.x/f " * 6000 + "\nf",
    "curl -O https://e.x/f;" * 4000,
])
def test_download_then_run_rule_is_linear(command):
    # The whole detector is already ~1 s on these on main (other rules); time the new rule alone.
    rule = next(rx for rx, desc in DANGEROUS_PATTERNS_COMPILED if desc == "execute a just-downloaded file")

    assert _elapsed(rule.search, command) < 2
