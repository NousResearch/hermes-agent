"""Default-deny regression: shell structures retain base Docker lifecycle matches."""

import pytest

from tools.approval import detect_dangerous_command

LOST = ["{\n echo 'docker restart victim'\n} > deploy.sh",
 ">deploy.sh echo 'docker restart victim'",
 "> deploy.sh echo 'docker restart victim'",
 ">>deploy.sh echo 'docker restart victim'",
 "&>deploy.sh echo 'docker restart victim'",
 ">|deploy.sh echo 'docker restart victim'",
 "1>deploy.sh echo 'docker restart victim'",
 "exec >deploy.sh; echo 'docker restart victim'",
 "exec >>deploy.sh; echo 'docker restart victim'",
 "exec >deploy.sh\necho 'docker restart victim'",
 "exec &>deploy.sh; printf 'docker restart victim\\n'",
 "exec 3>deploy.sh; echo 'docker restart victim' >&3",
 "exec 3>deploy.sh; echo 'docker restart victim' 1>&3",
 "function f {\n echo 'docker restart victim'\n}\nf > deploy.sh",
 "{ cat <<'EOF'\ndocker restart victim\nEOF\n} > deploy.sh",
 "{ cat <<'EOF'\ndocker restart victim\nEOF\n} >> deploy.sh",
 ">deploy.sh cat <<'EOF'\ndocker restart victim\nEOF",
 ">>deploy.sh cat <<'EOF'\ndocker restart victim\nEOF",
 ">deploy.sh cat <<< 'docker restart victim'",
 "{\n echo 'docker compose down'\n} > deploy.sh",
 ">deploy.sh echo 'docker compose down'",
 "> deploy.sh echo 'docker compose down'",
 ">>deploy.sh echo 'docker compose down'",
 "&>deploy.sh echo 'docker compose down'",
 ">|deploy.sh echo 'docker compose down'",
 "1>deploy.sh echo 'docker compose down'",
 "exec >deploy.sh; echo 'docker compose down'",
 "exec >>deploy.sh; echo 'docker compose down'",
 "exec >deploy.sh\necho 'docker compose down'",
 "exec 3>deploy.sh; echo 'docker compose down' >&3",
 "exec 3>deploy.sh; echo 'docker compose down' 1>&3",
 "function f {\n echo 'docker compose down'\n}\nf > deploy.sh",
 "{ cat <<'EOF'\ndocker compose down\nEOF\n} > deploy.sh",
 "{ cat <<'EOF'\ndocker compose down\nEOF\n} >> deploy.sh",
 ">deploy.sh cat <<'EOF'\ndocker compose down\nEOF",
 ">>deploy.sh cat <<'EOF'\ndocker compose down\nEOF",
 ">deploy.sh cat <<< 'docker compose down'"]

TEMPLATES = [
    "echo '{p}' > file{n}",
    " >file{n} echo '{p}'",
    "{{ echo '{p}'; }} > file{n}",
    "(echo '{p}') > file{n}",
    "echo '{p}' | sh -s {n}",
    "exec >file{n}; echo '{p}'",
    "FOO={n} echo '{p}'",
    "f{n}() {{ echo '{p}'; }}; f{n}",
    "cat <<'END' >file{n}\n{p}\nEND",
    "sh <(echo '{p}') {n}",
]
GENERATED = [t.format(p=p, n=n) for t in TEMPLATES for p in ("docker restart victim", "docker compose down") for n in range(10)]


@pytest.mark.parametrize("command", LOST + GENERATED)
def test_structured_commands_stay_flagged(command):
    assert detect_dangerous_command(command)[0], command


# Each exception is quoted message data in one explicitly allowed carrier.
PROSE = [
    "echo 'we will docker restart later'",
    "printf 'docker restart later'",
    "git commit -m 'note docker restart counts'",
    "hermes kanban comment t1 'note docker restart counts'",
    "hermes kanban create title --body 'note docker restart counts'",
]


@pytest.mark.parametrize("command", PROSE)
def test_quoted_message_exceptions(command):
    assert detect_dangerous_command(command) == (False, None, None)
