"""Approval coverage for Hermes security-sensitive configuration mutations."""

import pytest

from tools.approval import detect_dangerous_command
from tools.approval_detection import detect_execute_code_config_mutation


@pytest.mark.parametrize(
    "command",
    [
        "hermes config set approvals.mode off",
        "hermes config set approvals.cron_mode approve",
        "hermes config set approvals.single_query_mode approve",
        "hermes config set command_allowlist '[\"rm\"]'",
        "hermes config set security.protected_instruction_files false",
        "hermes config set terminal.backend local",
        "hermes config set terminal.container_persistent false",
        "hermes config set terminal.docker_network true",
        "hermes config set terminal.docker_extra_args '[\"--privileged\"]'",
        "hermes config set terminal.docker_extra_args '[\"--cap-add\",\"SYS_ADMIN\"]'",
        "hermes config set terminal.docker_volumes '[\"/:/host\"]'",
        "hermes config unset approvals.mode",
        "hermes -p research config set terminal.docker_extra_args '[\"--network=host\"]'",
    ],
)
def test_security_sensitive_hermes_config_mutations_are_dangerous(command):
    dangerous, _pattern_key, description = detect_dangerous_command(command)

    assert dangerous is True
    assert description == "modify Hermes security-sensitive configuration"


@pytest.mark.parametrize(
    "command",
    [
        "hermes config set terminal.timeout 60",
        "hermes config set display.language en",
        "hermes config set model.default test-model",
        "hermes config get approvals.mode",
        "hermes config get terminal.docker_extra_args",
        "hermes config check",
    ],
)
def test_benign_hermes_config_commands_are_not_dangerous(command):
    dangerous, _pattern_key, _description = detect_dangerous_command(command)

    assert dangerous is False


def test_docker_approval_boundary_is_based_on_host_access():
    """The command detector flags the mutation; the container fast-path decides
    whether an actually isolated Docker execution needs the ordinary prompt."""
    dangerous, _pattern_key, _description = detect_dangerous_command(
        'hermes config set terminal.docker_extra_args \'["--privileged"]\''
    )
    assert dangerous is True


@pytest.mark.parametrize(
    "code",
    [
        "open('/home/example/.hermes/config.yaml', 'w').write('approvals: {}')",
        "open(file='/home/example/.hermes/config.yaml', mode='w').write('approvals: {}')",
        "io.open('/home/example/.hermes/config.yaml', 'w')",
        (
            "from io import open as file_open\n"
            "file_open('/home/example/.hermes/config.yaml', 'w')"
        ),
        "Path('/home/example/.hermes/config.yaml').write_text('approvals: {}')",
        (
            "from pathlib import Path\n"
            "config_path = Path('/home/example/.hermes/config.yaml')\n"
            "config_path.write_text('changed')"
        ),
        (
            "from pathlib import Path as P\n"
            "config_path = P('/home/example/.hermes/config.yaml')\n"
            "config_path.write_bytes(b'changed')"
        ),
        "Path('/home/example/.hermes/config.yaml').write_bytes(b'config')",
        "Path('/home/example/.hermes/config.yaml').open('w').write('approvals: {}')",
        "os.open('/home/example/.hermes/config.yaml', os.O_WRONLY)",
        "os.open('/home/example/.hermes/config.yaml', os.O_RDONLY | 1)",
        "os.open('/home/example/.hermes/config.yaml', os.O_CLOEXEC | 1)",
        "os.unlink('/home/example/.hermes/config.yaml')",
        "os.replace('/tmp/evil.yaml', '/home/example/.hermes/config.yaml')",
        "shutil.copy(src='/tmp/input.yaml', dst='/home/example/.hermes/config.yaml')",
        "shutil.move(src='/tmp/input.yaml', dst='/home/example/.hermes/config.yaml')",
        "os.symlink('/tmp/attacker.yaml', '/home/example/.hermes/config.yaml')",
        "os.symlink(src='/tmp/attacker.yaml', dst='/home/example/.hermes/config.yaml')",
        (
            "import os as fs\n"
            "fs.replace('/tmp/input.yaml', "
            "'/home/example/.hermes/config.yaml')"
        ),
        (
            "from builtins import open as write_file\n"
            "write_file('/home/example/.hermes/config.yaml', 'w')"
        ),
        (
            "from pathlib import Path as P\n"
            "P('/home/example/.hermes/config.yaml').write_text('changed')"
        ),
        (
            "import pathlib as pl\n"
            "pl.Path('/home/example/.hermes/config.yaml').write_text('changed')"
        ),
        (
            "import os\n"
            "os.truncate('/home/example/.hermes/config.yaml', 0)"
        ),
        (
            "import os\n"
            "os.link('/tmp/source', '/home/example/.hermes/config.yaml')"
        ),
        (
            "from pathlib import Path\n"
            "Path('/home/example/.hermes/config.yaml').hardlink_to('/tmp/source')"
        ),
        (
            "from pathlib import Path\n"
            "config_path = Path.home() / '.hermes' / 'config.yaml'\n"
            "config_path.write_text('changed')"
        ),
        (
            "from pathlib import Path\n"
            "config_path = Path('/home/example/.hermes/config.yaml')\n"
            "writer = config_path.write_text\n"
            "writer('changed')"
        ),
        "get_config_path().write_text('approvals: {}')",
    ],
)
def test_execute_code_config_mutations_are_detected(code):
    detected, reason = detect_execute_code_config_mutation(code)

    assert detected is True
    assert reason
    assert "config.yaml" in reason


@pytest.mark.parametrize(
    "code",
    [
        "sum(range(10))",
        "open('/tmp/example.txt', 'w').write('hello')",
        "Path('/tmp/example.txt').write_text('hello')",
        (
            "from pathlib import Path\n"
            "Path.home().joinpath('ordinary.txt').write_text('hello')"
        ),
        (
            "Path('/tmp/' + '/home/example/.hermes/config.yaml')"
            ".write_text('test')"
        ),
        "Path('/home/example/.hermes/config.yaml').read_text()",
        (
            "from pathlib import Path\n"
            "Path('/home/example/.hermes/config.yaml').read_text()\n"
            "Path('/tmp/unrelated-output.txt').write_text('hello')"
        ),
        (
            "from pathlib import Path\n"
            "print('config.yaml')\n"
            "Path('/tmp/unrelated-output.txt').write_text('hello')"
        ),
        "os.open('/home/example/.hermes/config.yaml', os.O_RDONLY)",
        (
            "from os import open as os_open, O_RDONLY\n"
            "os_open('/home/example/.hermes/config.yaml', O_RDONLY)"
        ),
        "shutil.copy('/home/example/.hermes/config.yaml', '/tmp/config-backup.yaml')",
        (
            "import os\n"
            "os.truncate('/tmp/unrelated.txt', 0)"
        ),
        (
            "import os\n"
            "os.link('/tmp/source', '/tmp/unrelated.txt')"
        ),
        (
            "from pathlib import Path\n"
            "Path('/tmp/destination').hardlink_to('/home/example/.hermes/config.yaml')"
        ),
        "",
    ],
)
def test_execute_code_benign_or_read_only_code_is_not_flagged(code):
    detected, reason = detect_execute_code_config_mutation(code)

    assert detected is False
    assert reason is None


@pytest.mark.parametrize(
    "guard_name",
    ["check_dangerous_command", "check_all_command_guards"],
)
@pytest.mark.parametrize(
    "bypass",
    ["yolo", "off", "permanent_allowlist"],
)
def test_sensitive_config_guard_rejects_approval_bypasses(
    monkeypatch, guard_name, bypass
):
    """Security-sensitive config changes must not inherit general approval bypasses."""
    from tools import approval as approval_module
    from tools import approval_context

    monkeypatch.setattr(
        approval_module, "_YOLO_MODE_FROZEN", bypass == "yolo"
    )
    monkeypatch.setattr(
        approval_context,
        "_get_approval_mode",
        lambda: "off" if bypass == "off" else "manual",
    )
    monkeypatch.setattr(
        approval_module,
        "_command_matches_permanent_allowlist",
        lambda command: bypass == "permanent_allowlist",
    )

    guard = getattr(approval_module, guard_name)
    result = guard(
        "hermes config set approvals.mode off",
        "local",
    )

    assert result["approved"] is False, (
        f"{guard_name} silently allowed sensitive config mutation "
        f"under bypass={bypass}: {result}"
    )
    assert (
        result.get("pattern_key") == "hermes_security_config_mutation"
        or result.get("description")
        == "modify Hermes security-sensitive configuration"
    )


@pytest.mark.parametrize(
    ("prepared_decision", "expected_approved"),
    [
        pytest.param(
            {"approved": True, "message": None},
            False,
            id="policy-only-approval-is-blocked",
        ),
        pytest.param(
            {"approved": True, "message": None, "smart_approved": True},
            False,
            id="smart-approval-is-blocked",
        ),
        pytest.param(
            {"approved": True, "message": None, "user_approved": True},
            True,
            id="explicit-human-approval-is-preserved",
        ),
    ],
)
def test_prepared_guard_cannot_bypass_sensitive_config_protection(
    monkeypatch, prepared_decision, expected_approved
):
    """Prepared non-human approvals must not authorize sensitive config mutations."""
    from tools import approval as approval_module
    from tools import approval_context
    from agent import terminal_approval_batch

    command = "hermes config set approvals.mode off"

    monkeypatch.setattr(approval_module, "_YOLO_MODE_FROZEN", False)
    monkeypatch.setattr(
        approval_context, "_get_approval_mode", lambda: "manual"
    )
    monkeypatch.setattr(
        approval_module, "_command_matches_permanent_allowlist",
        lambda _command: False,
    )
    monkeypatch.setattr(
        terminal_approval_batch,
        "consume_prepared_guard",
        lambda *_args: prepared_decision,
    )

    result = approval_module.check_all_command_guards(command, "local")

    assert result["approved"] is expected_approved, (
        f"Unexpected result for prepared decision {prepared_decision}: {result}"
    )

    if expected_approved:
        assert result == prepared_decision
    else:
        assert (
            result.get("pattern_key") == "hermes_security_config_mutation"
        )
        assert (
            result.get("description")
            == "modify Hermes security-sensitive configuration"
        )
