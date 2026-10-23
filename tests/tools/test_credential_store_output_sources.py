"""Credential-store values in mixed terminal output come only from the store version the command
read, as the local executor attested it before running. Regression for #125000."""

import json
import os
import time

import pytest

from agent import redact
import tools.process_registry as processes
from tools import terminal_tool
from gateway.run_notifications import GatewayNotificationsMixin
from hermes_constants import get_hermes_home
from tools.environments.local import LocalEnvironment
from tools.process_registry_results import save_completed_result
from tools.terminal_tool_result import finalize_foreground_result


SHARED = "shared-store-value"
OTHER = "other-filesystem-value"
PUBLIC = "PUBLIC_CODE"
MARKER = "«redacted-secret»"
SURFACES = ("foreground", "spill", "poll", "completion", "notification", "receipt")


@pytest.fixture(autouse=True)
def _redaction_on(monkeypatch):
    monkeypatch.setattr(redact, "_REDACT_ENABLED", True)


def _age(*stores):
    """Push stores past the racy-signature window, as any store not edited seconds ago is."""
    aged = time.time() - 60
    for store in stores:
        os.utime(store, (aged, aged))


def _session(command, output, execution):
    return processes.ProcessSession(
        id="proc_store_test", command=command, output_buffer=output, total_output_chars=len(output),
        task_id="store-test", exited=True, exit_code=0, local_execution=execution,
    )


def _deliver(surface, execution, command, output, tmp_path, monkeypatch):
    """Feed captured output into the real consumers, with real stores and caches."""
    if surface in {"foreground", "spill"}:
        result = {"output": output, "returncode": 0}
        spill = tmp_path / "output.txt"
        if surface == "spill":
            spill.write_text(output, encoding="utf-8")
            result.update(full_output_path=str(spill), output_total_chars=len(output))
        payload = json.loads(finalize_foreground_result(
            command=command, result=result, env=None, env_type="local",
            effective_task_id="store-test", task_id="store-test", session_id="store-test",
            session_key="", workdir=None, command_cwd=None, approval_note=None, local_execution=execution,
        ))
        assert payload["exit_code"] == 0
        return spill.read_text(encoding="utf-8") if surface == "spill" else payload["output"]

    session = _session(command, output, execution)
    if surface == "poll":
        registry = processes.ProcessRegistry()
        registry._finished[session.id] = session
        monkeypatch.setattr(processes, "process_registry", registry)
        return processes._redact_process_result(registry.poll(session.id))["output_preview"]
    if surface == "completion":
        return GatewayNotificationsMixin._build_process_completion_event({}, session, session.id)["output"]
    if surface == "notification":
        return GatewayNotificationsMixin._redacted_output_tail(session, 2000)
    save_completed_result(session)
    return json.loads((get_hermes_home() / "logs/process-results/proc_store_test.json").read_text(encoding="utf-8"))["output"]


@pytest.mark.parametrize("surface", SURFACES)
@pytest.mark.parametrize("attested", [True, False], ids=["local", "unattested"])
def test_only_an_attested_store_version_supplies_values(surface, attested, tmp_path, monkeypatch):
    """A remote or unknown executor attests nothing, so a same-path host file never relaxes the
    conservative masking."""
    store = tmp_path / ".netrc"
    store.write_text(f"machine host\npassword {SHARED}\n", encoding="utf-8")
    _age(store)
    command = f'cat "{store}" app.py'
    execution = redact.local_execution(command, None) if attested else None

    result = _deliver(surface, execution, command, f"{SHARED}\n{OTHER}\n{PUBLIC}", tmp_path, monkeypatch)

    assert result == (f"{MARKER}\n{OTHER}\n{PUBLIC}" if attested else "\n".join([MARKER] * 3))


@pytest.mark.parametrize("surface", SURFACES)
def test_incomplete_snapshot_never_reuses_a_complete_inventory(surface, tmp_path, monkeypatch):
    prefix = f"machine host\npassword {SHARED}\n"
    monkeypatch.setattr(redact, "_STORE_READ_LIMIT", len(prefix))
    store = tmp_path / ".netrc"
    command = f'cat "{store}" app.py'
    output = f"{SHARED}\n{OTHER}\n{PUBLIC}"

    # The incomplete snapshot has exactly the same captured prefix as the complete one.
    for body, expected in (
        (prefix, f"{MARKER}\n{OTHER}\n{PUBLIC}"),
        (prefix + f"machine other\npassword {OTHER}\n", "\n".join([MARKER] * 3)),
        (prefix + f"machine other\npassword {OTHER}\n", "\n".join([MARKER] * 3)),
        (prefix, f"{MARKER}\n{OTHER}\n{PUBLIC}"),
    ):
        if not store.exists() or store.read_text() != body:
            store.write_text(body, encoding="utf-8", newline="\n")
        _age(store)
        execution = redact.local_execution(command, None)
        assert _deliver(surface, execution, command, output, tmp_path, monkeypatch) == expected


# --- Real local execution: the executor's HOME, and a store the command itself rewrites ---

APP_LINES = ["def handler(request):", "    token = load_token()  # PUBLIC_CODE_SURVIVES", "    return token"]


def _run_local(command, home, workdir):
    """The production foreground path: attestation, LocalEnvironment.execute, finalization."""
    env = LocalEnvironment(cwd=str(workdir), env={"HOME": str(home)})
    plan = terminal_tool._ExecPlan(config={}, env_type="local", effective_task_id="store-test", image="",
                                   cwd=str(workdir), host_cwd=None, effective_timeout=60)
    payload = json.loads(terminal_tool._run_foreground(
        command, env, plan, task_id="store-test", session_id="store-test", session_key="",
        workdir=str(workdir), approval_note=None, clear_interrupt=False, metered=False))
    assert payload["exit_code"] == 0, payload
    return payload["output"]


@pytest.fixture
def executor_home(tmp_path, monkeypatch):
    """A store under the executor's HOME, an app.py beside it, and a controller HOME that holds a
    different store, so resolving ``~`` on the controller would not match the printed values."""
    home, controller = tmp_path / "executor-home", tmp_path / "controller-home"
    home.mkdir()
    controller.mkdir()
    (controller / ".netrc").write_text("machine decoy\npassword decoy-controller-value\n", encoding="utf-8")
    monkeypatch.setenv("HOME", str(controller))
    monkeypatch.setenv("USERPROFILE", str(controller))
    (home / "app.py").write_text("\n".join(APP_LINES) + "\n", encoding="utf-8", newline="\n")
    return home


@pytest.mark.parametrize("spelling", ['"$HOME/.netrc"', "~/.netrc", "${HOME}/.netrc"])
def test_shell_home_spellings_resolve_against_the_executor(spelling, executor_home):
    store = executor_home / ".netrc"
    store.write_text(f"machine host\nlogin bob\npassword {SHARED}\n", encoding="utf-8", newline="\n")
    _age(store)

    out = _run_local(f"cat {spelling} app.py", executor_home, executor_home)

    assert SHARED not in out
    assert all(line in out.splitlines() for line in APP_LINES)


def test_a_store_rewritten_by_the_command_never_certifies_its_old_values(executor_home):
    """The command prints ``shared + old`` and leaves ``shared + new`` on disk. The post-command
    file is not the version that produced the output, so no printed credential survives."""
    store = executor_home / ".netrc"
    old = "rotated-old-secret-8R"
    store.write_text(f"machine a\npassword {SHARED}\nmachine b\npassword {old}\n", encoding="utf-8",
                     newline="\n")
    _age(store)
    rewrite = f"printf 'machine a\\npassword {SHARED}\\nmachine b\\npassword rotated-new-secret-9S\\n' > ~/.netrc"

    out = _run_local(f"cat ~/.netrc app.py; {rewrite}", executor_home, executor_home)

    assert SHARED not in out and old not in out
    assert "rotated-new-secret-9S" in store.read_text()


# --- Consumers that redact a window of the output ---

CONTINUATION = "continuation-value-q7X9m2K8r4T6"


@pytest.fixture
def long_pypirc(tmp_path):
    """An INI password whose continuation lines run far past every output window."""
    store = tmp_path / ".pypirc"
    filler = "".join(f"    filler-secret-line-{n:05d}\n" for n in range(3000))
    store.write_text(f"[pypi]\nusername = __token__\npassword = alpha\n{filler}    {CONTINUATION}\n",
                     encoding="utf-8", newline="\n")
    return store


def test_foreground_head_tail_never_shows_an_orphan_continuation(long_pypirc, tmp_path):
    out = _run_local(f'cat "{long_pypirc}"', tmp_path, tmp_path)

    assert "TRUNCATED" in out
    assert CONTINUATION not in out and "filler-secret-line" not in out
    assert "username = __token__" in out


@pytest.mark.parametrize("surface", ["poll", "log_tail", "log_offset", "wait", "heartbeat", "completion",
                                     "notification", "receipt"])
def test_background_windows_never_show_an_orphan_continuation(surface, long_pypirc, tmp_path, monkeypatch):
    command = f'cat "{long_pypirc}"'
    output = long_pypirc.read_text(encoding="utf-8")
    if surface in {"poll", "completion", "notification", "receipt"}:
        result = _deliver(surface, None, command, output, tmp_path, monkeypatch)
    else:
        session = _session(command, output, None)
        registry = processes.ProcessRegistry()
        registry._finished[session.id] = session
        monkeypatch.setattr(processes, "process_registry", registry)
        if surface == "heartbeat":
            session._heartbeat_total_at_last = 0
            registry._emit_heartbeat(session, time.time())
            result = registry.completion_queue.get_nowait()["output"]
        else:
            raw = {"log_tail": lambda: registry.read_log(session.id),
                   "log_offset": lambda: registry.read_log(session.id, offset=2000, limit=50),
                   "wait": lambda: registry.wait(session.id, timeout=1)}[surface]()
            result = processes._redact_process_result(raw)["output"]

    assert result
    assert CONTINUATION not in result and "filler-secret-line" not in result
