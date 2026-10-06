"""Credential-store snapshots must belong to the output producer. Regression for #125000."""

import json
from types import SimpleNamespace

import pytest

import agent.redact as redact
import tools.process_registry as processes
from gateway.run_notifications import GatewayNotificationsMixin
from hermes_constants import get_hermes_home
from tools.process_registry_results import save_completed_result
from tools.terminal_tool_result import finalize_foreground_result


SHARED = "shared-store-value"
OTHER = "other-filesystem-value"
PUBLIC = "PUBLIC_CODE"
SURFACES = ("foreground", "spill", "poll", "completion", "notification", "receipt")


def _deliver(surface, backend, command, output, tmp_path, monkeypatch, configured_backend=None):
    """Feed captured backend output into the real consumers, with real stores and caches."""
    if surface in {"foreground", "spill"}:
        result = {"output": output, "returncode": 0}
        spill = tmp_path / "output.txt"
        if surface == "spill":
            spill.write_text(output, encoding="utf-8")
            result.update(full_output_path=str(spill), output_total_chars=len(output))
        payload = json.loads(finalize_foreground_result(
            command=command, result=result, env=SimpleNamespace(is_local=backend == "local"),
            env_type=configured_backend or backend,
            effective_task_id="store-test", task_id="store-test", session_id="store-test",
            session_key="", workdir=None, command_cwd=None, approval_note=None,
        ))
        assert payload["exit_code"] == 0
        return spill.read_text(encoding="utf-8") if surface == "spill" else payload["output"]

    session = processes.ProcessSession(
        id="proc_store_test", command=command, output_buffer=output,
        task_id="store-test", exited=True, exit_code=0,
        pid_scope={"local": "host", "ssh": "sandbox"}.get(backend, ""),
    )
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
    return json.loads((get_hermes_home() / "logs/process-results/proc_store_test.json").read_text())["output"]


@pytest.mark.parametrize("surface, backend, configured_backend", [
    (surface, backend, backend) for surface in SURFACES for backend in ("local", "ssh", "unknown")
] + [("foreground", "ssh", "local"), ("spill", "ssh", "local")])
def test_same_path_on_another_backend_never_uses_host_inventory(
    surface, backend, configured_backend, tmp_path, monkeypatch,
):
    monkeypatch.setattr(redact, "_REDACT_ENABLED", True)
    store = tmp_path / ".netrc"
    store.write_text(f"machine host\npassword {SHARED}\n", encoding="utf-8")
    command = f'cat "{store}" app.py'
    output = f"{SHARED}\n{OTHER}\n{PUBLIC}"

    result = _deliver(surface, backend, command, output, tmp_path, monkeypatch, configured_backend)

    marker = "«redacted-secret»"
    expected = f"{marker}\n{OTHER}\n{PUBLIC}" if backend == "local" else "\n".join([marker] * 3)
    assert result == expected


@pytest.mark.parametrize("surface", SURFACES)
def test_incomplete_snapshot_never_reuses_a_complete_inventory(surface, tmp_path, monkeypatch):
    monkeypatch.setattr(redact, "_REDACT_ENABLED", True)
    prefix = f"machine host\npassword {SHARED}\n"
    monkeypatch.setattr(redact, "_STORE_READ_LIMIT", len(prefix))
    store = tmp_path / ".netrc"
    command = f'cat "{store}" app.py'
    output = f"{SHARED}\n{OTHER}\n{PUBLIC}"
    marker = "«redacted-secret»"

    # The incomplete snapshot has exactly the same captured prefix as the complete one.
    for body, expected in (
        (prefix, f"{marker}\n{OTHER}\n{PUBLIC}"),
        (prefix + f"machine other\npassword {OTHER}\n", "\n".join([marker] * 3)),
        (prefix + f"machine other\npassword {OTHER}\n", "\n".join([marker] * 3)),
        (prefix, f"{marker}\n{OTHER}\n{PUBLIC}"),
    ):
        if not store.exists() or store.read_text() != body:
            store.write_text(body, encoding="utf-8", newline="\n")
        assert _deliver(surface, "local", command, output, tmp_path, monkeypatch) == expected
