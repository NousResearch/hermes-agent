"""Background jobs spawned inside a restart-safe cron execution get their own scope unit.

Left in the execution's ``--collect`` scope, a detached job keeps that unit active for days
after the ledger row is terminal, and the gateway's release of the unit would kill it.
"""

import time

import pytest

import tools.process_registry as pr
from cron.worker_bootstrap import EXTERNAL_WORKER_ENV


@pytest.fixture()
def linux_no_gateway(monkeypatch):
    monkeypatch.setattr(pr, "_IS_LINUX", True)
    monkeypatch.setattr(pr, "_is_supervised_gateway_process", lambda: False)
    monkeypatch.setattr(pr, "_systemd_run_user_scope_available", lambda: True)
    monkeypatch.setattr("shutil.which", lambda name: f"/usr/bin/{name}")


def _session():
    return pr.ProcessSession(id="proc_test1", command="sleep 1", task_id="t1", started_at=time.time())


def test_cron_execution_background_spawn_gets_private_scope(monkeypatch, linux_no_gateway):
    monkeypatch.setenv(EXTERNAL_WORKER_ENV, "exec-1")
    session = _session()
    argv = pr.ProcessRegistry()._scope_argv(session, "sleep 1", "proc_test1", "Local")
    assert argv[0] == "/usr/bin/systemd-run"
    assert argv[argv.index("--unit") + 1] == "hermes-worker-proc_test1"
    assert session.systemd_unit == "hermes-worker-proc_test1.scope"


def test_plain_process_background_spawn_not_wrapped(monkeypatch, linux_no_gateway):
    monkeypatch.delenv(EXTERNAL_WORKER_ENV, raising=False)
    session = _session()
    argv = pr.ProcessRegistry()._scope_argv(session, "sleep 1", "proc_test1", "Local")
    assert "/usr/bin/systemd-run" not in argv
    assert not session.systemd_unit
