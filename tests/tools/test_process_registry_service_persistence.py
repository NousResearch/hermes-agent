"""A service declaration must survive a gateway restart, and only while alive.

Two invariants, both regressions found against the merged #17 service-node work:

1. **Round-trip.** ``_write_checkpoint`` → ``recover_from_checkpoint`` must
   preserve the five ``service_*`` fields. Without them a background service is
   adopted after a gateway restart as an anonymous process: ``service_name``
   comes back empty, ``collect_service_declarations()`` skips it, and the
   service silently disappears from the cron interflow graph while its process
   is still running. The process is the lease, so the declaration has to outlive
   the gateway exactly as long as the process does.

2. **Liveness is still probed.** Fixing (1) creates the mirror hazard: a
   recovered session's ``exited`` flag is stale (there is no waitable handle),
   so a service whose process died while the gateway was down would be reported
   live forever. ``collect_service_declarations`` must reconcile detached
   sessions against the real PID like every other read path does.

These assert the contract between the two halves (what is persisted must be what
is restored, and liveness must reflect the OS), not any particular field list.
"""
import json
import time
from unittest.mock import MagicMock, patch

import pytest

from tools.process_registry import ProcessRegistry, ProcessSession


@pytest.fixture
def registry():
    return ProcessRegistry()


def _service_session(sid="proc_dash", pid=4242):
    s = ProcessSession(
        id=sid,
        command="python3 dashboard.py",
        task_id="t1",
        started_at=time.time(),
        pid=pid,
        pid_scope="host",
        host_start_time=int(time.time()),
    )
    s.service_name = "Compendium Dashboard"
    s.service_description = "FastAPI dashboard on :8700 over the compendium."
    s.service_inputs = ["postgres:agentic_payments.transfers"]
    s.service_outputs = ["file:/tmp/dash-cache.json"]
    s.service_side_effects = ["notify:ops"]
    s.service_relationships = [
        {"predicate": "runs_in", "object": "runtime:docker"},
    ]
    s.service_source_files = ["dashboard.py", "app/routes/"]
    s.service_code_control = {
        "status": "verified",
        "enforcement": "merged-pull-request",
        "provider": "github",
        "repository": "owner/dashboard",
        "base_branch": "main",
        "revision": "a" * 40,
        "pull_request": {
            "number": 42,
            "url": "https://github.com/owner/dashboard/pull/42",
            "merged_at": "2026-09-03T12:00:00Z",
        },
    }
    return s


class TestServiceDeclarationSurvivesRestart:
    def test_fast_exit_cannot_be_reinserted_after_reader_finishes(self, registry, tmp_path):
        """Publication precedes reader start, so running/finished never overlap."""
        with patch("tools.process_registry.CHECKPOINT_PATH", tmp_path / "procs.json"):
            session = registry.spawn_local(
                "exit 0",
                cwd="/tmp",
                service_declaration={
                    "name": "Fast service",
                    "description": "Exits immediately.",
                    "inputs": [],
                    "outputs": [],
                    "side_effects": [],
                },
            )
            session._reader_thread.join(timeout=3)

        assert session.id not in registry._running
        assert session.id in registry._finished

    def test_pty_checkpoint_failure_does_not_execute_command_twice(self, registry):
        pty_process = MagicMock(pid=5555)
        pty_module = MagicMock()
        pty_module.PtyProcess.spawn.return_value = pty_process

        def checkpoint(*, strict=False):
            if strict:
                raise RuntimeError("checkpoint failed")

        with (
            patch.dict("sys.modules", {"ptyprocess": pty_module}),
            patch("tools.process_registry._find_shell", return_value="/bin/bash"),
            patch("subprocess.Popen") as pipe_spawn,
            patch.object(registry, "_write_checkpoint", side_effect=checkpoint),
        ):
            with pytest.raises(RuntimeError, match="checkpoint failed"):
                registry.spawn_local("side-effecting-command", use_pty=True)

        pty_module.PtyProcess.spawn.assert_called_once()
        pty_process.terminate.assert_called_once_with(force=True)
        pipe_spawn.assert_not_called()
        assert not registry._running

    def test_spawn_rolls_back_when_initial_checkpoint_cannot_commit(self, registry, tmp_path):
        declaration = {
            "name": "Durable API",
            "description": "Must not run anonymously.",
            "inputs": [],
            "outputs": ["http://127.0.0.1:8764"],
            "side_effects": [],
        }
        with (
            patch("tools.process_registry.CHECKPOINT_PATH", tmp_path / "procs.json"),
            patch("utils.atomic_json_write", side_effect=OSError("disk full")),
        ):
            with pytest.raises(RuntimeError, match="checkpoint"):
                registry.spawn_local(
                    "sleep 30",
                    cwd="/tmp",
                    service_declaration=declaration,
                )

        assert registry._running == {}

    def test_spawn_persists_complete_service_before_returning(self, registry, tmp_path):
        """The first durable checkpoint is already a complete service lease.

        A gateway crash immediately after ``spawn_local`` returns must never
        recover a running but anonymous process.  Registration is therefore an
        input to spawn, not a second mutation the terminal tool performs later.
        """
        checkpoint = tmp_path / "procs.json"
        declaration = {
            "name": "Atomic API",
            "description": "Serves the atomic registration test.",
            "inputs": ["file:/tmp/input"],
            "outputs": ["http://127.0.0.1:8765"],
            "side_effects": [],
        }
        with patch("tools.process_registry.CHECKPOINT_PATH", checkpoint):
            session = registry.spawn_local(
                "sleep 30",
                cwd="/tmp",
                task_id="atomic-service",
                service_declaration=declaration,
            )
            try:
                entry = json.loads(checkpoint.read_text(encoding="utf-8"))[0]
                assert entry["session_id"] == session.id
                assert entry["service_name"] == declaration["name"]
                assert entry["service_description"] == declaration["description"]
                assert entry["service_inputs"] == declaration["inputs"]
                assert entry["service_outputs"] == declaration["outputs"]
            finally:
                registry.kill_process(session.id)

    def test_health_gated_service_is_hidden_until_lease_commit(self, registry, tmp_path):
        checkpoint = tmp_path / "procs.json"
        declaration = {
            "name": "Ready API",
            "description": "Only visible after readiness passes.",
            "inputs": [],
            "outputs": ["http://127.0.0.1:8766"],
            "side_effects": [],
        }
        health_spec = {
            "type": "http",
            "url": "http://127.0.0.1:8766/health",
            "expected_status": 200,
            "timeout_seconds": 2.0,
            "startup_timeout_seconds": 30.0,
        }
        evidence = {
            "status": "healthy",
            "probe": "http",
            "target": health_spec["url"],
            "checked_at": "2026-08-23T22:00:00Z",
            "latency_ms": 3.2,
            "message": "HTTP 200",
        }
        with patch("tools.process_registry.CHECKPOINT_PATH", checkpoint):
            session = registry.spawn_local(
                "sleep 30",
                cwd="/tmp",
                service_declaration=declaration,
                service_health=health_spec,
            )
            try:
                assert registry.collect_service_declarations() == []
                registry.commit_service_lease(session.id, evidence)
                services = registry.collect_service_declarations(probe_health=False)
                assert services[0]["health"] == evidence

                entry = json.loads(checkpoint.read_text(encoding="utf-8"))[0]
                assert entry["service_lease_state"] == "active"
                assert entry["service_health"] == health_spec
                assert entry["service_health_evidence"] == evidence
            finally:
                registry.kill_process(session.id)

    def test_checkpoint_round_trip_preserves_declaration(self, registry, tmp_path):
        """The declaration a service registered with must be the one it comes
        back with — otherwise it vanishes from the graph on restart."""
        checkpoint = tmp_path / "procs.json"
        original = _service_session()
        with patch("tools.process_registry.CHECKPOINT_PATH", checkpoint):
            registry._running[original.id] = original
            registry._write_checkpoint()

            # Persisted at all?
            entry = json.loads(checkpoint.read_text())[0]
            assert entry["service_name"] == original.service_name
            assert entry["service_description"] == original.service_description
            assert entry["service_inputs"] == original.service_inputs
            assert entry["service_outputs"] == original.service_outputs
            assert entry["service_side_effects"] == original.service_side_effects
            assert entry["service_relationships"] == original.service_relationships
            assert entry["service_source_files"] == original.service_source_files
            assert entry["service_code_control"] == original.service_code_control

            # Restored into a NEW registry, with the process still alive.
            fresh = ProcessRegistry()
            with patch.object(fresh, "_host_pid_is_ours", return_value=True), \
                 patch.object(ProcessRegistry, "_safe_host_start_time",
                              return_value=original.host_start_time):
                assert fresh.recover_from_checkpoint() == 1

            revived = fresh._running[original.id]
            assert revived.service_name == original.service_name
            assert revived.service_description == original.service_description
            assert revived.service_inputs == original.service_inputs
            assert revived.service_outputs == original.service_outputs
            assert revived.service_side_effects == original.service_side_effects
            assert revived.service_relationships == original.service_relationships
            assert revived.service_source_files == original.service_source_files
            assert revived.service_code_control == original.service_code_control

    def test_recovered_service_still_appears_in_the_graph(self, registry, tmp_path):
        """End-to-end of the actual symptom: after a restart the still-running
        service must still be collected for build_cron_graph."""
        checkpoint = tmp_path / "procs.json"
        original = _service_session()
        with patch("tools.process_registry.CHECKPOINT_PATH", checkpoint):
            registry._running[original.id] = original
            registry._write_checkpoint()

            fresh = ProcessRegistry()
            with patch.object(fresh, "_host_pid_is_ours", return_value=True), \
                 patch.object(ProcessRegistry, "_safe_host_start_time",
                              return_value=original.host_start_time):
                fresh.recover_from_checkpoint()

            with patch.object(fresh, "_host_pid_is_ours", return_value=True):
                services = fresh.collect_service_declarations()

        assert len(services) == 1
        assert services[0]["label"] == "Compendium Dashboard"
        assert services[0]["inputs"] == ["postgres:agentic_payments.transfers"]
        assert services[0]["description"]  # required by normalize_service_declaration
        assert services[0]["relationships"] == original.service_relationships
        assert services[0]["source_files"] == original.service_source_files
        assert services[0]["code_control"] == original.service_code_control

    def test_declaration_shape_matches_graph_builder(self, registry, tmp_path):
        """A recovered service must feed build_cron_graph and converge with a
        cron on the shared resource node — the whole point of declaring it."""
        from cron.jobs import build_cron_graph

        checkpoint = tmp_path / "procs.json"
        original = _service_session()
        with patch("tools.process_registry.CHECKPOINT_PATH", checkpoint):
            registry._running[original.id] = original
            registry._write_checkpoint()
            fresh = ProcessRegistry()
            with patch.object(fresh, "_host_pid_is_ours", return_value=True), \
                 patch.object(ProcessRegistry, "_safe_host_start_time",
                              return_value=original.host_start_time):
                fresh.recover_from_checkpoint()
            with patch.object(fresh, "_host_pid_is_ours", return_value=True):
                services = fresh.collect_service_declarations()

        jobs = [{
            "id": "indexer",
            "name": "indexer",
            "outputs": ["postgres:agentic_payments.transfers"],
        }]
        graph = build_cron_graph(jobs=jobs, services=services)
        shared = "postgres:agentic_payments.transfers"

        # The shared ref must be ONE node, not one per producer/consumer.
        assert sum(1 for n in graph["nodes"] if n["id"] == shared) == 1
        assert {(e["source"], e["target"], e["type"]) for e in graph["edges"]} >= {
            ("indexer", shared, "writes"),
            (shared, original.id, "reads"),
            (original.id, "github:owner/dashboard", "source_repository"),
            (original.id, "pr:owner/dashboard#42", "released_via"),
            (original.id, f"git:{'a' * 40}", "runs_revision"),
        }


class TestRecoveredServiceLiveness:
    def test_dead_recovered_service_is_not_reported_live(self, registry, tmp_path):
        """The mirror hazard of persisting the declaration: if the process died
        while the gateway was down, the service must NOT still be in the graph.
        Presence in _running is not evidence of liveness for detached sessions."""
        checkpoint = tmp_path / "procs.json"
        original = _service_session()
        with patch("tools.process_registry.CHECKPOINT_PATH", checkpoint):
            registry._running[original.id] = original
            registry._write_checkpoint()

            fresh = ProcessRegistry()
            # Alive at recovery time...
            with patch.object(fresh, "_host_pid_is_ours", return_value=True), \
                 patch.object(ProcessRegistry, "_safe_host_start_time",
                              return_value=original.host_start_time):
                fresh.recover_from_checkpoint()
            assert fresh._running[original.id].detached is True

            # ...but the PID is gone (or recycled) by the time we build a graph.
            with patch.object(fresh, "_host_pid_is_ours", return_value=False):
                services = fresh.collect_service_declarations()

        assert services == [], "a dead service must not appear as a live node"

    def test_live_recovered_service_is_reported(self, registry, tmp_path):
        """Symmetry check — the liveness probe must not drop a service whose
        process genuinely survived, or the fix would hide every service."""
        checkpoint = tmp_path / "procs.json"
        original = _service_session()
        with patch("tools.process_registry.CHECKPOINT_PATH", checkpoint):
            registry._running[original.id] = original
            registry._write_checkpoint()
            fresh = ProcessRegistry()
            with patch.object(fresh, "_host_pid_is_ours", return_value=True), \
                 patch.object(ProcessRegistry, "_safe_host_start_time",
                              return_value=original.host_start_time):
                fresh.recover_from_checkpoint()
            with patch.object(fresh, "_host_pid_is_ours", return_value=True):
                services = fresh.collect_service_declarations()

        assert [s["label"] for s in services] == ["Compendium Dashboard"]


class TestBackwardCompatibility:
    def test_old_checkpoint_without_service_keys_recovers(self, tmp_path):
        """A checkpoint written by a build predating the service fields must
        recover as a plain background process, not raise."""
        checkpoint = tmp_path / "procs.json"
        checkpoint.write_text(json.dumps([{
            "session_id": "proc_legacy",
            "command": "sleep 999",
            "pid": 5150,
            "pid_scope": "host",
            "host_start_time": int(time.time()),
            "cwd": "/tmp",
            "started_at": time.time(),
            "task_id": "t1",
        }]))
        with patch("tools.process_registry.CHECKPOINT_PATH", checkpoint):
            fresh = ProcessRegistry()
            with patch.object(fresh, "_host_pid_is_ours", return_value=True):
                assert fresh.recover_from_checkpoint() == 1
                revived = fresh._running["proc_legacy"]
                assert revived.service_name == ""
                assert revived.service_inputs == []
                assert revived.service_relationships == []
                assert revived.service_source_files == []
                # Not a service, so it contributes no graph node.
                assert fresh.collect_service_declarations() == []
