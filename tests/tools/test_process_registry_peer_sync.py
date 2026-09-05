"""Cross-gateway background-process visibility.

A ``terminal(background=true)`` process spawned by the messaging gateway lives in
THAT process's ``ProcessRegistry``; Hermes Desktop runs its own gateway with its
own registry, so the row is invisible there. The shared crash-recovery checkpoint
(``~/.hermes/processes.json``) is the only thing both processes see, so these
tests pin the contract for importing a peer's rows from it:

* only live host PIDs whose recorded ``host_start_time`` still matches are
  adopted (a recycled PID must never be adopted, let alone signalled),
* a peer mirror is display/status only: it never re-publishes the checkpoint,
  never notifies, and never signals a PID this process does not own,
* the persisted output tail is bounded and redacted even when the user turned
  secret redaction off.
"""

import json
import os
import threading
import time
from unittest.mock import patch

import pytest

import agent.redact
import tools.process_registry as process_registry_module
import utils
from gateway.status import get_process_start_time
from tools.process_registry import (
    CHECKPOINT_OUTPUT_TAIL_CHARS,
    ProcessRegistry,
    ProcessSession,
)

LIVE_PID = os.getpid()
DEAD_PID = 999999999

# Redaction fixtures. Both are shaped to trip a real pattern (the ``ghp_`` prefix
# rule and the ENV-assignment rule) while spelling out on their face that they are
# not credentials, so a scanner hit here is obviously a test string.
FAKE_TOKEN = "ghp_00SYNTHETICNOTAREALTOKEN00"
FAKE_SECRET = "SYNTHETICNOTAREALSECRET"


@pytest.fixture()
def registry():
    return ProcessRegistry()


@pytest.fixture()
def checkpoint(tmp_path):
    """Redirect the shared checkpoint at a temp file and hand back the path."""
    path = tmp_path / "processes.json"
    with patch("tools.process_registry.CHECKPOINT_PATH", path):
        yield path


def _peer_entry(**overrides) -> dict:
    """A checkpoint entry as another gateway process would have written it."""
    entry = {
        "session_id": "proc_peerabc12345",
        "command": "npm run dev",
        "pid": LIVE_PID,
        "pid_scope": "host",
        "host_start_time": get_process_start_time(LIVE_PID),
        "cwd": "/srv/app",
        "started_at": time.time() - 30,
        "task_id": "t-telegram",
        "owner_task_id": "t-telegram",
        "session_key": "telegram:4242",
        "parent_session_id": "20260901_120000_abcdef",
        "notify_on_complete": True,
        "output_tail": "listening on :3000\n",
    }
    entry.update(overrides)
    return entry


def _write(checkpoint_path, *entries) -> None:
    checkpoint_path.write_text(json.dumps(list(entries)), encoding="utf-8")


def _ids(registry) -> set:
    return {row["session_id"] for row in registry.list_sessions()}


class TestPeerImport:
    def test_live_peer_process_becomes_a_visible_detached_row(self, registry, checkpoint):
        _write(checkpoint, _peer_entry())

        registry.sync_from_checkpoint()

        row = next(r for r in registry.list_sessions() if r["session_id"] == "proc_peerabc12345")
        assert row["status"] == "running"
        assert row["peer"] is True
        assert row["detached"] is True
        session = registry.get("proc_peerabc12345")
        # The durable identities the desktop matches ownership on must survive.
        assert session.session_key == "telegram:4242"
        assert session.parent_session_id == "20260901_120000_abcdef"
        assert session.output_buffer == "listening on :3000\n"

    def test_recycled_or_dead_pid_is_never_adopted(self, registry, checkpoint):
        _write(
            checkpoint,
            _peer_entry(session_id="proc_dead", pid=DEAD_PID),
            # Alive, but the kernel start time proves it is a different process now.
            _peer_entry(session_id="proc_recycled", host_start_time=1),
            # Legacy entry with no identity baseline at all: liveness alone is not proof.
            _peer_entry(session_id="proc_nobaseline", host_start_time=None),
            # In-sandbox PIDs mean nothing on this host.
            _peer_entry(session_id="proc_sandbox", pid_scope="sandbox"),
        )

        registry.sync_from_checkpoint()

        assert _ids(registry) == set()

    def test_locally_owned_rows_are_never_replaced_by_the_checkpoint(self, registry, checkpoint):
        mine = ProcessSession(
            id="proc_mine", command="真 command", session_key="desktop-key",
            started_at=time.time(), output_buffer="local output")
        registry._running[mine.id] = mine
        _write(checkpoint, _peer_entry(session_id="proc_mine", command="spoofed", session_key="attacker"))

        registry.sync_from_checkpoint()

        assert registry.get("proc_mine") is mine
        assert mine.command == "真 command"
        assert mine.session_key == "desktop-key"
        assert mine.output_buffer == "local output"

    def test_repeated_sync_refreshes_the_tail_without_duplicating_rows(self, registry, checkpoint):
        _write(checkpoint, _peer_entry())
        registry.sync_from_checkpoint()
        first = registry.get("proc_peerabc12345")

        _write(checkpoint, _peer_entry(output_tail="listening on :3000\nrequest served\n"))
        registry.sync_from_checkpoint()

        assert len(registry.list_sessions()) == 1
        assert registry.get("proc_peerabc12345") is first
        assert first.output_buffer == "listening on :3000\nrequest served\n"

    def test_peer_row_disappears_once_its_owner_stops_publishing_it(self, registry, checkpoint):
        _write(checkpoint, _peer_entry())
        registry.sync_from_checkpoint()
        assert _ids(registry) == {"proc_peerabc12345"}

        _write(checkpoint)  # owner finished the process and rewrote the checkpoint

        registry.sync_from_checkpoint()

        assert _ids(registry) == set()
        assert registry.get("proc_peerabc12345") is None


def _rows(checkpoint_path) -> dict:
    """``session_id`` → row, as currently published in the shared checkpoint."""
    return {
        str(row.get("session_id")): row
        for row in json.loads(checkpoint_path.read_text(encoding="utf-8"))
    }


def _track(registry, session_id: str, **kw) -> ProcessSession:
    """Register a live, locally-owned process on *registry*."""
    session = ProcessSession(
        id=session_id, command=kw.pop("command", "npm run dev"), pid=LIVE_PID,
        host_start_time=get_process_start_time(LIVE_PID), started_at=time.time(), **kw)
    registry._running[session_id] = session
    return session


# ``~/.hermes/processes.json`` has one writer per gateway PROCESS (messaging
# gateway, desktop backend, CLI) and they publish independently. A read-modify-
# write that treats the file as its own would silently delete every other live
# gateway's rows — and, since Desktop now renders peers from this file, delete
# the user's visible background work along with the crash-recovery record.
class TestMultiWriterCheckpoint:
    def test_interleaved_publishes_preserve_every_live_owners_rows(self, checkpoint):
        gateway_a, gateway_b = ProcessRegistry(), ProcessRegistry()
        _track(gateway_a, "proc_a")
        gateway_a._write_checkpoint()
        _track(gateway_b, "proc_b")
        gateway_b._write_checkpoint()

        gateway_a._write_checkpoint()

        assert set(_rows(checkpoint)) == {"proc_a", "proc_b"}

    def test_owner_completion_removes_only_that_owners_row(self, checkpoint):
        gateway_a, gateway_b = ProcessRegistry(), ProcessRegistry()
        _track(gateway_a, "proc_a1")
        _track(gateway_a, "proc_a2")
        gateway_a._write_checkpoint()
        _track(gateway_b, "proc_b1")
        gateway_b._write_checkpoint()

        gateway_a._running.pop("proc_a1")  # finished on A
        gateway_a._write_checkpoint()

        assert set(_rows(checkpoint)) == {"proc_a2", "proc_b1"}

        # B republishing must not resurrect the row A just retired.
        gateway_b._write_checkpoint()

        assert set(_rows(checkpoint)) == {"proc_a2", "proc_b1"}

    def test_rows_whose_tracked_process_is_gone_are_pruned(self, checkpoint):
        _write(
            checkpoint,
            _peer_entry(session_id="proc_dead", pid=DEAD_PID),
            _peer_entry(session_id="proc_live"),
            # An in-sandbox row is unreadable without its owner, and no owner
            # stamp means the writer is long gone.
            _peer_entry(session_id="proc_sandbox", pid_scope="sandbox"),
        )
        gateway_a = ProcessRegistry()
        _track(gateway_a, "proc_a")

        gateway_a._write_checkpoint()

        assert set(_rows(checkpoint)) == {"proc_a", "proc_live"}

    def test_concurrent_writers_never_lose_a_row(self, checkpoint):
        gateway_a, gateway_b = ProcessRegistry(), ProcessRegistry()
        _track(gateway_a, "proc_a")
        _track(gateway_b, "proc_b")
        start = threading.Barrier(2)

        def hammer(reg):
            start.wait(5)
            for _ in range(25):
                reg._write_checkpoint()

        threads = [threading.Thread(target=hammer, args=(reg,)) for reg in (gateway_a, gateway_b)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(30)

        assert set(_rows(checkpoint)) == {"proc_a", "proc_b"}

    def test_a_slow_write_never_resurrects_a_row_retired_meanwhile(self, checkpoint):
        """Two overlapping writes from ONE registry.

        ``_write_checkpoint`` snapshots the running set, then does the redaction and
        disk work outside that snapshot's lock. A writer still carrying an older
        snapshot must not land after the write that retired a row and put it back on
        disk. The inter-process lock cannot help here — both writers are in THIS
        process, so it only decides which of the two stale orderings wins.
        """
        registry = ProcessRegistry()
        _track(registry, "proc_retired")
        snapshot_taken = threading.Event()
        retired_write_landed = threading.Event()
        real_tail = process_registry_module._checkpoint_output_tail
        real_json_write = utils.atomic_json_write

        def stall_the_first_writer(text):
            # Reached once the stale snapshot exists but before anything is written.
            if not snapshot_taken.is_set():
                snapshot_taken.set()
                # Bounded: once serialized, the retiring write CANNOT land first, so
                # this waits out the bound instead of hanging the suite.
                retired_write_landed.wait(2.0)
            return real_tail(text)

        def note_empty_write(path, data, **kw):
            real_json_write(path, data, **kw)
            if not data:
                retired_write_landed.set()

        with patch.object(process_registry_module, "_checkpoint_output_tail", stall_the_first_writer), \
                patch.object(utils, "atomic_json_write", note_empty_write):
            stale_writer = threading.Thread(target=registry._write_checkpoint, daemon=True)
            stale_writer.start()
            assert snapshot_taken.wait(10), "the first writer never reached its snapshot"

            registry._running.clear()  # the process finished
            registry._write_checkpoint()

            stale_writer.join(10)
            assert not stale_writer.is_alive()

        assert set(_rows(checkpoint)) == set()

    def test_a_write_that_cannot_lock_is_skipped_rather_than_clobbering(self, checkpoint):
        """Publishing is a read-modify-write. Doing it unsynchronized is exactly how
        another gateway's row gets dropped, so an unavailable lock must cost us our
        own update — recoverable on the next write — not someone else's row."""
        gateway_b = ProcessRegistry()
        _track(gateway_b, "proc_b")
        gateway_b._write_checkpoint()

        gateway_a = ProcessRegistry()
        _track(gateway_a, "proc_a")
        with patch.object(process_registry_module, "_flock", side_effect=OSError("no locks available")):
            gateway_a._write_checkpoint()

        assert set(_rows(checkpoint)) == {"proc_b"}

        # Transient: the very next write publishes normally.
        gateway_a._write_checkpoint()

        assert set(_rows(checkpoint)) == {"proc_a", "proc_b"}

    def test_reads_still_work_when_the_lock_is_unavailable(self, checkpoint):
        """A read cannot destroy anything, and ``atomic_json_write`` already makes a
        torn view unlikely — so discovery degrades rather than going blind."""
        _write(checkpoint, _peer_entry())
        registry = ProcessRegistry()

        with patch.object(process_registry_module, "_flock", side_effect=OSError("no locks available")):
            assert registry.sync_from_checkpoint() == 1

    def test_startup_recovery_leaves_a_live_owners_rows_alone(self, checkpoint):
        gateway_b = ProcessRegistry()
        _track(gateway_b, "proc_b")
        gateway_b._write_checkpoint()

        # A second gateway booting must not adopt work another LIVE gateway is
        # already driving — two owners would both believe they may kill it.
        fresh = ProcessRegistry()

        assert fresh.recover_from_checkpoint() == 0
        assert set(_rows(checkpoint)) == {"proc_b"}

    def test_orphaned_rows_from_a_dead_owner_are_still_recovered(self, checkpoint):
        # Crash recovery is the whole reason this file exists: a row with no live
        # owner and a live process must still be adopted at startup.
        _write(checkpoint, _peer_entry(session_id="proc_orphan"))

        assert ProcessRegistry().recover_from_checkpoint() == 1

    def test_a_row_with_no_start_time_baseline_is_never_adopted(self, checkpoint):
        """Adoption puts a process in local, KILLABLE state, and a recovered session
        with no baseline makes ``_terminate_host_pid`` skip its identity check — so a
        recycled PID would be tree-killed. Recovery must be as strict as mirroring."""
        _write(checkpoint, _peer_entry(session_id="proc_unidentified", host_start_time=None))
        adopter = ProcessRegistry()

        assert adopter.recover_from_checkpoint() == 0
        assert adopter.get("proc_unidentified") is None
        # Nobody can act on it safely, so it is not left behind either.
        assert set(_rows(checkpoint)) == set()

    def test_adopting_an_orphan_leaves_exactly_one_row_for_it(self, checkpoint):
        _write(checkpoint, _peer_entry(session_id="proc_orphan"))
        adopter = ProcessRegistry()
        adopter.recover_from_checkpoint()

        adopter._write_checkpoint()

        published = json.loads(checkpoint.read_text(encoding="utf-8"))
        # The adopted row is republished under the NEW owner; the dead owner's
        # copy must not linger beside it as a second row for the same process.
        assert [row["session_id"] for row in published] == ["proc_orphan"]


class TestPeerMirrorIsDisplayOnly:
    def test_a_mirror_never_claims_the_owners_row(self, registry, checkpoint):
        _write(checkpoint, _peer_entry())
        registry.sync_from_checkpoint()
        published = _rows(checkpoint)["proc_peerabc12345"]

        registry._write_checkpoint()

        # Preserved verbatim: this process renders the row but never becomes its
        # author, so the owner's next update is still the authoritative one.
        assert _rows(checkpoint) == {"proc_peerabc12345": published}

    def test_our_own_published_rows_are_never_mirrored_back(self, registry, checkpoint):
        _track(registry, "proc_mine1234")
        registry._write_checkpoint()
        registry._running.clear()

        # The row is still on disk, stamped by us. Re-importing it would turn our
        # own process into a "peer" we then refuse to kill.
        assert registry.sync_from_checkpoint() == 0
        assert registry.get("proc_mine1234") is None

    def test_kill_refuses_to_signal_a_pid_this_process_does_not_own(self, registry, checkpoint):
        _write(checkpoint, _peer_entry())
        registry.sync_from_checkpoint()

        with patch.object(ProcessRegistry, "_terminate_host_pid") as terminate:
            result = registry.kill_process("proc_peerabc12345")

        terminate.assert_not_called()
        assert result["status"] == "error"
        assert "another" in result["error"].lower()
        # Still visible and still reported as running — we did not fake an exit.
        assert registry.get("proc_peerabc12345").exited is False

    def test_mirror_never_queues_a_completion_notification(self, registry, checkpoint):
        _write(checkpoint, _peer_entry(notify_on_complete=True))
        registry.sync_from_checkpoint()
        session = registry.get("proc_peerabc12345")

        # The owning gateway is the only party with a completion contract.
        assert session.notify_on_complete is False
        registry._move_to_finished(session)
        assert registry.completion_queue.empty()

    def test_mirror_is_excluded_from_local_lifecycle_bookkeeping(self, registry, checkpoint):
        _write(checkpoint, _peer_entry())
        registry.sync_from_checkpoint()

        # kill_all / scale-to-zero / session-reset all reason about processes
        # THIS gateway owns; a mirror must not answer for them.
        assert registry.has_any_active() is False
        assert registry.has_active_for_session("telegram:4242") is False
        assert registry.count_running() == 0
        with patch.object(ProcessRegistry, "_terminate_host_pid") as terminate:
            assert registry.kill_all() == 0
        terminate.assert_not_called()


class TestCheckpointOutputTail:
    def _running(self, registry, output: str, command: str = "npm run dev") -> ProcessSession:
        session = ProcessSession(
            id="proc_local1234", command=command, pid=LIVE_PID,
            host_start_time=get_process_start_time(LIVE_PID),
            started_at=time.time(), output_buffer=output)
        registry._running[session.id] = session
        return session

    def test_tail_is_published_for_peers_and_bounded(self, registry, checkpoint):
        self._running(registry, "x" * (CHECKPOINT_OUTPUT_TAIL_CHARS + 5_000) + "TAIL-END")

        registry._write_checkpoint()

        entry = json.loads(checkpoint.read_text(encoding="utf-8"))[0]
        assert entry["output_tail"].endswith("TAIL-END")
        assert len(entry["output_tail"]) <= CHECKPOINT_OUTPUT_TAIL_CHARS

    def test_secrets_are_masked_even_when_the_user_disabled_redaction(
        self, registry, checkpoint, monkeypatch):
        monkeypatch.setattr(agent.redact, "_REDACT_ENABLED", False)
        self._running(
            registry,
            f"connecting with {FAKE_TOKEN}\nDB_PASSWORD={FAKE_SECRET}\n",
            command=f"deploy --token {FAKE_TOKEN} DB_PASSWORD={FAKE_SECRET}")

        registry._write_checkpoint()

        entry = json.loads(checkpoint.read_text(encoding="utf-8"))[0]
        blob = entry["output_tail"] + entry["command"]
        assert FAKE_TOKEN not in blob
        # ENV/JSON assignments are secrets on this boundary too — the checkpoint
        # is a file on disk that a peer gateway renders for the user.
        assert FAKE_SECRET not in blob

    def test_a_secret_straddling_the_tail_cut_is_still_masked(self, registry, checkpoint):
        """Slicing before redacting decapitates ``DB_PASSWORD=`` at the cut, so the
        pattern no longer matches and the value is published verbatim."""
        value = FAKE_SECRET
        # Place the cut exactly between the assignment's key and its value: the
        # published tail then begins at the raw secret.
        after = "z" * (CHECKPOINT_OUTPUT_TAIL_CHARS - len(value) - 1)
        self._running(registry, "y" * 20_000 + f"DB_PASSWORD={value}\n" + after)

        registry._write_checkpoint()

        entry = json.loads(checkpoint.read_text(encoding="utf-8"))[0]
        assert value not in entry["output_tail"]
        assert len(entry["output_tail"]) <= CHECKPOINT_OUTPUT_TAIL_CHARS

    def test_live_output_checkpointing_is_throttled_but_still_lands(self, registry, checkpoint):
        session = self._running(registry, "")
        writes = []
        done = threading.Event()

        def record(*_a, **_kw):
            writes.append(session.output_buffer)
            done.set()

        with patch.object(ProcessRegistry, "_write_checkpoint", record):
            for i in range(200):
                registry._ingest_output(session, f"line {i}\n")
            # Per-chunk writes would be 200 disk rewrites for one burst.
            assert len(writes) < 20
            burst_writes = len(writes)
            done.clear()
            # The tail of a burst must not be lost to the throttle window.
            assert done.wait(10), "throttled tail was never flushed"

        assert len(writes) > burst_writes
        assert writes[-1].endswith("line 199\n")
