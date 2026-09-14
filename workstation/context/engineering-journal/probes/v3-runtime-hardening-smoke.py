"""Native process-boundary smoke for the V3 runtime contracts.

This probe deliberately uses only explicit temporary paths. It proves the
independent supervisor and cross-process session ownership on Windows; it does
not claim Desktop/Electron integration for the Python-only V3 contracts.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workstation.runtime import ExecutionStatus, EvidenceState
from workstation.session_lifecycle import SessionLease
from workstation.supervisor import RuntimeSupervisor, SupervisorState


def main() -> int:
    repo_root = REPO_ROOT
    with tempfile.TemporaryDirectory(prefix="HermesV3RuntimeSmoke-") as raw_root:
        root = Path(raw_root)
        supervisor = RuntimeSupervisor.from_command(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            state_path=root / "supervisor.json",
            checkpoint_dir=root / "checkpoints",
        )
        if not supervisor.start() or not supervisor.health().healthy:
            raise RuntimeError("independent supervisor did not start a healthy child")
        print("V3_SUPERVISOR_START_PASS", supervisor.state.value)

        runtime = supervisor.runtime
        if runtime is None or runtime.process is None:
            raise RuntimeError("supervisor did not expose the child process")
        runtime.process.terminate()
        runtime.process.wait(timeout=5)
        time.sleep(0.02)
        if not supervisor.watchdog_step().healthy or supervisor.state != SupervisorState.RUNNING:
            raise RuntimeError("supervisor did not recover the stopped child")
        print("V3_SUPERVISOR_RECOVERY_PASS", supervisor.state.value)
        supervisor.stop()

        lease_path = root / "session.lock"
        lease = SessionLease(lease_path, owner_id="parent")
        lease.acquire()
        child_code = (
            "import sys; from pathlib import Path; "
            "from workstation.session_lifecycle import SessionLease; "
            "SessionLease(Path(sys.argv[1]), owner_id='child').acquire()"
        )
        contender = subprocess.run(
            [sys.executable, "-c", child_code, str(lease_path)],
            cwd=repo_root,
            capture_output=True,
            text=True,
        )
        lease.release()
        if contender.returncode == 0 or "already owned" not in contender.stderr:
            raise RuntimeError("cross-process session ownership was not rejected")
        print("V3_SESSION_OWNERSHIP_PASS")

        evidence = EvidenceState("task-smoke", "session-smoke")
        evidence.transition(ExecutionStatus.RUNNING, now="2026-09-11T10:00:00+00:00")
        if evidence.status != ExecutionStatus.STALLED:
            raise RuntimeError("running without evidence was not stalled")
        evidence.add_evidence("process_id", "child-1", ttl_seconds=1, now="2026-09-11T10:00:00+00:00")
        evidence.reconcile(now="2026-09-11T10:00:02+00:00")
        if evidence.status != ExecutionStatus.STALLED:
            raise RuntimeError("stale evidence did not degrade state")
        print("V3_EVIDENCE_RECONCILIATION_PASS")

    print("V3_RUNTIME_HARDENING_CLASSIFICATION=VALIDATED_CONTRACT_BOUNDARY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
