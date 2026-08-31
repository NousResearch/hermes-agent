"""Exclusive leases for the checkpoint store, including deletion of the store itself."""

import logging
import os
import threading
import time
from pathlib import Path
from typing import BinaryIO, Dict, Optional

logger = logging.getLogger(__name__)

# The shared object database, refs, indexes, metadata, and ledgers form one
# transaction domain.  Git protects individual files, but it cannot protect a
# just-created, not-yet-referenced object graph from another process running
# ``gc --prune=now``.  Interactive checkpoints must not stall a tool call for
# long; explicit maintenance may wait for an in-flight checkpoint to finish.
_LOCK_POLL_INTERVAL = 0.05
_IS_WINDOWS = os.name == "nt"


class CheckpointStoreBusy(RuntimeError):
    """The checkpoint-store writer lease could not be acquired in time."""


class _TransactionLockState:
    def __init__(self) -> None:
        self.gate = threading.RLock()
        self.depth = 0
        self.handle: Optional[BinaryIO] = None


_transaction_lock_pid = os.getpid()
_transaction_lock_guard = threading.Lock()
_transaction_lock_states: Dict[str, _TransactionLockState] = {}


def _transaction_lock_path(base: Path) -> Path:
    """Keep the lease outside ``base`` so ``clear_all`` cannot unlink it."""
    base = base.expanduser().resolve()
    return base.parent / f".{base.name}.transaction.lock"


def _transaction_lock_state(lock_path: Path) -> _TransactionLockState:
    """Return process-local reentrant state, resetting inherited fork state."""
    global _transaction_lock_pid, _transaction_lock_guard, _transaction_lock_states
    pid = os.getpid()
    if pid != _transaction_lock_pid:
        # A child must never reuse an inherited descriptor or an RLock whose
        # owning thread existed only in the parent.
        for inherited in _transaction_lock_states.values():
            if inherited.handle is not None:
                # Close, but do not unlock: flock state belongs to the shared
                # open-file description and an unlock here would release the
                # parent's still-live lease.
                try:
                    inherited.handle.close()
                except OSError:
                    pass
        _transaction_lock_pid = pid
        _transaction_lock_guard = threading.Lock()
        _transaction_lock_states = {}
    key = os.path.normcase(str(lock_path))
    with _transaction_lock_guard:
        return _transaction_lock_states.setdefault(key, _TransactionLockState())


def _try_lock_file(handle) -> bool:
    handle.seek(0)
    if _IS_WINDOWS:
        import msvcrt
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            return True
        except OSError:
            return False
    import fcntl
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except (BlockingIOError, OSError):
        return False


def _unlock_file(handle) -> None:
    handle.seek(0)
    if _IS_WINDOWS:
        import msvcrt
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class _CheckpointTransaction:
    """Cross-process exclusive, process/thread-reentrant writer lease."""

    def __init__(self, base: Path, timeout: float, operation: str) -> None:
        self.lock_path = _transaction_lock_path(base)
        self.timeout = max(0.0, float(timeout))
        self.operation = operation
        self.entered = False

    def __enter__(self):
        self.pid = os.getpid()
        self.state = _transaction_lock_state(self.lock_path)
        deadline = time.monotonic() + self.timeout
        candidate_handle = None
        if not self.state.gate.acquire(timeout=max(0.0, deadline - time.monotonic())):
            raise CheckpointStoreBusy(
                f"checkpoint store busy during {self.operation} "
                f"(waited {self.timeout:g}s)"
            )
        try:
            if self.state.depth == 0:
                self.lock_path.parent.mkdir(parents=True, exist_ok=True)
                handle = open(self.lock_path, "a+b", buffering=0)
                candidate_handle = handle
                # Windows byte-range locking requires the byte to exist.
                handle.seek(0, os.SEEK_END)
                if handle.tell() == 0:
                    handle.write(b"\0")
                    handle.flush()
                while not _try_lock_file(handle):
                    if time.monotonic() >= deadline:
                        handle.close()
                        raise CheckpointStoreBusy(
                            f"checkpoint store busy during {self.operation} "
                            f"(waited {self.timeout:g}s)"
                        )
                    time.sleep(min(_LOCK_POLL_INTERVAL, max(0.0, deadline - time.monotonic())))
                self.state.handle = handle
            self.state.depth += 1
            self.entered = True
            return self
        except OSError as exc:
            if candidate_handle is not None and not candidate_handle.closed:
                candidate_handle.close()
            self.state.gate.release()
            raise CheckpointStoreBusy(
                f"cannot acquire checkpoint store lease during {self.operation}: {exc}"
            ) from exc
        except BaseException:
            if candidate_handle is not None and not candidate_handle.closed:
                candidate_handle.close()
            self.state.gate.release()
            raise

    def __exit__(self, exc_type, exc, tb) -> None:
        if not self.entered:
            return
        if self.pid != os.getpid():
            # An inherited context must not unlock the parent's open-file description.
            _transaction_lock_state(self.lock_path)
            self.entered = False
            return
        try:
            self.state.depth -= 1
            if self.state.depth == 0:
                handle = self.state.handle
                self.state.handle = None
                if handle is not None:
                    try:
                        _unlock_file(handle)
                    except OSError as unlock_error:
                        logger.warning(
                            "Could not release checkpoint store lease %s: %s",
                            self.lock_path, unlock_error,
                        )
                    finally:
                        handle.close()
        finally:
            self.entered = False
            self.state.gate.release()
