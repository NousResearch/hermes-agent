"""Profile-generation fence for retained TUI/Desktop session state."""

from __future__ import annotations

from collections.abc import Callable, MutableMapping
from contextlib import contextmanager
from pathlib import Path
import threading
from typing import Any, Iterator

from hermes_cli.profile_incarnation import (
    ensure_profile_incarnation,
    profile_incarnation_lease,
    profile_incarnation_matches,
    read_profile_incarnation,
)
from hermes_constants import (
    named_profile_home_is_unavailable,
    profile_deletion_marker_path,
)


def _resolved(profile_home: Path | str) -> Path:
    try:
        return Path(profile_home).resolve()
    except OSError:
        return Path(profile_home)


def _current_incarnation(profile_home: Path | str) -> str | None:
    try:
        return read_profile_incarnation(profile_home)
    except (OSError, RuntimeError):
        return None


class ProfileLifecycleFence:
    """Remember generations retired inside one gateway process.

    Disk state is the cross-process authority: retirement tombstones a home
    before its sessions are torn down, and only publication of a generation
    (with its own incarnation token) or a rollback clears that tombstone. This
    set only ADDS rejection of retired ``(home, incarnation)`` pairs, so no
    check ever needs the profile lease: callers hold ``_sessions_lock``, which
    lease holders (publish, delete) take in turn.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.retired_incarnations: set[tuple[str, str]] = set()

    @staticmethod
    def key(profile_home: Path | str) -> str:
        return str(_resolved(profile_home))

    def _is_retired(self, profile_home: Path | str, incarnation: str | None) -> bool:
        # key() costs a resolve(); with nothing retired in this process no key can match.
        if incarnation is None or not self.retired_incarnations:
            return False
        key = self.key(profile_home)
        with self._lock:
            return (key, incarnation) in self.retired_incarnations

    def capture(self, profile_home: Path | str | None) -> str | None:
        if profile_home is None:
            return None
        # The lazy backfill refuses tombstoned homes, so it cannot revive a retirement.
        incarnation = ensure_profile_incarnation(profile_home)
        if self._is_retired(profile_home, incarnation):
            raise FileNotFoundError(f"Profile incarnation is retired: {profile_home}")
        return incarnation

    @contextmanager
    def lease(
        self,
        profile_home: Path | str,
        expected_incarnation: str | None,
        *,
        require_incarnation: bool = True,
    ) -> Iterator[Path]:
        """Bind one profile resource without crossing a mutation boundary."""
        with profile_incarnation_lease(
            profile_home,
            expected_incarnation,
            require_incarnation=require_incarnation,
        ) as home:
            if self._is_retired(home, expected_incarnation):
                raise FileNotFoundError(f"Profile incarnation is retired: {home}")
            yield home

    def rejected(
        self,
        profile_home: Path | str,
        expected_incarnation: str | None = None,
        *,
        require_incarnation: bool = False,
    ) -> bool:
        if self._is_retired(profile_home, expected_incarnation):
            return True
        try:
            named_marker = profile_deletion_marker_path(profile_home)
            if named_profile_home_is_unavailable(profile_home):
                return True
        except Exception:
            return True
        if named_marker is None:
            return False
        if expected_incarnation is None:
            return require_incarnation
        return not profile_incarnation_matches(profile_home, expected_incarnation)

    def retire(
        self,
        profile_home: Path | str,
        incarnation: str | None = None,
    ) -> None:
        if incarnation is None:
            incarnation = _current_incarnation(profile_home)
        # A tokenless (legacy) generation is fenced by its tombstone alone.
        if incarnation is not None:
            key = self.key(profile_home)
            with self._lock:
                self.retired_incarnations.add((key, incarnation))

    def allow(
        self,
        profile_home: Path | str,
        incarnation: str | None = None,
    ) -> None:
        if incarnation is None:
            incarnation = _current_incarnation(profile_home)
        # Rollback of a failed delete admits the unchanged generation.  A
        # same-name recreate has a fresh token, so its call leaves the retired
        # predecessor tuple intact.
        if incarnation is not None:
            key = self.key(profile_home)
            with self._lock:
                self.retired_incarnations.discard((key, incarnation))

    def retire_sessions(
        self,
        profile_home: Path | str,
        incarnation: str | None,
        *,
        launch_home: Path | str,
        sessions: MutableMapping[str, dict],
        sessions_lock: Any,
        close_session: Callable[[str], bool],
        close_launch_db: Callable[[], int],
    ) -> int:
        """Fence a profile and tear down every retained in-process session."""
        target = _resolved(profile_home)
        retiring_launch_home = target == _resolved(launch_home)

        def belongs(session: dict) -> bool:
            if retiring_launch_home and not session.get("profile_home"):
                return True
            raw = session.get("profile_home")
            if not raw:
                return False
            return _resolved(raw) == target

        with sessions_lock:
            self.retire(target, incarnation)
            session_ids = [sid for sid, session in sessions.items() if belongs(session)]

        retired = 0
        unsettled: list[str] = []
        for sid in session_ids:
            if close_session(sid):
                retired += 1
            else:
                unsettled.append(sid)
        if retiring_launch_home:
            retired += close_launch_db()
        if unsettled:
            raise RuntimeError(
                "Profile still has active session turn(s): " + ", ".join(unsettled)
            )
        return retired
