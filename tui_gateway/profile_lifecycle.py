"""Profile-generation fence for retained TUI/Desktop session state."""

from __future__ import annotations

from collections.abc import Callable, MutableMapping
from contextlib import contextmanager
from pathlib import Path
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


class ProfileLifecycleFence:
    """Track retired paths and incarnations inside one gateway process."""

    def __init__(self) -> None:
        self.retired_homes: set[str] = set()
        self.retired_incarnations: set[tuple[str, str]] = set()

    @staticmethod
    def key(profile_home: Path | str) -> str:
        try:
            return str(Path(profile_home).resolve())
        except OSError:
            return str(Path(profile_home))

    def capture(self, profile_home: Path | str | None) -> str | None:
        if profile_home is None:
            return None
        return ensure_profile_incarnation(profile_home)

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
            key = self.key(home)
            if key in self.retired_homes or (
                expected_incarnation is not None
                and (key, expected_incarnation) in self.retired_incarnations
            ):
                raise FileNotFoundError(
                    f"Profile incarnation is stale or home is unavailable: {home}"
                )
            yield home

    def rejected(
        self,
        profile_home: Path | str,
        expected_incarnation: str | None = None,
        *,
        require_incarnation: bool = False,
    ) -> bool:
        key = self.key(profile_home)
        if key in self.retired_homes:
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
            if require_incarnation:
                return True
            return False
        if (key, expected_incarnation) in self.retired_incarnations:
            return True
        return not profile_incarnation_matches(profile_home, expected_incarnation)

    def retire(
        self,
        profile_home: Path | str,
        incarnation: str | None = None,
    ) -> None:
        key = self.key(profile_home)
        self.retired_homes.add(key)
        if incarnation is None:
            try:
                incarnation = read_profile_incarnation(profile_home)
            except (OSError, RuntimeError):
                incarnation = None
        if incarnation is not None:
            self.retired_incarnations.add((key, incarnation))

    def allow(
        self,
        profile_home: Path | str,
        incarnation: str | None = None,
    ) -> None:
        key = self.key(profile_home)
        self.retired_homes.discard(key)
        if incarnation is None:
            try:
                incarnation = read_profile_incarnation(profile_home)
            except (OSError, RuntimeError):
                incarnation = None
        # Rollback of a failed delete admits the unchanged generation.  A
        # same-name recreate has a fresh token, so its call leaves the retired
        # predecessor tuple intact.
        if incarnation is not None:
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
        try:
            target = Path(profile_home).resolve()
        except OSError:
            target = Path(profile_home)
        try:
            resolved_launch_home = Path(launch_home).resolve()
        except OSError:
            resolved_launch_home = Path(launch_home)
        retiring_launch_home = target == resolved_launch_home

        def belongs(session: dict) -> bool:
            if retiring_launch_home and not session.get("profile_home"):
                return True
            raw = session.get("profile_home")
            if not raw:
                return False
            try:
                return Path(raw).resolve() == target
            except OSError:
                return Path(raw) == target

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
