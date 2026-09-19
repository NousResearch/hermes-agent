"""What the table needs from an operation, and how an owner is named."""

from __future__ import annotations

import threading
from typing import Any, Dict, NamedTuple, Optional, Protocol, runtime_checkable

from hermes_constants import get_process_hermes_home, hermes_home_key


class Owner(NamedTuple):
    """The profile an operation was opened under and the key inside it: a session key for a card
    or a prompt, a server name for a settings-page OAuth flow."""

    profile: str
    key: str

    @classmethod
    def of(cls, key: str, *, profile_home: Optional[str] = None) -> "Owner":
        """A session record names its profile home only for a non-default profile; ``None`` means
        the process home, which is what the tool thread's turn override resolves to."""
        return cls(hermes_home_key(profile_home or get_process_hermes_home()), key)

    @classmethod
    def current(cls, key: str) -> "Owner":
        """The owner as seen from the calling thread: the turn's profile override, or the launch home."""
        return cls(hermes_home_key(), key)


@runtime_checkable
class Operation(Protocol):
    op_id: str
    # ``None`` until ``Operations.open`` stamps the opening thread's profile and the operation's key.
    owner: Optional[Owner]
    key: str
    deadline_at: float
    wake: threading.Event

    @property
    def settled(self) -> bool: ...

    @property
    def settled_by(self) -> Optional[str]: ...

    def settle(self, reason: str) -> bool:
        """Compare-and-set: the first caller freezes the result and returns True."""
        ...

    def snapshot(self) -> Dict[str, Any]:
        """The payload a reconnecting client restores the operation from."""
        ...
