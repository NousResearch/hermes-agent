"""Profile-scoped internal identities for process-local browser state."""
from __future__ import annotations

import hashlib
from typing import Optional

from hermes_constants import get_hermes_home, hermes_home_key


class BrowserTaskKey(str):
    """Opaque, typed browser key; raw task IDs and profile paths are not stringified into it."""

    def __new__(cls, profile_key: str, owner_task_id: str, *, local: bool = False):
        profile_digest = hashlib.sha256(profile_key.encode("utf-8", errors="surrogatepass")).hexdigest()[:32]
        task_digest = hashlib.sha256(owner_task_id.encode("utf-8", errors="surrogatepass")).hexdigest()[:32]
        value = f"browser-{profile_digest}-{task_digest}" + ("::local" if local else "")
        obj = super().__new__(cls, value)
        obj.profile_key = profile_key
        obj.owner_task_id = owner_task_id
        obj.local = local
        return obj

    def with_local(self, local: bool) -> "BrowserTaskKey":
        return self if self.local == local else BrowserTaskKey(self.profile_key, self.owner_task_id, local=local)


def browser_task_key(task_id: Optional[str] = None) -> BrowserTaskKey:
    """Qualify a raw caller ID once; typed internal keys survive cleanup without home context."""
    if isinstance(task_id, BrowserTaskKey):
        return task_id
    raw_id = "default" if task_id is None else str(task_id)
    home = str(get_hermes_home())
    return BrowserTaskKey(hermes_home_key(home), raw_id)
