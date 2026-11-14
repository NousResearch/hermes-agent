"""Isolated interactive OAuth attempts; publish one coherent grant only after success."""

import contextlib
import contextvars
from pathlib import Path
import tempfile

_STAGED_POOL = contextvars.ContextVar("_STAGED_POOL", default=None)


def staged_token_directory(pool_path: str) -> Path | None:
    """Redirect physical IO for this attempt only; authority still names the canonical pool."""
    staged = _STAGED_POOL.get()
    return staged[1] if staged is not None and staged[0] == pool_path else None


class OAuthAuthorization:
    """Keep token/client/metadata writes private until the caller commits its configuration.

    Other processes can keep refreshing the canonical pool throughout the browser interaction.
    Cancellation discards only this attempt; it never restores a potentially consumed token.
    """

    def __init__(self, storage, home: str, server_name: str, requested: dict):
        from tools.mcp_oauth import PoolAuthority
        self.original = storage
        self.home, self.server_name, self.requested = home, server_name, requested
        self.authority = PoolAuthority.capture(storage.pool_path, home, server_name, requested)

    def __enter__(self):
        from hermes_constants import mkdir_under_hermes_home
        from tools.mcp_oauth import HermesTokenStorage
        parent = Path(self.original.pool_path).parent
        mkdir_under_hermes_home(parent)
        self.directory = tempfile.TemporaryDirectory(prefix=".reauth-", dir=parent)
        self.context_token = _STAGED_POOL.set((self.original.pool_path, Path(self.directory.name)))
        self.storage = HermesTokenStorage(self.server_name, token_dir=parent)
        return self

    def __exit__(self, *exc):
        _STAGED_POOL.reset(self.context_token)
        self.directory.cleanup()

    @contextlib.contextmanager
    def publish(self):
        """Hold the canonical refresh fence through replacement AND the caller's config save."""
        from tools.mcp_oauth import (
            OAuthStorageDetachedError, _storage_fingerprint, advance_pool_epoch, hold_refresh_fence,
        )
        original = self.original
        with hold_refresh_fence(Path(original.pool_path)):
            verdict = self.authority.revocation()
            if verdict is not None:
                raise OAuthStorageDetachedError(verdict[0])
            backup = original.snapshot(fence_held=True)
            replacement = self.storage.snapshot()
            owner = original.quarantine()
            restored = False
            committed = False
            try:
                original.remove()
                original._restore_snapshot_files(replacement)
                yield
                committed = True
                original.quarantine(owner=owner, recoverable=True)
                if self.authority.saved_fingerprint != _storage_fingerprint(self.requested):
                    advance_pool_epoch(Path(original.pool_path))
                restored = True
            except BaseException:
                if not committed:
                    original._restore_snapshot_files(backup)
                    for path in (*original._state_paths(), original._cimd_rejected_path()):
                        if path.name not in backup:
                            path.unlink(missing_ok=True)
                    restored = True
                raise
            finally:
                if restored:
                    original.cancel_quarantine(owner)
