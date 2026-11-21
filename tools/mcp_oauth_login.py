"""Stage an explicit browser grant without deleting the active profile's OAuth state."""

import asyncio
from contextlib import contextmanager
import os
from pathlib import Path
import shutil
import stat
from tempfile import TemporaryDirectory, mkdtemp


def _read_state(storage):
    state = {}
    for path in (*storage._state_paths(), storage._cimd_rejected_path()):
        try:
            info = path.lstat()
        except FileNotFoundError:
            continue
        if not stat.S_ISREG(info.st_mode):
            raise OSError("OAuth state must be regular files")
        state[path.name] = path.read_bytes()
    return state


def require_settled_publication(storage):
    """An interrupted multi-file publish must be reconciled before any new auth."""
    if any(storage._tokens_path().parent.glob(f".oauth-login-{storage._server_name}.*")):
        raise RuntimeError("OAuth publication recovery required; preserved state must be reconciled before connecting")


def _sync_directory(path):
    # POSIX directory open/fsync failures are transaction failures, unlike the
    # general-purpose best-effort writer. Windows does not expose a directory
    # fsync through os.open: file fsync + atomic rename is its durability limit.
    if os.name == "nt":
        return
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _publication_paths(storage):
    # Publish the token last: it must not point at a client registration we have
    # not installed yet. Keep the refresh-lock inode owned by the existing fence.
    return (*storage._state_paths()[1:], storage._cimd_rejected_path(), storage._tokens_path())


def _prepare_publication(storage, candidate, original):
    from utils import atomic_write_bytes

    directory = Path(mkdtemp(prefix=f".oauth-login-{storage._server_name}.", dir=storage._tokens_path().parent))
    try:
        # All allocation, writes and fsyncs finish before touching a live file.
        # Rollback then needs only renames, even after ENOSPC starts persisting.
        for suffix, state in (("next", candidate), ("previous", original), ("rollback", original)):
            for name, raw in state.items():
                atomic_write_bytes(directory / (name + "." + suffix), raw, mode=0o600, fsync_dir=True)
        _sync_directory(directory)
        _sync_directory(directory.parent)
    except BaseException:
        shutil.rmtree(directory)
        raise
    return directory


def _publish(storage, candidate, original):
    directory = _prepare_publication(storage, candidate, original)
    recovery_required = False
    try:
        try:
            for path in _publication_paths(storage):
                if path.name in candidate:
                    os.replace(directory / (path.name + ".next"), path)
                else:
                    path.unlink(missing_ok=True)
            _sync_directory(directory.parent)
        except BaseException:
            # Restore every member, not just the first one that can be written.
            for path in _publication_paths(storage):
                try:
                    if path.name in original:
                        os.replace(directory / (path.name + ".rollback"), path)
                    else:
                        path.unlink(missing_ok=True)
                except OSError:
                    recovery_required = True
            try:
                _sync_directory(directory.parent)
            except OSError:
                recovery_required = True
            if recovery_required:
                raise RuntimeError(f"OAuth publication recovery required; private recovery files retained at {directory}") from None
            raise
    finally:
        if not recovery_required:
            shutil.rmtree(directory)


async def _snapshot(storage):
    from tools.mcp_oauth import acquire_refresh_fence, release_refresh_fence

    fd = await acquire_refresh_fence(storage._tokens_path())
    try:
        require_settled_publication(storage)
        return _read_state(storage)
    finally:
        release_refresh_fence(fd)


async def _promote(storage, candidate, original):
    from tools.mcp_oauth import acquire_refresh_fence, release_refresh_fence

    fd = await acquire_refresh_fence(storage._tokens_path())
    try:
        require_settled_publication(storage)
        if _read_state(storage) != original:
            raise RuntimeError("OAuth state changed during authorization; existing state preserved. Retry login.")
        _publish(storage, _read_state(candidate), original)
    finally:
        release_refresh_fence(fd)


@contextmanager
def staged_browser_oauth(name):
    """Yield a commit callback; failure/cancellation never publishes candidate state.

    Use the existing context-local profile binding so the probe's MCP-loop task
    and provider cache resolve to the candidate, not the live profile. Only cached
    discovery metadata is seeded: old tokens must not masquerade as a new grant.
    """
    from hermes_constants import get_hermes_home, reset_hermes_home_override, set_hermes_home_override
    from tools.mcp_oauth import HermesTokenStorage
    from tools.mcp_oauth_manager import get_manager
    from utils import atomic_write_bytes

    home = get_hermes_home()
    storage = HermesTokenStorage(name, hermes_home=home)
    original = asyncio.run(_snapshot(storage))
    manager = get_manager()
    with TemporaryDirectory(prefix="mcp-login-") as temporary:
        candidate_home = Path(temporary)
        candidate = HermesTokenStorage(name, hermes_home=candidate_home)
        metadata = original.get(storage._meta_path().name)
        if metadata is not None:
            atomic_write_bytes(candidate._meta_path(), metadata, mode=0o600)
        token = set_hermes_home_override(candidate_home)
        try:
            def commit():
                tokens = asyncio.run(candidate.get_tokens())
                if tokens is None or not tokens.access_token:
                    raise RuntimeError("No OAuth token was obtained; existing state preserved")
                asyncio.run(_promote(storage, candidate, original))
                manager.evict(name, hermes_home=home)

            yield commit
        finally:
            manager.evict(name, hermes_home=candidate_home)
            reset_hermes_home_override(token)
