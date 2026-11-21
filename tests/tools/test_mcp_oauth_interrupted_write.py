"""A token write interrupted between fsync and rename must not lose a rotated refresh token.

The authorization server rotates the refresh token when it answers the refresh POST, before
``set_tokens`` persists the new pair. A process that dies after staging the temp file but before
renaming it (``os._exit`` skips ``_atomic_write``'s cleanup) leaves ``<server>.json`` holding the
already-rotated predecessor; replaying it makes a rotating server revoke the whole grant.
"""

import asyncio
import json
import os
import tempfile
import time
from unittest.mock import MagicMock

import pytest

pytest.importorskip("mcp.client.auth.oauth2", reason="MCP SDK 1.26.0+ required for OAuth support")

from tools.mcp_oauth import HermesTokenStorage

ISSUER = "https://as.example.com"


def _save(storage: HermesTokenStorage, refresh: str) -> None:
    token = MagicMock()
    token.model_dump.return_value = {"access_token": f"at-{refresh}", "token_type": "Bearer",
                                     "expires_in": 900, "refresh_token": refresh}
    storage.bind_issuer(ISSUER)
    asyncio.run(storage.set_tokens(token))


def _age(path, seconds: float) -> None:
    then = time.time() - seconds
    os.utime(path, (then, then))


def _interrupted_write(directory, stem: str, payload: dict, *, age: float):
    """What a process killed between fsync and rename leaves behind: the mkstemp-named temp file."""
    fd, name = tempfile.mkstemp(dir=directory, prefix=f".{stem}_", suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(payload, f)
    _age(name, age)
    return name


def _pair(refresh: str, issuer: str | None = ISSUER) -> dict:
    data = {"access_token": f"at-{refresh}", "token_type": "Bearer", "expires_in": 900,
            "expires_at": time.time() + 900, "refresh_token": refresh}
    if issuer is not None:
        data["hermes_issuer"] = issuer
    return data


def test_get_tokens_adopts_the_newer_interrupted_write(tmp_path):
    storage = HermesTokenStorage("srv", hermes_home=tmp_path)
    _save(storage, "rotated-away")
    tokens = tmp_path / "mcp-tokens" / "srv.json"
    _age(tokens, 120)
    tmp = _interrupted_write(tokens.parent, "srv", _pair("live"), age=60)

    loaded = asyncio.run(storage.get_tokens())

    assert loaded is not None and loaded.refresh_token == "live"
    assert json.loads(tokens.read_text())["refresh_token"] == "live"
    assert not os.path.exists(tmp)
    assert storage.loaded_issuer == ISSUER


def _truncate(tmp, tokens) -> None:
    with open(tmp, "w") as f:
        f.write('{"access_token": "at", "refresh_')
    _age(tmp, 60)


# case -> (temp-file stem, temp-file issuer, temp-file age, after-staging step,
#          tokens still loaded, temp file left in place)
_NOT_ADOPTED = {
    # superseded by a later complete write: stale credential copy, deleted
    "older_than_saved": ("srv", ISSUER, 300, None, True, False),
    # may be a live writer between mkstemp and rename
    "too_young": ("srv", ISSUER, 0, None, True, True),
    # died mid-write: nothing complete to adopt
    "partial_json": ("srv", ISSUER, 60, _truncate, True, True),
    # never hand a refresh token to a different authorization server
    "other_issuer": ("srv", "https://evil.example.com", 60, None, True, True),
    # never resurrect tokens after an invalidation deleted the saved file
    "no_saved_file": ("srv", ISSUER, 60, lambda tmp, tokens: tokens.unlink(), False, True),
    # ``srv_x``'s temp file is not ``srv``'s
    "other_server_prefix": ("srv_x", ISSUER, 60, None, True, True),
}


@pytest.mark.parametrize("case", sorted(_NOT_ADOPTED))
def test_interrupted_write_is_not_adopted(tmp_path, case):
    stem, issuer, age, after, still_loaded, tmp_kept = _NOT_ADOPTED[case]
    storage = HermesTokenStorage("srv", hermes_home=tmp_path)
    _save(storage, "saved")
    tokens = tmp_path / "mcp-tokens" / "srv.json"
    _age(tokens, 120)
    tmp = _interrupted_write(tokens.parent, stem, _pair("from-tmp", issuer=issuer), age=age)
    if after is not None:
        after(tmp, tokens)

    loaded = asyncio.run(storage.get_tokens())

    assert (loaded.refresh_token if loaded else None) == ("saved" if still_loaded else None)
    assert os.path.exists(tmp) == tmp_kept


def test_remove_clears_interrupted_writes(tmp_path):
    """Logging out must not leave a credential copy that a later load could adopt."""
    storage = HermesTokenStorage("srv", hermes_home=tmp_path)
    _save(storage, "saved")
    tokens = tmp_path / "mcp-tokens" / "srv.json"
    tmp = _interrupted_write(tokens.parent, "srv", _pair("from-tmp"), age=60)

    storage.remove()

    assert not os.path.exists(tmp)
    assert asyncio.run(storage.get_tokens()) is None
