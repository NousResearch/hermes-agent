"""
tools/skill_locks.py — Distributed write lock for team-scope skills.

Context
-------
In SaaS mode, multiple Hermes agent workers on the same team may attempt to
edit the same team skill concurrently.  Personal-scope writes are always
isolated per user (no collision possible) and never go through this module.

Mechanism
---------
DynamoDB conditional writes as a distributed lock.  Each skill key gets a lock
row with a TTL:

    skill_key  (PK)  — canonical S3 key prefix, e.g.
                        "team/slack/TTEAM01/skills/my-skill"
    worker_id         — random UUID identifying the lock owner
    ttl               — Unix epoch seconds; DynamoDB expires the row after
                        this time AND our conditional write checks it directly
                        (DDB TTL deletion lags up to 48h, but the check is
                        instantaneous because we compare against time.time())

acquire_skill_lock(): conditional PutItem — succeeds only when the row is
    absent OR its TTL has already passed.  Returns True on success, False when
    another worker holds a live lock.

release_skill_lock(): conditional DeleteItem — only the lock owner (matching
    worker_id) may release it.  Silently succeeds if the row is already gone
    (crash-recovery scenario: DDB TTL or a previous explicit release removed it).

Table
-----
Table name defaults to "hermes-skill-locks" and is configurable via
HERMES_SKILL_LOCKS_TABLE.  Provisioned via
infra/terraform/dynamodb-skill-locks/main.tf.

Degradation
-----------
- boto3 not installed: ImportError caught at import; re-raised at call time
  with a clear message.  Local dev (non-SaaS) is never broken.
- DynamoDB unavailable (network, table not created yet): acquire returns False
  with a logged warning; the caller returns the standard retry message.
  We never silently allow the write — fail closed, not open.
- Table not yet provisioned (ResourceNotFoundException): treated as lock
  unavailable; callers should NOT proceed with team writes — the lock
  infrastructure is required.  This forces the operator to apply the Terraform
  before SaaS team writes go live.

Usage
-----
    from tools.skill_locks import acquire_skill_lock, release_skill_lock

    skill_key = f"{identity.team_scope}/skills/{name}"
    worker_id = str(uuid.uuid4())

    if not acquire_skill_lock(skill_key, worker_id):
        return {"error": f"Skill '{name}' is currently being edited. Please retry."}
    try:
        # safe to do the S3 read-modify-write here
        write_skill(name, content, scope="team", identity=identity)
    finally:
        release_skill_lock(skill_key, worker_id)

Concurrency model
-----------------
This is intentionally simple: no Redlock, no Zookeeper.  DynamoDB conditional
writes are sufficient for Hermes-scale concurrency (agent turns last seconds,
not microseconds, and team-skill edits are rare compared to personal writes).
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from typing import Optional

logger = logging.getLogger(__name__)

# Configurable via env var so tests can point at a local DynamoDB mock.
_TABLE_NAME = os.environ.get("HERMES_SKILL_LOCKS_TABLE", "hermes-skill-locks")

# Default lock TTL in seconds.  Long enough to cover the entire skill edit
# round-trip (S3 read + LLM content generation + S3 write) with headroom.
DEFAULT_TTL_SECONDS: int = 30

# Lazy import — boto3 is a SaaS-only dep; local mode must never fail to import.
_boto3_import_error: Optional[ImportError] = None
try:
    import boto3
    from botocore.exceptions import ClientError
except ImportError as _exc:
    boto3 = None  # type: ignore[assignment]
    ClientError = Exception  # type: ignore[assignment,misc]
    _boto3_import_error = _exc


def _require_boto3() -> None:
    """Raise a clear ImportError if boto3 is not installed."""
    if _boto3_import_error is not None:
        raise ImportError(
            "boto3 is required for distributed skill locking in SaaS mode. "
            f"Original error: {_boto3_import_error}"
        ) from _boto3_import_error


def _lock_table():
    """Return a DynamoDB Table resource for the lock table.

    Callers must call _require_boto3() first.
    """
    ddb = boto3.resource("dynamodb")
    return ddb.Table(_TABLE_NAME)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def acquire_skill_lock(
    skill_key: str,
    worker_id: str,
    ttl_seconds: int = DEFAULT_TTL_SECONDS,
) -> bool:
    """Attempt to acquire an exclusive write lock on *skill_key*.

    Parameters
    ----------
    skill_key:
        Canonical identifier for the locked resource — typically the S3 prefix
        of the skill being written, e.g. ``"team/slack/TTEAM01/skills/foo"``.
    worker_id:
        Unique string (UUID) identifying the lock owner.  Used to enforce that
        only the owner can release the lock.
    ttl_seconds:
        Seconds until the lock auto-expires via DynamoDB TTL.  Defaults to 30.
        Set higher if the write path is slower (e.g. includes a blocking LLM
        call).

    Returns
    -------
    True  — lock acquired; the caller may proceed with the team-scope write.
    False — lock NOT acquired; another worker holds a live lock.  The caller
            should return a friendly retry error rather than proceeding.

    Raises
    ------
    ImportError
        When boto3 is not installed (local dev without SaaS deps).
    """
    _require_boto3()
    expires_at = int(time.time()) + ttl_seconds

    try:
        table = _lock_table()
        table.put_item(
            Item={
                "skill_key": skill_key,
                "worker_id": worker_id,
                "ttl": expires_at,
            },
            # Succeed when:
            #   1. The lock row doesn't exist yet (attribute_not_exists), OR
            #   2. The existing lock has already expired (#ttl < :now).
            #
            # This is an atomic compare-and-swap — no separate read step needed.
            ConditionExpression=(
                "attribute_not_exists(skill_key) OR #t < :now"
            ),
            ExpressionAttributeNames={"#t": "ttl"},
            ExpressionAttributeValues={":now": int(time.time())},
        )
        logger.debug(
            "skill_lock: acquired lock for %r (worker=%s, expires=%d)",
            skill_key, worker_id, expires_at,
        )
        return True

    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "")  # type: ignore[attr-defined]
        if code == "ConditionalCheckFailedException":
            logger.info(
                "skill_lock: lock contention on %r (worker %s could not acquire)",
                skill_key, worker_id,
            )
            return False
        # Any other DDB error (network, table missing, throttling) → fail closed:
        # do NOT silently allow the write, but log the error so ops can diagnose.
        logger.error(
            "skill_lock: DynamoDB error acquiring lock for %r: %s (%s)",
            skill_key, exc, code,
        )
        raise


def release_skill_lock(skill_key: str, worker_id: str) -> None:
    """Release a lock previously acquired by *worker_id*.

    Only the lock owner (identified by *worker_id*) may release.  If the lock
    has already been released (e.g. TTL expired and another worker acquired it),
    the conditional delete fails silently — this is the correct behaviour for
    crash recovery.

    Parameters
    ----------
    skill_key:
        Same value used in ``acquire_skill_lock``.
    worker_id:
        Same UUID used in ``acquire_skill_lock``.

    Raises
    ------
    ImportError
        When boto3 is not installed.
    """
    _require_boto3()

    try:
        table = _lock_table()
        table.delete_item(
            Key={"skill_key": skill_key},
            # Only delete if this worker still owns the lock.
            # If the TTL expired and another worker took the lock,
            # their worker_id won't match — we leave their lock intact.
            ConditionExpression="worker_id = :wid",
            ExpressionAttributeValues={":wid": worker_id},
        )
        logger.debug(
            "skill_lock: released lock for %r (worker=%s)", skill_key, worker_id
        )

    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "")  # type: ignore[attr-defined]
        if code == "ConditionalCheckFailedException":
            # The lock was already released (TTL expiry or double-release).
            # This is expected in crash-recovery paths — log at DEBUG, not ERROR.
            logger.debug(
                "skill_lock: lock for %r already released or expired (worker=%s) — ignoring",
                skill_key, worker_id,
            )
            return
        # Any other error: log and swallow — release is best-effort.
        # Letting this propagate would skip the ``finally`` block in callers,
        # hiding the original write result.
        logger.error(
            "skill_lock: DynamoDB error releasing lock for %r: %s (%s)",
            skill_key, exc, code,
        )


@contextmanager
def team_skill_lock(skill_key: str, worker_id: str, ttl_seconds: int = DEFAULT_TTL_SECONDS):
    """Context manager: acquire a distributed team-skill write lock.

    Raises RuntimeError if the lock cannot be acquired (contention).
    Releases the lock in ``finally`` — guaranteed even if the write raises.

    Usage::

        with team_skill_lock(skill_key, worker_id):
            write_skill(name, content, scope="team", identity=identity)

    Raises
    ------
    RuntimeError
        When the lock is held by another worker.
    ImportError
        When boto3 is not installed.
    """
    if not acquire_skill_lock(skill_key, worker_id, ttl_seconds=ttl_seconds):
        skill_name = skill_key.rsplit("/", 1)[-1]
        raise RuntimeError(
            f"Skill '{skill_name}' is currently being edited by another agent worker. "
            "Please retry in a moment."
        )
    try:
        yield
    finally:
        release_skill_lock(skill_key, worker_id)
