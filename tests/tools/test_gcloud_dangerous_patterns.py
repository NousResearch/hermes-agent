"""Tests for the gcloud delete guards in DANGEROUS_PATTERNS.

These patterns protect against accidental cloud resource destruction
(Cloud Run services, Firestore/Datastore, generic gcloud resources).
They were added in response to real AgentX workflows where `gcloud run
services delete` and friends are easy to type wrong and irreversible.

Every gcloud `... delete ...` invocation should require explicit user
approval, even under yolo.  (Yolo bypasses DANGEROUS_PATTERNS, so the
true safety net is HARDLINE_PATTERNS — but the dangerous list is the
right home for "require user confirmation" rather than "block
unconditionally".)
"""

import pytest

from tools.approval import (
    detect_dangerous_command,
)


# Commands that MUST trigger a dangerous-command approval prompt.
# Add more service groups as they're discovered in real workflows.
_SHOULD_MATCH = [
    # Cloud Run — destroys service + all revisions, no rollback.
    "gcloud run services delete agentx",
    "gcloud run services delete agentx --region=us-east1",
    "gcloud run delete agentx",
    # Firestore / Datastore — indexes, databases, documents.
    "gcloud firestore databases delete --database=foo",
    "gcloud firestore indexes delete foo",
    "gcloud datastore indexes delete foo",
    # Generic gcloud resource groups — covered by the structural pattern.
    "gcloud compute instances delete my-vm",
    "gcloud sql instances delete my-db",
    "gcloud container clusters delete my-cluster",
    "gcloud storage buckets delete gs://my-bucket",
    "gcloud pubsub topics delete my-topic",
    "gcloud secrets delete my-secret",
    "gcloud projects delete my-project",
]


# Read-only / non-destructive commands that must NOT be flagged.
_SHOULD_NOT_MATCH = [
    "gcloud run services describe agentx",
    "gcloud run services list",
    "gcloud run deploy agentx --image=... --region=us-east1",
    "gcloud firestore databases list",
    "gcloud firestore databases create --database=foo",
    "gcloud compute instances list",
    "gcloud sql instances create my-db",
    "gcloud config get project",
    "gcloud --version",
    "gcloud auth login",
]


@pytest.mark.parametrize("cmd", _SHOULD_MATCH)
def test_gcloud_delete_is_dangerous(cmd):
    """All gcloud `... delete ...` invocations must require approval."""
    is_dangerous, _reason, _pattern = detect_dangerous_command(cmd)
    assert is_dangerous, f"expected DANGEROUS_PATTERNS to flag: {cmd!r}"


@pytest.mark.parametrize("cmd", _SHOULD_NOT_MATCH)
def test_gcloud_non_delete_is_not_dangerous(cmd):
    """Non-delete gcloud commands (list, describe, deploy, create) must pass."""
    is_dangerous, reason, _pattern = detect_dangerous_command(cmd)
    assert not is_dangerous, (
        f"unexpected DANGEROUS_PATTERNS match on safe command {cmd!r} "
        f"(reason={reason!r})"
    )
