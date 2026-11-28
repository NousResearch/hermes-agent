"""A reaper and a compatible claimant must agree which one owns admission (#85342)."""

import json
import subprocess

from tools.environments import docker_reuse_owners as ownership


def test_reap_in_progress_rejects_claim_before_start(tmp_path, monkeypatch):
    store = ownership.ReuseOwners(tmp_path)
    claimant = ownership.ReuseOwners(tmp_path)
    labels = {"hermes-task-id": "task", "hermes-profile": "profile"}
    original = store.reserve("hermes-box", labels, "fp-old")
    store.finish("hermes-box", "cid-1")
    store.release(original)
    store.supersede(labels, "fp-new")
    record = {
        "Id": "cid-1", "Name": "/hermes-box",
        "Config": {"Labels": {
            ownership.OWNER_LABEL: store.namespace, "hermes-runtime-fingerprint": "fp-old"}},
        "State": {"Running": True},
    }
    claims, starts, calls = [], [], []

    def docker(argv, **kwargs):
        calls.append(argv[1])
        if argv[1] == "inspect":
            # A second real registry handle tries to revive this exact container
            # after reap admitted it, but before the external destructive operation.
            claims.append(claimant.claim("cid-1", lambda: starts.append("cid-1") or True))
            return subprocess.CompletedProcess(argv, 0, json.dumps(record), "")
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(ownership, "run_capture", docker)
    monkeypatch.setattr(ownership, "strip_sandbox_privileges", lambda *_: None)
    try:
        removed = store.reap("docker", 0, None)
        assert starts == [], "a reaper-admitted container must not be handed to a new owner"
        assert claims == [None]
        assert calls == ["inspect", "stop", "rm"]
        assert removed == 1
        with store.transaction() as db:
            assert db.execute("SELECT retired FROM containers").fetchone() == (1,)
            assert db.execute("SELECT token FROM owners").fetchall() == []
    finally:
        for lease in claims:
            if lease is not None:
                claimant.release(lease)
