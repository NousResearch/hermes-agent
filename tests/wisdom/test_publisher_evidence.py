from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from hermes_wisdom.client import VersionDetail
from hermes_wisdom.contract import author_description_hash
from tests.wisdom.test_share_queue import sharing  # noqa: F401
from tests.wisdom.test_share_staging import staged  # noqa: F401


def test_only_own_recent_profile_usage_enters_reviewable_description(
    staged, monkeypatch
):
    from hermes_wisdom.publisher_evidence import publication_description

    service, package, source, _ = staged
    store = service.store
    skill = store.register_skill(
        source, content_hash=package.source_content_hash, source_kind="local"
    )
    (source.parent / "other").mkdir()
    other = store.register_skill(
        source.parent / "other", content_hash="other", source_kind="local"
    )
    monkeypatch.setattr(
        "hermes_wisdom.publisher_evidence._profile_timezone",
        lambda: (ZoneInfo("Australia/Brisbane"), "Australia/Brisbane"),
    )
    at = datetime(2026, 9, 9, 15, tzinfo=timezone.utc)
    with store.transaction() as db:
        db.executemany(
            "INSERT INTO usage_day VALUES(?,?,?,?)",
            [
                (skill, "2026-09-04", "Australia/Brisbane", 2),
                (skill, "2026-09-10", "Australia/Brisbane", 1),
                (skill, "2026-09-03", "Australia/Brisbane", 50),
                (skill, "2026-09-11", "Australia/Brisbane", 60),
                (skill, "2026-09-05", "UTC", 70),
                (other, "2026-09-05", "Australia/Brisbane", 80),
            ],
        )
    result = publication_description(store, skill, package.plain_description, at=at)
    assert result.startswith(package.plain_description + "\n\n")
    assert "Publisher usage (client-reported)" in result
    assert "3 invocations on 2 days" in result
    assert "2026-09-04 through 2026-09-10" in result
    assert (
        "may span local revisions" in result
        and "do not verify successful outcomes" in result
    )
    assert str(source) not in result and "Australia/Brisbane" not in result
    # Missing evidence is not evidence of zero use or a synthetic recommendation.
    (source.parent / "unused").mkdir()
    empty = store.register_skill(
        source.parent / "unused", content_hash="unused", source_kind="local"
    )
    assert publication_description(store, empty, "Owner copy", at=at) == "Owner copy"
    with store.transaction() as db:
        db.execute("UPDATE local_skill SET source_kind='managed' WHERE id=?", (skill,))
    assert publication_description(store, skill, "Owner copy", at=at) == "Owner copy"


@pytest.mark.parametrize("remove_evidence", [False, True])
def test_publisher_evidence_is_frozen_reviewed_and_hash_bound_before_upload(
    sharing, monkeypatch, remove_evidence
):
    service, mediation, actor, shown, _, source, _ = sharing
    monkeypatch.setattr(
        "hermes_wisdom.publisher_evidence._profile_timezone",
        lambda: (timezone.utc, "UTC"),
    )
    day = datetime.now(timezone.utc).date().isoformat()
    retain_after = (datetime.now(timezone.utc).date() - timedelta(days=35)).isoformat()
    skill = shown["facts"]["skill_id"]
    service.store.record_usage_day(
        skill, day, timezone_name="UTC", retain_after=retain_after
    )
    mediation.consent.resolve("org", shown["id"], actor, "confirm")
    items = mediation.prepare(
        "org", actor, runtime={"model": "selected", "provider": "chosen"}, history=[]
    )
    final = items[0]["interaction"]
    inspection = mediation.consent.resolve("org", final["id"], actor, "inspect")[
        "inspection"
    ]
    description = inspection["description"]
    assert "Publisher usage (client-reported)" in description
    assert "1 invocation on 1 day" in description
    assert inspection["path"] == "Author description"
    assert final["facts"]["hashes"]["author_description"] == author_description_hash(
        description
    )
    assert service.client.uploaded == service.client.publications == 0
    draft = service.store.latest_draft_for_source(
        skill,
        service._candidate_event_context(
            mediation.queue.assessments("org")[0]["reference"]["event_id"]
        )[2],
    )
    package_job = next(
        job
        for job in mediation.queue.assessments("org")
        if job["reference"].get("prepared_draft_id")
    )
    assert package_job["reference"]["prepared_draft_id"] == draft["id"]
    # Later invocations must not silently change a package already offered for approval.
    service.store.record_usage_day(
        skill, day, timezone_name="UTC", retain_after=retain_after
    )
    from hermes_wisdom.agent_led.schemas import SharePackage
    from tests.wisdom.test_agent_led import _package_json

    package = SharePackage.model_validate_json(
        _package_json("notes", draft["source_hash"])
    )
    retried = service.prepare_share_package(
        package, source_path=source, staging_root=service.store.root / "share-staging"
    )
    assert retried["drafted_description"] == description
    assert retried["hashes"] == final["facts"]["hashes"]
    if remove_evidence:
        # The owner can remove the evidence, but the old confirmation cannot approve the edit.
        service.save_prepared(
            draft["id"],
            author_description="Owner chose not to share usage.",
            files=retried["files"],
        )
        rejected = mediation.consent.resolve("org", final["id"], actor, "confirm")
        assert rejected["state"] == "stale"
        assert service.client.uploaded == service.client.publications == 0
        edited = service.prepare_share_package(
            package,
            source_path=source,
            staging_root=service.store.root / "share-staging",
        )
        assert edited["drafted_description"] == "Owner chose not to share usage."
        final = mediation.consent.resolve("org", final["id"], actor, "recheck")
        description = edited["drafted_description"]
    result = mediation.consent.resolve("org", final["id"], actor, "confirm")
    assert result["state"] == "completed"
    assert service.client.uploaded == service.client.publications == 1
    assert service.client.submissions[0]["description"] == description
    published = service.client.drafts["draft-1"]
    assert (
        published.authorDescriptionHash
        == final["facts"]["hashes"]["author_description"]
    )

    # The recipient's exact-version inspection retains the reviewed copy and its label.
    def version(skill_id, number):
        assert (skill_id, number) == ("published-skill", 1)
        return VersionDetail(
            skill={"id": skill_id, "state": "active"},
            version={
                "version": number,
                "content_hash": published.contentHash,
                "author_description": published.authorDescription,
            },
        )

    monkeypatch.setattr(service.client, "version", version, raising=False)
    facts = mediation._skill_facts({
        "reference": {
            "kind": "skill",
            "skill_id": "published-skill",
            "version": 1,
            "notification": {"author_description": "Unreviewed feed copy"},
        }
    })
    assert facts["version"]["author_description"] == description
    assert "Unreviewed feed copy" not in str(facts)
