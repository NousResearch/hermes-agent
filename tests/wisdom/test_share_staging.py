import json
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from pathlib import Path
from unittest.mock import Mock

import pytest

from hermes_wisdom.agent_led.schemas import SharePackage, SchemaRejected
from hermes_wisdom.agent_led.share_flow import (
    ShareFlow,
    package_hash,
    submit_via_service,
    write_package_to_staging,
)
from hermes_wisdom.client import WisdomConflict
from hermes_wisdom.service import WisdomService, _source_fingerprint
from hermes_wisdom.store import WisdomStore
from tests.wisdom.test_service import FakeClient
from tests.wisdom.test_agent_led import _package_json


@pytest.fixture
def staged(tmp_path, monkeypatch):
    source = tmp_path / "skills" / "notes"
    source.mkdir(parents=True)
    (source / "SKILL.md").write_text(
        "---\nname: notes\ndescription: Private original\n---\nORIGINAL_PRIVATE_SETUP\n",
        encoding="utf-8",
    )
    (source / "references").mkdir()
    (source / "references" / "private.md").write_text(
        "PRIVATE_REFERENCE", encoding="utf-8"
    )
    package = SharePackage.model_validate_json(
        _package_json("notes", _source_fingerprint(source))
    )
    fake = FakeClient()
    service = WisdomService(store=WisdomStore(tmp_path / "state"), client=fake)
    monkeypatch.setattr(service, "_eligible_paths", lambda: [source])
    monkeypatch.setattr(service, "_enqueue_professionalism_review", Mock())
    monkeypatch.setattr(
        service,
        "_require_professionalism_review",
        Mock(return_value={"status": "unavailable"}),
    )
    return service, package, source, tmp_path / "staging"


def test_generated_package_is_the_prepared_and_uploaded_package(staged):
    service, package, source, root = staged
    prepared = service.prepare_share_package(
        package, source_path=source, staging_root=root
    )
    text = json.dumps(prepared["files"])
    assert "Generalized body." in text
    assert "ORIGINAL_PRIVATE_SETUP" not in text and "PRIVATE_REFERENCE" not in text
    assert "Release Notes" in text
    assert service.client.uploaded == 0 and service.client.submissions == []
    security = prepared["security_check"]
    draft = service.store.draft(prepared["local_draft_id"])
    assert security["source"] == "local_preflight"
    assert security["content_hash"] == draft["content_hash"]
    assert security["author_description_hash"] == draft["description_hash"]
    assert security["upload_allowed"] is True
    rows = {row["key"]: row["status"] for row in security["checks"]}
    assert rows["private_keys"] == rows["live_credentials"] == "pass"
    assert rows["organization_policy"] == rows["personal_information"] == "pending"
    assert "ORIGINAL_PRIVATE_SETUP" in (source / "SKILL.md").read_text()
    result = service.suggest(
        "notes",
        description=prepared["drafted_description"],
        system_specification=prepared["system_specification"],
    )
    assert (
        result["draft"]["contentHash"] == service.store.draft("draft-1")["content_hash"]
    )
    uploaded_overlay = Path(service.store.draft("draft-1")["overlay_path"])
    assert "Generalized body." in (uploaded_overlay / "SKILL.md").read_text()
    assert not (uploaded_overlay / "references" / "private.md").exists()
    setup = (uploaded_overlay / "refs" / "wisdom-setup.md").read_text()
    setup_data = json.loads(setup.split("```json\n", 1)[1].split("\n```", 1)[0])
    assert setup_data["verification_step"] == "gh --version"
    assert setup_data["execution_requires_user_approval"] is True
    review = service._require_professionalism_review.call_args.kwargs
    assert review["content_hash"] == result["draft"]["contentHash"]
    assert review["package_root"] == uploaded_overlay
    assert service.client.submissions[0]["professionalism_review"] == {
        "status": "unavailable"
    }


def test_prepared_security_never_attaches_results_to_changed_bytes(staged):
    service, package, source, root = staged
    prepared = service.prepare_share_package(package, source_path=source, staging_root=root)
    (Path(prepared["overlay_path"]) / "SKILL.md").write_text("changed")
    with pytest.raises(WisdomConflict, match="prepared package changed"):
        service._prepared_result(service.store.draft(prepared["local_draft_id"]))
    assert service.client.uploaded == 0


def test_background_package_waits_for_parallel_reviews(staged, monkeypatch):
    service, package, source, root = staged
    scan_started, review_started = Event(), Event()

    def scan(_path):
        scan_started.set()
        assert review_started.wait(5), "review must start alongside the local scan"
        return {"guard": {"allowed": True, "findings": []}}

    def review(**kwargs):
        review_started.set()
        assert scan_started.wait(5), "scan must not wait for the model"
        assert kwargs["content_hash"] != package.source_content_hash
        return {"status": "pass", "summary": "No concerning language detected."}

    monkeypatch.setattr("hermes_wisdom.service._scan_summary", scan)
    monkeypatch.setattr(service, "_require_professionalism_review", review)
    result = service.prepare_share_package(
        package, source_path=source, staging_root=root, finish_reviews=True
    )
    assert result["professionalism_check"]["status"] == "pass"
    assert result["security_check"]["source"] == "local_preflight"
    assert service.client.uploaded == 0


@pytest.mark.parametrize("status", ["unavailable", "pending"])
def test_background_package_requires_terminal_professionalism(staged, monkeypatch, status):
    service, package, source, root = staged
    monkeypatch.setattr(service, "_require_professionalism_review", lambda **_: {"status": status})
    if status == "pending":
        with pytest.raises(WisdomConflict, match="still running"):
            service.prepare_share_package(
                package, source_path=source, staging_root=root, finish_reviews=True
            )
    else:
        result = service.prepare_share_package(
            package, source_path=source, staging_root=root, finish_reviews=True
        )
        assert result["professionalism_check"]["status"] == "unavailable"
    assert service.client.uploaded == 0


def test_background_review_rejects_bytes_changed_during_model_call(staged, monkeypatch):
    service, package, source, root = staged
    scanned = Event()

    def scan(_path):
        scanned.set()
        return {"guard": {"allowed": True}}

    def review(**kwargs):
        assert scanned.wait(5)
        (kwargs["package_root"] / "SKILL.md").write_text("Changed during review")
        return {"status": "pass"}

    monkeypatch.setattr("hermes_wisdom.service._scan_summary", scan)
    monkeypatch.setattr(service, "_require_professionalism_review", review)
    with pytest.raises(WisdomConflict, match="changed"):
        service.prepare_share_package(
            package, source_path=source, staging_root=root, finish_reviews=True
        )
    assert service.client.uploaded == 0


def test_staging_is_content_addressed_and_concurrent_retries_do_not_overwrite(staged):
    _, package, _, root = staged
    with ThreadPoolExecutor(max_workers=2) as pool:
        paths = list(
            pool.map(lambda _: write_package_to_staging(package, root), range(2))
        )
    assert paths[0] == paths[1]
    assert package_hash(package).removeprefix("sha256:") in str(paths[0])
    (paths[0] / "SKILL.md").write_text("tampered", encoding="utf-8")
    with pytest.raises(SchemaRejected, match="changed"):
        write_package_to_staging(package, root)
    assert (paths[0] / "SKILL.md").read_text() == "tampered"


@pytest.mark.parametrize("attack", ["extra", "symlink", "hardlink"])
def test_staging_rejects_foreign_files_without_following_them(staged, attack, tmp_path):
    _, package, _, root = staged
    target = write_package_to_staging(package, root)
    outside = tmp_path / "outside"
    outside.write_text("untouched", encoding="utf-8")
    if attack == "extra":
        (target / "extra.md").write_text("unexpected", encoding="utf-8")
    elif attack == "symlink":
        (target / "extra.md").symlink_to(outside)
    else:
        (target / "SKILL.md").unlink()
        (target / "SKILL.md").hardlink_to(outside)
    with pytest.raises(SchemaRejected):
        write_package_to_staging(package, root)
    assert outside.read_text() == "untouched"


def test_source_edit_requires_new_package_before_preparation(staged):
    service, package, source, root = staged
    (source / "SKILL.md").write_text("changed", encoding="utf-8")
    with pytest.raises(WisdomConflict, match="source changed"):
        service.prepare_share_package(package, source_path=source, staging_root=root)
    assert not root.exists() and service.client.uploaded == 0


def test_retry_reuses_prepared_draft_and_never_reopens_submitted_state(staged):
    service, package, source, root = staged
    first = service.prepare_share_package(
        package, source_path=source, staging_root=root
    )
    second = service.prepare_share_package(
        package, source_path=source, staging_root=root
    )
    assert first["local_draft_id"] == second["local_draft_id"]
    assert first["overlay_path"] == second["overlay_path"]
    service.store.set_draft_state(first["local_draft_id"], "submitted")
    with pytest.raises(WisdomConflict):
        service.prepare_share_package(package, source_path=source, staging_root=root)
    assert service.store.draft(first["local_draft_id"])["state"] == "submitted"


def test_flow_approval_retries_after_failure_and_uses_generated_bytes(staged, tmp_path):
    service, package, source, root = staged
    flow = ShareFlow(root=tmp_path / "flows")
    flow.start(source)
    flow.package(model_call=lambda *_: package.model_dump_json())
    adapter = submit_via_service(service, root)
    with pytest.raises(TimeoutError):
        flow.approve(submit=Mock(side_effect=TimeoutError))
    resumed = ShareFlow(flow.flow_id, root=flow.root)
    result = resumed.approve(submit=adapter)
    assert result["prepared_review"]["network_submission"] is False
    assert "Generalized body." in json.dumps(result["prepared_review"]["files"])
    no_repeat = Mock(side_effect=AssertionError("already prepared"))
    assert resumed.approve(submit=no_repeat) == result
    assert service.client.uploaded == 0


def test_changed_flow_package_invalidates_approval(staged, tmp_path):
    _, package, source, _ = staged
    flow = ShareFlow(root=tmp_path / "flows")
    flow.start(source)
    flow.package(model_call=lambda *_: package.model_dump_json())
    flow.state["package"]["files"][0]["content"] = "different bytes"
    with pytest.raises(SchemaRejected, match="changed"):
        flow.approve(submit=Mock(side_effect=AssertionError("must not prepare")))


@pytest.mark.parametrize(
    "identity", ["../outside", "../../auth", "/absolute", "name.json"]
)
def test_flow_identity_cannot_escape_its_directory(tmp_path, identity):
    with pytest.raises(ValueError):
        ShareFlow(identity, root=tmp_path)


def test_packaged_file_whitespace_is_not_silently_rewritten():
    from hermes_wisdom.agent_led.schemas import PackagedFile

    text = "\n  preserve indentation\n\n"
    item = PackagedFile(path="refs/example.md", content=text)
    assert PackagedFile.model_validate_json(item.model_dump_json()).content == text


def test_prepared_overlay_edit_is_detected_without_overwriting_it(staged):
    service, package, source, root = staged
    first = service.prepare_share_package(
        package, source_path=source, staging_root=root
    )
    skill = Path(first["overlay_path"]) / "SKILL.md"
    skill.write_text("Owner changed this overlay", encoding="utf-8")
    with pytest.raises(WisdomConflict, match="prepared package changed"):
        service.prepare_share_package(package, source_path=source, staging_root=root)
    assert skill.read_text() == "Owner changed this overlay"
    assert len(list((service.store.root / "share-prepared").iterdir())) == 1


def test_source_change_during_preparation_cleans_uncommitted_overlay(
    staged, monkeypatch
):
    import hermes_wisdom.service as service_module

    service, package, source, root = staged
    prepare = service_module.prepare_package

    def prepare_then_change(*args, **kwargs):
        result = prepare(*args, **kwargs)
        (source / "SKILL.md").write_text("Source changed during packaging")
        return result

    monkeypatch.setattr(service_module, "prepare_package", prepare_then_change)
    with pytest.raises(WisdomConflict, match="source changed during preparation"):
        service.prepare_share_package(package, source_path=source, staging_root=root)
    assert list((service.store.root / "share-prepared").iterdir()) == []
    assert service.client.uploaded == 0


def test_concurrent_preparation_keeps_one_canonical_overlay(staged):
    service, package, source, root = staged
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(
            pool.map(
                lambda _: service.prepare_share_package(
                    package, source_path=source, staging_root=root
                ),
                range(2),
            )
        )
    assert results[0]["local_draft_id"] == results[1]["local_draft_id"]
    assert results[0]["overlay_path"] == results[1]["overlay_path"]
    assert len(list((service.store.root / "share-prepared").iterdir())) == 1
    assert service.client.uploaded == 0


def test_failed_draft_write_cleans_overlay_and_retry_succeeds(staged, monkeypatch):
    service, package, source, root = staged
    record = service.store.record_draft

    def fail_after_write(*args, **kwargs):
        record(*args, **kwargs)
        raise OSError("simulated storage failure")

    monkeypatch.setattr(service.store, "record_draft", fail_after_write)
    with pytest.raises(OSError, match="simulated storage failure"):
        service.prepare_share_package(package, source_path=source, staging_root=root)
    assert list((service.store.root / "share-prepared").iterdir()) == []
    monkeypatch.setattr(service.store, "record_draft", record)
    result = service.prepare_share_package(
        package, source_path=source, staging_root=root
    )
    assert Path(result["overlay_path"]).is_dir()
    assert service.client.uploaded == 0
