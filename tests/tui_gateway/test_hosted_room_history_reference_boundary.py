"""Independent read-only candidate review probes; synthetic temporary native state only."""
import hashlib
import json
import re
from copy import deepcopy
from pathlib import Path
import threading

import pytest

from gateway import hosted_room_attachments as files, hosted_room_driver as driver, hosted_rooms
from tests.tui_gateway.test_hosted_room_native_phase1 import native as native, WAIT
from tests.tui_gateway.test_hosted_room_produced_media import room as room, ROOM_ID
from tests.tui_gateway.test_hosted_room_history_context import archive, prepare, cold
from tui_gateway import hosted_room_history_context as history


@pytest.fixture(autouse=True)
def offline_metadata(monkeypatch):
    # Metadata is not under review. Keep the actual native preprocessing path,
    # but never allow its context-window discovery to reach a provider.
    from agent import model_metadata
    monkeypatch.setattr(model_metadata, "get_model_context_length", lambda *a, **kw: 1_000_000)


def prepared_attachment(room):
    data = b"MANDATORY_USER_BYTES"
    uploaded = room.attachments.put(room_id=ROOM_ID, upload_id="ordinary-input", kind="file",
        name="ordinary.txt", mime="text/plain", data=data)
    manifest = {k: uploaded[k] for k in ("attachment_id", "kind", "name", "size", "mime")}
    room.send(room_id=ROOM_ID, event_id="current-input", payload={
        "text": "@default compare both archived sources", "thread_id": "new-thread", "attachments": [manifest]})
    task, = driver.list_tasks(room.db_path, room_id=ROOM_ID, status="queued")
    binding, = room.bindings()
    return binding, task, manifest, data


def capture_staging(room):
    observed = []
    original = room.rpc.stage_attachment
    def capture(**kwargs):
        staged = original(**kwargs)
        observed.append((kwargs, dict(staged)))
        return staged
    room.rpc.stage_attachment = capture
    return observed


def test_actual_runner_reads_both_sources_from_its_own_prompt_reference(native, room):
    archived = archive(room)
    binding, task, mandatory, mandatory_bytes = prepared_attachment(room)
    immutable = deepcopy(task["payload"])
    expected = history.load_history_context(room, binding, task)
    assert expected is not None
    native.record(room.session_id)["cwd"] = str(native.home.parent)
    native.record(room.session_id)["explicit_cwd"] = True
    sid = native.resume_handle(native.resolves_by_title("Group: " + ROOM_ID))
    agent = native.make_agent("causal-reader")
    agent.session_id = native.record(sid)["session_key"]
    observed = capture_staging(room)
    ran = []
    def read_from_prompt(*args, **kwargs):
        # No source path/data passed to this fake runner, only the native runner input.
        prompt = args[0] if args else kwargs["user_message"]
        refs = re.findall(r"^History archive path \(JSON-encoded; read with file tools\): (.+)$", prompt, re.MULTILINE)
        assert len(refs) == 1
        path = Path(json.loads(refs[0]))
        assert path.is_relative_to(native.home.parent)
        assert mandatory_bytes.decode() in prompt  # Ordinary user file expansion stays native.
        archived_payload = json.loads(path.read_bytes())
        surfaces = {s["surface"]: s["records"][0]["text"] for s in archived_payload["legacy_history"]["sources"]}
        assert "DESKTOP_ONLY_FACT" not in prompt and "TELEGRAM_ONLY_FACT" not in prompt
        ran.append((threading.current_thread(), prompt, surfaces))
        return {"final_response": surfaces["desktop"] + " | " + surfaces["telegram"]}
    agent.run_conversation = read_from_prompt
    native.ready_agent(sid).release_turn.set()
    room.runtime._process_room(binding)
    assert len(ran) == 1, room.runtime.status()
    ran[0][0].join(WAIT)
    assert not ran[0][0].is_alive()
    durable = driver.get_task(room.db_path, task["identity"])
    assert durable["status"] == "settled", durable
    assert durable["payload"] == immutable
    assert "DESKTOP_ONLY_FACT" in durable["result"]["text"]
    assert "TELEGRAM_ONLY_FACT" in durable["result"]["text"]
    assert [a[0]["attachment"]["attachment_id"] for a in observed] == [mandatory["attachment_id"], expected[1]["attachment_id"]]
    assert not room.rpc._attachment_attempts and not room.rpc._staged_attachments
    assert json.loads(expected[2]) == archived["payload"]
    print("CAUSAL: runner read actual prompt refs, mandatory bytes and both source facts; terminal text depends on archive bytes")


@pytest.mark.parametrize("boundary", ["prompt", "index", "count", "bytes", "stage"])
def test_failed_extra_context_rolls_back_native_staging(native, room, monkeypatch, boundary):
    archive(room)
    binding, task, mandatory, mandatory_bytes = prepared_attachment(room)
    immutable = deepcopy(task["payload"])
    context = history.load_history_context(room, binding, task)
    assert context is not None
    index, history_manifest, history_bytes = context
    sid = native.resume_handle(native.resolves_by_title("Group: " + ROOM_ID))
    native.record(sid)["attached_images"] = ["preexisting-image-sentinel"]
    observed = capture_staging(room)
    original = room.rpc.stage_attachment
    if boundary == "prompt":
        monkeypatch.setattr(driver, "MAX_PROMPT_BYTES", len(task["payload"]["prompt"].encode()) + len(index.encode()))
    elif boundary == "index":
        room.runtime.history_context_loader = lambda *_: ("x" * (history.MAX_HISTORY_INDEX_BYTES + 1), history_manifest, history_bytes)
    elif boundary == "count":
        monkeypatch.setattr(files, "MAX_TASK_ATTACHMENTS", len(task["payload"]["attachments"]))
    elif boundary == "bytes":
        monkeypatch.setattr(files, "MAX_TASK_ATTACHMENT_BYTES", mandatory["size"] + history_manifest["size"] - 1)
    else:
        def fail_history(**kwargs):
            if kwargs["attachment"]["attachment_id"] == history_manifest["attachment_id"]:
                raise RuntimeError("injected second attachment failure")
            return original(**kwargs)
        room.rpc.stage_attachment = fail_history
    room.runtime._process_room(binding)
    durable = driver.get_task(room.db_path, task["identity"])
    assert durable["status"] == "failed", (boundary, durable)
    assert durable["admitted_at"] is None
    assert durable["payload"] == immutable
    assert not any(a.invocations for a in native.agents)
    assert native.record(sid)["attached_images"] == ["preexisting-image-sentinel"]
    assert not room.rpc._attachment_attempts and not room.rpc._staged_attachments
    expected_staged = 0 if boundary in ("count", "bytes") else 1 if boundary == "stage" else 2
    assert len(observed) == expected_staged, observed
    assert all(not Path(result["path"]).exists() for _, result in observed)
    assert room.attachments.read(room_id=ROOM_ID, event_id="current-input", attachment_id=mandatory["attachment_id"], recipient_member_id="default").data == mandatory_bytes
    assert room.attachments.read(room_id=ROOM_ID, event_id=history.ADOPTION_EVENT_ID, attachment_id=history_manifest["attachment_id"], recipient_member_id="default").data == history_bytes
    print("ROLLBACK:", boundary, "staged", len(observed), "admission absent; native files removed; immutable canonical files retained", durable["result"])


def test_archive_identity_retention_and_recipient_boundaries(native, room):
    archive(room)
    binding, task = prepare(room)
    context = history.load_history_context(room, binding, task)
    assert context is not None
    index, manifest, data = context
    digest = hashlib.sha256(data).hexdigest()
    assert digest == json.loads(index.split("\n", 1)[1])["sha256"]
    with hosted_rooms._transaction(room.db_path) as conn:
        rows = conn.execute("SELECT * FROM hosted_room_attachments WHERE event_id=?", (history.ADOPTION_EVENT_ID,)).fetchall()
        assert len(rows) == 1
        row = dict(rows[0])
        assert row["upload_id"] == "legacy-history:" + digest
        assert row["viewer_access"] == 0 and row["expires_at"] is None
        assert json.loads(row["recipient_member_ids_json"]) == ["default", "peer"]
    for overrides in ({"viewer": True, "recipient_member_id": None}, {"recipient_member_id": "foreign"}, {"event_id": "current-input"}, {"room_id": "foreign-room"}):
        args = dict(room_id=ROOM_ID, event_id=history.ADOPTION_EVENT_ID, attachment_id=manifest["attachment_id"], recipient_member_id="default")
        with pytest.raises(files.AttachmentNotFoundError):
            room.attachments.read(**(args | overrides))
    room.attachments.prune(now=room.attachments.clock() + files.UNCOMMITTED_TTL_SECONDS * 10)
    assert history.load_history_context(cold(room), binding, task) == (index, manifest, data)
    assert room.attachments.read(room_id=ROOM_ID, event_id=history.ADOPTION_EVENT_ID, attachment_id=manifest["attachment_id"], recipient_member_id="peer").data == data
    print("OWNERSHIP: stable SHA/upload/attachment identity, both frozen recipients, no viewer/cross-room/cross-event/foreign access; cold read after future prune")


def test_forged_task_context_does_not_acquire_archive(native, room):
    archive(room)
    binding, task = prepare(room)
    forged = deepcopy(task)
    forged["payload"]["input_context"]["event_seqs"] = [99999]
    with pytest.raises(RuntimeError, match="admitted task input event is missing"):
        history.load_history_context(room, binding, forged)
    with hosted_rooms._transaction(room.db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM hosted_room_attachments").fetchone()[0] == 0
    assert driver.get_task(room.db_path, task["identity"])["payload"] == task["payload"]


def captured_native_turn(native, room, binding):
    sid = native.resume_handle(native.resolves_by_title("Group: " + ROOM_ID))
    agent = native.make_agent("capture-boundary")
    agent.session_id = native.record(sid)["session_key"]
    captured = []
    def capture(*args, **kwargs):
        captured.append((threading.current_thread(), args[0] if args else kwargs["user_message"]))
        return {"final_response": "captured without model"}
    agent.run_conversation = capture
    native.ready_agent(sid).release_turn.set()
    room.runtime._process_room(binding)
    assert len(captured) == 1, room.runtime.status()
    captured[0][0].join(WAIT)
    assert not captured[0][0].is_alive()
    return captured[0][1]


def test_in_workspace_archive_stays_bounded_through_native_preprocessing(native, room):
    def larger(payload):
        payload["legacy_history"]["sources"][0]["records"][0]["text"] += "x" * 140_000
    archive(room, change=larger)
    binding, task = prepare(room)
    context = history.load_history_context(room, binding, task)
    assert context is not None
    index, manifest, data = context
    native.record(room.session_id)["cwd"] = str(native.home.parent)
    native.record(room.session_id)["explicit_cwd"] = True
    staged = capture_staging(room)
    prompt = captured_native_turn(native, room, binding)
    facts_in_prompt = all(f in prompt for f in ("DESKTOP_ONLY_FACT", "TELEGRAM_ONLY_FACT"))
    actual_bytes = len(prompt.encode())
    durable = driver.get_task(room.db_path, task["identity"])
    assert durable["status"] == "settled"
    assert len(staged) == 1 and Path(staged[0][1]["path"]).read_bytes() == data
    print("POST-PREPROCESS:", dict(archive_bytes=len(data), base_prompt_bytes=len(task["payload"]["prompt"].encode()), index_bytes=len(index.encode()), actual_runner_prompt_bytes=actual_bytes, limit=driver.MAX_PROMPT_BYTES, facts_in_prompt=facts_in_prompt, admitted=durable["admitted_at"] is not None))
    assert actual_bytes <= driver.MAX_PROMPT_BYTES, "native @file expansion occurs after checked runtime budget and admission"
    assert not facts_in_prompt, "history must stay referenced, not automatically inlined"


def test_provenance_strings_remain_inert_through_native_preprocessing(native, room, tmp_path):
    cwd = tmp_path / "workspace"
    cwd.mkdir()
    (cwd / "unrequested.txt").write_text("LOCAL_FILE_NOT_REQUESTED_BY_USER", encoding="utf-8")
    def provenance(payload):
        payload["legacy_history"]["sources"][0]["room_name"] = "Old @file:unrequested.txt "
    archive(room, change=provenance)
    binding, task = prepare(room)
    native.record(room.session_id)["cwd"] = str(cwd)
    native.record(room.session_id)["explicit_cwd"] = True
    before = deepcopy(task["payload"])
    prompt = captured_native_turn(native, room, binding)
    included = "LOCAL_FILE_NOT_REQUESTED_BY_USER" in prompt
    assert "LOCAL_FILE_NOT_REQUESTED_BY_USER" not in before["prompt"]
    assert driver.get_task(room.db_path, task["identity"])["payload"] == before
    print("PROVENANCE-PREPROCESS:", dict(unrequested_workspace_file_in_runner_prompt=included, source="historical room_name", user_input="@default compare both archived sources"))
    assert not included, "bounded historical provenance activated native @file expansion"
