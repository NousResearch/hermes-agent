"""Full review prompts cross the real Telegram HTTP stack in their owning home."""

import json

from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from tests.fakes.platforms.telegram_standin import TelegramStandin
from tools import write_approval as wa


def test_immediate_prompt_routes_a_b_a_and_buttons_are_on_last_chunk(tmp_path, monkeypatch):
    from tools.memory_prompt_text import render_cards

    # No fake sender: stage_write reaches PTB and the local Bot API stand-in.
    with TelegramStandin() as server:
        homes = [tmp_path / name for name in ("alpha", "beta")]
        for index, home in enumerate(homes):
            home.mkdir()
            (home / "config.yaml").write_text(json.dumps({"platforms": {"telegram": {
                "enabled": True, "token": server.token,
                "home_channel": {"platform": "telegram", "chat_id": str(111 + index), "name": "review"},
                "extra": {"base_url": server.api_base, "base_file_url": server.file_base},
            }}}))
        for home, chat in ((homes[0], "111"), (homes[1], "112"), (homes[0], "111")):
            token = set_hermes_home_override(home)
            try:
                before = len(server.calls_of("sendMessage"))
                rec = wa.stage_write("memory", {"action": "replace", "target": "user",
                    "matched_entry": "entire old 😀\n" * 1100, "old_text": "entire old",
                    "content": "entire new <literal>\n" * 1100}, summary="change", origin="pytest")
                calls = server.calls_of("sendMessage")[before:]
                assert len(calls) > 3
                assert all(str(c.params["chat_id"]) == chat for c in calls)
                saved = wa.get_pending("memory", rec["id"])
                assert [c.params["text"] for c in calls] == [c.text for c in render_cards(saved, limit=3700)]
                assert all(c.params["text"].startswith(f"default wants to modify {home / 'memories/USER.md'}\n") for c in calls)
                assert all(c.params["disable_notification"] for c in calls[:-1])
                assert not calls[-1].params["disable_notification"]
                assert all(c.params.get("entities") for c in calls)
                assert all(not c.params.get("reply_markup") for c in calls[:-1])
                row = calls[-1].params["reply_markup"]["inline_keyboard"]
                assert [[b["callback_data"] for b in buttons] for buttons in row] == [[
                    f"mem:{action}:{rec['id']}:1" for action in ("approve", "reject")], [
                    f"mem:{action}:{rec['id']}:1" for action in ("snooze", "edit")], [f"mem:discuss:{rec['id']}:1"]]
                saved = wa.get_pending("memory", rec["id"])
                assert saved["last_prompt_at"] >= rec["created_at"]
                assert saved["prompt"]["chat_id"] == chat
            finally:
                reset_hermes_home_override(token)


def test_no_binding_keeps_record_without_marking_delivery(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    rec = wa.stage_write("memory", {"action": "add", "target": "user", "content": "new"},
                         summary="new", origin="pytest")
    assert wa.get_pending("memory", rec["id"]) is not None
    assert "last_prompt_at" not in wa.get_pending("memory", rec["id"])
    assert "no Telegram binding" in caplog.text


def test_legacy_span_patch_opens_a_complete_entry_review(tmp_path, monkeypatch):
    from gateway.config import HomeChannel, Platform, PlatformConfig
    from tools.memory_prompt_send import send_prompt
    from tools.memory_tool import load_on_disk_store
    from utils import atomic_json_write

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    original = 'Drink tea daily. Keep the controller last.'
    assert load_on_disk_store().add('user', original)['success']
    record = {'id': 'abcdef12', 'subsystem': 'memory', 'action': 'patch', 'payload': {
        'action': 'patch', 'target': 'user', 'matched_entry': original,
        'old_text': 'tea', 'content': 'coffee'}}
    atomic_json_write(wa._pending_path('memory', record['id']), record)
    with TelegramStandin() as server:
        transport = PlatformConfig(enabled=True, token=server.token,
            extra={'base_url': server.api_base, 'base_file_url': server.file_base})
        destination = HomeChannel(platform=Platform.TELEGRAM, chat_id='111', name='review')
        assert send_prompt('memory', record, delivery=(transport, destination))
        text = '\n'.join(call.params['text'] for call in server.calls_of('sendMessage'))
        assert original in text
        assert 'Drink coffee daily. Keep the controller last.' in text
    saved = wa.get_pending('memory', record['id'])
    assert saved['payload']['action'] == 'replace'
    assert saved['payload']['content'] == 'Drink coffee daily. Keep the controller last.'
    assert load_on_disk_store().user_entries == [original]
