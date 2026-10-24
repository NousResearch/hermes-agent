"""Durable, per-profile reminders use the same real Telegram delivery path as staging."""

import json
import importlib
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import gateway.run as gateway_run  # Bootstrap before the per-test home IO guard.
from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from tests.fakes.platforms.telegram_standin import TelegramStandin
from tools import write_approval as wa
from utils import atomic_json_write


def configure(home, server, chat="111", nag_hours=2):
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(json.dumps({
        "tools": {"write_approval": {"nag_hours": nag_hours}},
        "platforms": {"telegram": {"enabled": True, "token": server.token,
            "home_channel": {"platform": "telegram", "chat_id": chat, "name": "review"},
            "extra": {"base_url": server.api_base, "base_file_url": server.file_base}}},
    }))


def stage_and_set_clock(**clocks):
    rec = wa.stage_write("memory", {"action": "add", "target": "user", "content": "whole memory"},
                         summary="new", origin="pytest")
    rec = wa.get_pending("memory", rec["id"])
    rec.update(clocks)
    atomic_json_write(wa._pending_path("memory", rec["id"]), rec)
    return rec


def test_due_unanswered_snoozed_and_silence(tmp_path, monkeypatch):
    from tools.memory_prompt_cadence import tick

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    now = time.time()
    with TelegramStandin() as server:
        configure(tmp_path, server)
        assert tick(now=now) == 0
        assert not server.calls_of("sendMessage")
        overdue = stage_and_set_clock(last_prompt_at=now - 7201)
        recent = stage_and_set_clock(last_prompt_at=now - 1)
        asleep = stage_and_set_clock(last_prompt_at=now - 9000, snooze_until=now + 20)
        awake = stage_and_set_clock(last_prompt_at=now - 1, snooze_until=now - 1)
        before = len(server.calls_of("sendMessage"))
        assert tick(now=now) == 2
        assert len(server.calls_of("sendMessage")) - before == 2
        assert tick(now=now) == 0
        assert tick(now=now + 21) == 1
        for rec in (overdue, recent, asleep, awake):
            wa.discard_pending("memory", rec["id"])
        before = len(server.calls_of("sendMessage"))
        assert tick(now=now + 90000) == 0
        assert len(server.calls_of("sendMessage")) == before


def test_config_loader_default_custom_and_disabled(tmp_path, monkeypatch):
    from hermes_cli.config import load_config
    from tools.memory_prompt_cadence import tick

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    assert load_config()["tools"]["write_approval"]["nag_hours"] == 2
    now = time.time()
    with TelegramStandin() as server:
        configure(tmp_path, server, nag_hours=0)
        rec = stage_and_set_clock(last_prompt_at=now - 3601, snooze_until=now - 1)
        before = len(server.calls_of("sendMessage"))
        assert tick(now=now) == 0
        assert len(server.calls_of("sendMessage")) == before
        rec.pop("snooze_until")
        atomic_json_write(wa._pending_path("memory", rec["id"]), rec)
        configure(tmp_path, server, nag_hours=4)
        assert tick(now=now) == 0
        configure(tmp_path, server, nag_hours=1)
        assert tick(now=now) == 1


def test_missing_binding_retries_when_configured(tmp_path, monkeypatch):
    from tools.memory_prompt_cadence import tick

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    rec = stage_and_set_clock()
    assert "last_prompt_at" not in rec
    assert tick() == 0
    with TelegramStandin() as server:
        configure(tmp_path, server)
        assert tick() == 1
        assert tick() == 0
        assert len(server.calls_of("sendMessage")) == 1


def test_scheduled_job_visits_each_owning_profile(tmp_path, monkeypatch):
    # Execute the registered chore through the actual housekeeping scheduler and scope wrapper.
    fake_home = tmp_path / "home"
    a = fake_home / ".hermes"
    b = a / "profiles" / "b"
    monkeypatch.setattr(type(tmp_path), "home", classmethod(lambda cls: fake_home))
    monkeypatch.setenv("HERMES_HOME", str(a))
    monkeypatch.setattr(gateway_run, "_housekeeping_chore",
        lambda label, fn: fn() if label == "Memory review reminders" else None)
    class OneTick:
        done = False
        def is_set(self):
            return self.done
        def wait(self, timeout=None):
            self.done = True
            return True

    with TelegramStandin() as server:
        for home, chat in ((a, "111"), (b, "222")):
            configure(home, server, chat)
            token = set_hermes_home_override(home)
            try:
                stage_and_set_clock(last_prompt_at=time.time() - 7201)
            finally:
                reset_hermes_home_override(token)
        before = len(server.calls_of("sendMessage"))
        runner = SimpleNamespace(config=SimpleNamespace(multiplex_profiles=True))
        gateway_run._start_gateway_housekeeping(OneTick(), interval=0, runner=runner)
        assert [str(c.params["chat_id"]) for c in server.calls_of("sendMessage")[before:]] == ["111", "222"]


def test_restart_retries_a_never_delivered_prompt_once(tmp_path, monkeypatch):
    import tools.memory_prompt_cadence as cadence
    import tools.memory_prompt_send as sender

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    rec = stage_and_set_clock()  # Staged without a binding: no successful send/receipt.
    assert "prompt" not in rec
    importlib.reload(sender)
    importlib.reload(cadence)
    with TelegramStandin() as server:
        configure(tmp_path, server)
        # Restart overlap: two schedulers see the same missed record.
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda _: cadence.tick(), range(2)))
        assert sorted(results) == [0, 1]
        importlib.reload(sender)
        importlib.reload(cadence)
        assert cadence.tick() == 0
        assert len(server.calls_of("sendMessage")) == 1
        assert len(wa.get_pending("memory", rec["id"])["prompts"]) == 1


def test_answered_records_never_return_after_restart(tmp_path, monkeypatch):
    import tools.memory_prompt_cadence as cadence
    from hermes_cli.write_approval_commands import handle_pending_subcommand
    from tools.memory_tool import load_on_disk_store

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    with TelegramStandin() as server:
        configure(tmp_path, server)
        approved, rejected = stage_and_set_clock(), stage_and_set_clock()
        handle_pending_subcommand("memory", ["approve", approved["id"]], memory_store=load_on_disk_store())
        handle_pending_subcommand("memory", ["reject", rejected["id"]])
        assert not wa.list_pending("memory")
        before = len(server.calls_of("sendMessage"))
        importlib.reload(cadence)
        assert cadence.tick(now=time.time() + 90000) == 0
        assert len(server.calls_of("sendMessage")) == before
        assert load_on_disk_store().user_entries == ["whole memory"]


def test_answer_during_delivery_does_not_resurrect_record(tmp_path, monkeypatch):
    from tools.memory_prompt_cadence import tick
    import tools.memory_prompt_send as sender

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    rec = stage_and_set_clock()
    deliver = sender._deliver
    async def deliver_then_answer(record, *args):
        receipt = await deliver(record, *args)
        # The CLI can discard a pending record while HTTP delivery is in flight.
        wa.discard_pending("memory", record["id"])
        return receipt
    monkeypatch.setattr(sender, "_deliver", deliver_then_answer)
    with TelegramStandin() as server:
        configure(tmp_path, server)
        assert tick() == 0
        assert wa.get_pending("memory", rec["id"]) is None
        assert tick() == 0
        assert len(server.calls_of("sendMessage")) == 1


def test_partial_send_recovers_known_parts_after_restart(tmp_path, monkeypatch):
    import tools.memory_prompt_send as sender
    from tools.memory_prompt_cadence import tick
    from telegram import Bot
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    with TelegramStandin() as server:
        configure(tmp_path, server)
        real_send = Bot.send_message
        attempts = 0
        async def interrupted(self, *args, **kwargs):
            nonlocal attempts
            attempts += 1
            if attempts == 2:
                raise RuntimeError('connection lost before sending part two')
            return await real_send(self, *args, **kwargs)
        monkeypatch.setattr(Bot, 'send_message', interrupted)
        rec = wa.stage_write('memory', {'action':'add', 'target':'user', 'content':'long 😀 paragraph.\n\n' * 600},
                             summary='long', origin='pytest')
        saved = wa.get_pending('memory', rec['id'])
        assert len(saved['delivery_in_progress']['message_ids']) == 1
        assert 'last_prompt_at' not in saved
        first_text = server.calls_of('sendMessage')[0].params['text']
        monkeypatch.setattr(Bot, 'send_message', real_send)
        importlib.reload(sender)
        assert tick() == 1
        texts = [c.params['text'] for c in server.calls_of('sendMessage')]
        assert texts.count(first_text) == 1
        assert 'delivery_in_progress' not in wa.get_pending('memory', rec['id'])
        assert tick() == 0
        assert sum(not c.params['disable_notification'] for c in server.calls_of('sendMessage')) == 1
