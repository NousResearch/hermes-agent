"""Parity pins between typed legacy mapping and current #91802 gates."""

import json
import textwrap
from pathlib import Path

import pytest

from agent.bot_control_plane import (
    LegacyMessageAgentState,
    legacy_message_agent_dispatch_decision,
    legacy_message_agent_injection_decision,
)
from tools import bot_mode_dm, bot_mode_probe


@pytest.fixture(autouse=True)
def _fresh_probe_cache():
    bot_mode_probe._reset_cache_for_tests()
    yield
    bot_mode_probe._reset_cache_for_tests()


def _home(tmp_path: Path, *, managed: bool, legacy_soul: bool = False) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    if managed:
        profile = home / "profiles" / "researcher"
        profile.mkdir(parents=True)
        (profile / "profile.yaml").write_text(
            textwrap.dedent(
                """\
                description: teammate for parity tests
                ui_meta:
                  hermes-bots:
                    shape: cloud
                """
            ),
            encoding="utf-8",
        )
    if legacy_soul:
        (home / "SOUL.md").write_text(
            "# Existing bot identity\n\n## Messaging other agents\nlegacy protocol\n",
            encoding="utf-8",
        )
    return home


class _FakeDB:
    def __init__(self, home: Path, title: str):
        self.db_path = str(home / "state.db")
        self.title = title

    def get_session_title(self, _session_id):
        return self.title


class _FakeAgent:
    def __init__(
        self,
        home: Path,
        *,
        title: str,
        protocol_enabled: bool,
        schema_present: bool,
    ):
        self._session_db = _FakeDB(home, title)
        self._session_title_hint = None
        self._bot_mode_protocol = protocol_enabled
        self.session_id = "session-parity"
        self.tools = [bot_mode_dm.message_agent_tool_schema()] if schema_present else []
        self.valid_tool_names = set()


@pytest.mark.parametrize(
    ("protocol", "schema", "canonical", "managed"),
    (
        (False, True, True, True),
        (True, True, False, False),
        (True, False, True, True),
        (True, False, True, False),
        (True, False, False, True),
    ),
)
def test_injection_mapping_matches_current_gate(
    tmp_path, protocol, schema, canonical, managed
):
    title = "Bot Chat" if canonical else "Ordinary chat"
    agent = _FakeAgent(
        _home(tmp_path, managed=managed),
        title=title,
        protocol_enabled=protocol,
        schema_present=schema,
    )
    mapped = legacy_message_agent_injection_decision(
        LegacyMessageAgentState(
            protocol_enabled=protocol,
            schema_present=schema,
            canonical_bot_chat=canonical,
            managed_install=managed,
        )
    )
    assert bot_mode_dm.ensure_message_agent_tool(agent) is mapped.allowed


def test_legacy_soul_dedupe_still_injects_on_managed_install(tmp_path):
    home = _home(tmp_path, managed=True, legacy_soul=True)
    agent = _FakeAgent(
        home,
        title="Bot Chat",
        protocol_enabled=True,
        schema_present=False,
    )

    # #92784-era injection is gated on managed-install identity, not on
    # whether the prompt probe emits text. A legacy SOUL heading suppresses
    # duplicate prompt text but must not silently remove message_agent.
    assert bot_mode_probe.get_bot_mode_protocol_section(home) == ""
    assert bot_mode_dm.ensure_message_agent_tool(agent) is True


@pytest.mark.parametrize(
    ("protocol", "schema", "canonical", "managed"),
    (
        (False, False, True, True),
        (True, True, False, True),
        (True, False, True, False),
    ),
)
def test_dispatch_mapping_matches_current_gate(
    tmp_path, monkeypatch, protocol, schema, canonical, managed
):
    title = "Bot Chat" if canonical else "Ordinary chat"
    agent = _FakeAgent(
        _home(tmp_path, managed=managed),
        title=title,
        protocol_enabled=protocol,
        schema_present=schema,
    )

    import tools.terminal_tool as terminal_tool_module

    monkeypatch.setattr(
        terminal_tool_module,
        "terminal_tool",
        lambda *_args, **_kwargs: json.dumps({"session_id": "proc-parity"}),
    )
    result = json.loads(
        bot_mode_dm.message_agent_tool(
            target="researcher",
            message="parity probe",
            agent=agent,
        )
    )
    actual_allowed = result.get("status") == "sent"
    mapped = legacy_message_agent_dispatch_decision(
        LegacyMessageAgentState(
            protocol_enabled=protocol,
            schema_present=schema,
            canonical_bot_chat=canonical,
            managed_install=managed,
        )
    )
    assert actual_allowed is mapped.allowed
