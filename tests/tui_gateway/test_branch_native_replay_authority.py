"""Display-only branches must not replay native authority for discarded tool exchanges."""

from copy import deepcopy
import threading
from contextlib import contextmanager

import pytest

from agent.agent_runtime_helpers import copy_reasoning_content_for_api, reasoning_replay_field_for_api


class _BedrockAgent:
    provider = "bedrock"
    model = "anthropic.claude-sonnet-4"
    base_url = ""
    api_mode = "bedrock_converse"
    verbose_logging = False
    ephemeral_system_prompt = ""
    _current_turn_timestamp = 100.0
    _reasoning_replay_field = "off"
    _copy_reasoning_content_for_api = copy_reasoning_content_for_api
    _reasoning_replay_field_for_api = reasoning_replay_field_for_api

    @staticmethod
    def _extract_reasoning(message):
        return getattr(message, "reasoning_content", None)

    @staticmethod
    def _strip_think_blocks(text):
        return text

    @staticmethod
    def _needs_thinking_reasoning_pad():
        return False

    @staticmethod
    def _should_sanitize_tool_calls():
        return False

    @staticmethod
    def _split_responses_tool_id(value):
        return value, None

    @staticmethod
    def _deterministic_call_id(name, args, index):
        return f"call-{index}"

    @staticmethod
    def _derive_responses_function_call_id(value, response_item_id=None):
        return value


def _parent_history():
    from agent.bedrock_adapter import normalize_converse_response
    from agent.chat_completion_helpers import build_assistant_message

    response = normalize_converse_response({
        "output": {"message": {"role": "assistant", "content": [
            {"reasoningContent": {"redactedContent": b"opaque-thinking"}},
            {"text": "Visible answer before lookup."},
            {"toolUse": {"toolUseId": "call_1", "name": "web_search", "input": {"query": "hello"}}},
        ]}}, "stopReason": "tool_use",
    })
    assistant = build_assistant_message(_BedrockAgent(), response.choices[0].message, "tool_calls")
    assistant.update(timestamp=42.0, display_metadata={"label": "keep"})
    return [
        {"role": "user", "content": "Find it."}, assistant,
        {"role": "tool", "tool_call_id": "call_1", "content": "Found it."},
        {"role": "assistant", "content": "Final answer."},
    ]


def _api_history(history, agent):
    from agent.turn_context import build_api_messages

    history = deepcopy(history) + [{"role": "user", "content": "Continue."}]
    api, _ = build_api_messages(
        agent, history, current_turn_user_idx=len(history) - 1,
        ext_prefetch_cache=None, plugin_user_context=None, moa_config=None, active_system_prompt="",
    )
    return api


def _assert_safe_wire(history):
    from agent.bedrock_adapter import convert_messages_to_converse

    _, wire = convert_messages_to_converse(_api_history(history, _BedrockAgent()))
    blocks = [block for row in wire for block in row["content"]]
    calls = {block["toolUse"]["toolUseId"] for block in blocks if "toolUse" in block}
    results = {block["toolResult"]["toolUseId"] for block in blocks if "toolResult" in block}
    assert calls <= results, f"orphan native toolUses: {calls - results}"
    text = "\n".join(block["text"] for block in blocks if "text" in block)
    assert "Visible answer before lookup." in text
    assert "Final answer." in text
    assert not calls


def test_display_branch_native_tool_exchange_is_not_authoritative_after_sqlite_restart(tmp_path):
    from hermes_state import SessionDB
    from tui_gateway import server

    history = _parent_history()
    original = deepcopy(history)
    path = tmp_path / "state.db"
    with SessionDB(path) as db:
        db.create_session("parent", source="desktop", model=_BedrockAgent.model)
        db.append_messages_batch("parent", history)
        branch = server._branch_source_history(db, {
            "history": history, "history_lock": threading.Lock(),
        }, "parent")
        assert [row["role"] for row in branch] == ["user", "assistant", "assistant"]
        assert branch[1]["_reasoning_route"] == original[1]["_reasoning_route"]
        assert branch[1]["display_metadata"] == {"label": "keep"}
        assert branch[1]["timestamp"] == 42.0
        server._persist_branch(
            db, "child", "parent", "Branch", branch, source="desktop", cwd=str(tmp_path),
            profile_name="default", model=_BedrockAgent.model, copy_fields=server._BRANCH_COPY_FIELDS,
        )
    with SessionDB(path) as db:
        restored, _ = db.get_resume_conversations("child")
    # Test the durable path first: the original bug hides from generic tool-tail repair
    # because persistence drops tool_calls but retains the native ordered carrier.
    _assert_safe_wire(restored)
    _assert_safe_wire(branch)
    assert history[1]["bedrock_content_blocks"] == original[1]["bedrock_content_blocks"]
    assert history[1]["tool_calls"] == original[1]["tool_calls"]


@pytest.mark.parametrize("lazy", [False, True], ids=["eager-seed", "first-prompt-fallback"])
def test_seeded_branch_native_tool_exchange_is_not_authoritative_after_sqlite_restart(tmp_path, monkeypatch, lazy):
    from hermes_state import SessionDB
    from tui_gateway import server

    path = tmp_path / "state.db"
    with SessionDB(path) as db:
        db.create_session("parent", source="desktop", model=_BedrockAgent.model)
        db.append_messages_batch("parent", _parent_history())
        _, parent = db.get_resume_conversations("parent")
        # The desktop sends a display seed: no generic tool_calls or tool results.
        seed = server._coerce_seed_history(server._history_to_messages(parent))
        assert all(row["role"] != "tool" and "tool_calls" not in row for row in seed)
        if lazy:
            db.create_session("child", source="desktop", model=_BedrockAgent.model)

            @contextmanager
            def session_db(_session):
                yield db

            monkeypatch.setattr(server, "_session_db", session_db)
            record = {"session_key": "child", "seeded": True, "history": seed, "history_lock": threading.Lock()}
            server._persist_branch_seed(record)
            assert record["_branch_seed_persisted"]
        else:
            server._persist_branch(
                db, "child", "parent", "Branch", seed, source="desktop", cwd=str(tmp_path),
                profile_name="default", model=_BedrockAgent.model, copy_fields=server._BRANCH_COPY_FIELDS,
            )
    with SessionDB(path) as db:
        restored, _ = db.get_resume_conversations("child")
    _assert_safe_wire(restored)
    _assert_safe_wire(seed)


class _AnthropicAgent(_BedrockAgent):
    provider = "anthropic"
    model = "claude-sonnet-4"
    base_url = "https://api.anthropic.com"
    api_mode = "anthropic_messages"


def test_anthropic_seed_discards_ordered_tool_authority_after_sqlite_restart(tmp_path):
    from types import SimpleNamespace
    from agent.chat_completion_helpers import build_assistant_message
    from agent.transports.anthropic import AnthropicTransport
    from hermes_state import SessionDB
    from tui_gateway import server

    transport = AnthropicTransport()
    normalized = transport.normalize_response(SimpleNamespace(
        content=[
            SimpleNamespace(type="thinking", thinking="Lookup reasoning", signature="opaque-signature"),
            SimpleNamespace(type="text", text="Visible lookup answer."),
            SimpleNamespace(type="tool_use", id="call_1", name="web_search", input={"query": "hello"}),
        ], stop_reason="tool_use",
    ))
    assistant = build_assistant_message(_AnthropicAgent(), normalized, "tool_calls")
    assert any(block.get("type") == "tool_use" for block in assistant["anthropic_content_blocks"])
    path = tmp_path / "state.db"
    with SessionDB(path) as db:
        db.create_session("parent", source="desktop", model=_AnthropicAgent.model)
        db.append_messages_batch("parent", [
            {"role": "user", "content": "Lookup."}, assistant,
            {"role": "tool", "tool_call_id": "call_1", "content": "Result."},
        ])
        _, parent = db.get_resume_conversations("parent")
        seed = server._coerce_seed_history(server._history_to_messages(parent))
        server._persist_branch(
            db, "child", "parent", "Branch", seed, source="desktop", cwd=str(tmp_path),
            profile_name="default", model=_AnthropicAgent.model, copy_fields=server._BRANCH_COPY_FIELDS,
        )
    with SessionDB(path) as db:
        restored, _ = db.get_resume_conversations("child")
    for history in (restored, seed):
        assert not any(row.get("anthropic_content_blocks") for row in history)
        _, wire = transport.convert_messages(_api_history(history, _AnthropicAgent()))
        blocks = [block for row in wire for block in (
            row["content"] if isinstance(row["content"], list) else [{"type": "text", "text": row["content"]}]
        )]
        assert not [block for block in blocks if block.get("type") == "tool_use"]
        assert "Visible lookup answer." in "\n".join(block.get("text", "") for block in blocks)
        assert seed[1]["reasoning_details"] == assistant["reasoning_details"]
        assert seed[1]["_reasoning_route"] == assistant["_reasoning_route"]


class _CodexAgent(_BedrockAgent):
    provider = "openai-codex"
    model = "gpt-5-codex"
    base_url = "https://chatgpt.com/backend-api/codex"
    api_mode = "codex_responses"


@pytest.mark.parametrize("projection", ["branch", "seed"])
@pytest.mark.parametrize("changed", [False, True], ids=["exact-native-row", "display-projection"])
def test_codex_display_projection_cannot_override_branch_text_after_sqlite_restart(tmp_path, projection, changed):
    from types import SimpleNamespace
    from agent.chat_completion_helpers import build_assistant_message
    from agent.codex_responses_adapter import _normalize_codex_response, _chat_messages_to_responses_input
    from hermes_state import SessionDB
    from tui_gateway import server

    response, finish = _normalize_codex_response(SimpleNamespace(status="completed", output=[
        SimpleNamespace(type="reasoning", id="rs_1", encrypted_content="opaque", summary=[]),
        SimpleNamespace(type="message", role="assistant", id="msg_1", status="completed", phase="final_answer",
                        content=[SimpleNamespace(type="output_text", text="Original native answer.")]),
    ]), issuer_kind="codex_backend", issuer_model=_CodexAgent.model)
    assistant = build_assistant_message(_CodexAgent(), response, finish)
    path = tmp_path / "state.db"
    with SessionDB(path) as db:
        db.create_session("parent", source="desktop", model=_CodexAgent.model)
        db.append_messages_batch("parent", [{"role": "user", "content": "Question."}, assistant])
        _, parent = db.get_resume_conversations("parent")
        # Like a merged/projected desktop bubble: visible text is no longer the native row.
        if changed:
            parent[1]["content"] = "Visible display answer."
        if projection == "branch":
            history = server._visible_branch_history(parent)
        else:
            history = server._coerce_seed_history(server._history_to_messages(parent))
        server._persist_branch(
            db, "child", "parent", "Branch", history, source="desktop", cwd=str(tmp_path),
            profile_name="default", model=_CodexAgent.model, copy_fields=server._BRANCH_COPY_FIELDS,
        )
    with SessionDB(path) as db:
        restored, _ = db.get_resume_conversations("child")
    for replay in (restored, history):
        items = _chat_messages_to_responses_input(
            _api_history(replay, _CodexAgent()), current_issuer_kind="codex_backend",
            current_issuer_model=_CodexAgent.model,
        )
        text = "\n".join(part.get("text", "") for item in items if item.get("type") == "message"
                         for part in item["content"])
        if changed:
            assert "Visible display answer." in text
            assert "Original native answer." not in text
        else:
            assert "Original native answer." in text
            assert replay[1]["codex_message_items"] == assistant["codex_message_items"]
        assert replay[1]["codex_reasoning_items"] == assistant["codex_reasoning_items"]
        assert replay[1]["_reasoning_route"] == assistant["_reasoning_route"]


@pytest.mark.parametrize("field,blocks", [
    ("anthropic_content_blocks", [{"type": "thinking", "thinking": "think", "signature": "signed"},
                                  {"type": "text", "text": "Native answer."}]),
    ("bedrock_content_blocks", [{"reasoningContent": {"redactedContentBase64": "b3BhcXVl"}},
                                {"text": "Native answer."}]),
])
@pytest.mark.parametrize("changed", [False, True], ids=["safe-thinking-row", "projected-text"])
def test_ordered_native_branch_carrier_matches_visible_text(tmp_path, field, blocks, changed):
    from hermes_state import SessionDB
    from tui_gateway import server

    row = {"role": "assistant", "content": "Display answer." if changed else "Native answer.",
           field: blocks, "reasoning": "keep readable thinking", "_reasoning_route": "original-route"}
    marker = {"role": "user", "content": "Model changed.", "display_kind": "model_switch",
              "display_metadata": {"model": "original"}, "timestamp": 42.0}
    path = tmp_path / "state.db"
    with SessionDB(path) as db:
        db.create_session("parent", source="desktop", model="test-model")
        db.append_messages_batch("parent", [marker, row])
        _, parent = db.get_resume_conversations("parent")
        history = server._visible_branch_history(parent)
        server._persist_branch(
            db, "child", "parent", "Branch", history, source="desktop", cwd=str(tmp_path),
            profile_name="default", model="test-model", copy_fields=server._BRANCH_COPY_FIELDS,
        )
    with SessionDB(path) as db:
        restored, _ = db.get_resume_conversations("child")
    for replay in (restored, history, server._coerce_seed_history([row])):
        assistant = next(message for message in replay if message["role"] == "assistant")
        if changed:
            assert not assistant.get(field), "display text is not a complete native row"
        else:
            assert assistant[field] == blocks
        assert assistant["reasoning"] == row["reasoning"]
        assert assistant["_reasoning_route"] == row["_reasoning_route"]
    assert history[0]["display_kind"] == marker["display_kind"]
    assert restored[0]["display_metadata"] == marker["display_metadata"]
    assert restored[0]["timestamp"] == marker["timestamp"]
    assert row[field] == blocks


@pytest.mark.parametrize("projection", ["branch", "seed"])
@pytest.mark.parametrize("native_type", [[], {}], ids=["list-type", "dict-type"])
@pytest.mark.parametrize("field", ["anthropic_content_blocks", "bedrock_content_blocks", "codex_message_items"])
def test_malformed_native_type_does_not_crash_visible_projection(projection, native_type, field):
    from tui_gateway import server

    blocks = [{"type": native_type, "text": "Visible answer."}]
    carrier = [{"type": "message", "content": blocks}] if field == "codex_message_items" else blocks
    source = [{"role": "assistant", "content": "Visible answer.", field: carrier}]
    original = deepcopy(source)
    history = (server._visible_branch_history(source) if projection == "branch"
               else server._coerce_seed_history(source))
    assert history[0]["content"] == "Visible answer."
    assert history[0][field] == carrier
    assert source == original


def _empty_native_assistant(provider):
    from types import SimpleNamespace
    from agent.chat_completion_helpers import build_assistant_message

    if provider == "bedrock":
        from agent.bedrock_adapter import normalize_converse_response
        agent = _BedrockAgent()
        response = normalize_converse_response({
            "output": {"message": {"role": "assistant", "content": [
                {"reasoningContent": {"reasoningText": {"text": "Readable thinking", "signature": "signed"}}},
            ]}}, "stopReason": "end_turn",
        })
        normalized, finish = response.choices[0].message, "stop"
        field = "bedrock_content_blocks"
    elif provider == "anthropic":
        from agent.transports.anthropic import AnthropicTransport
        agent = _AnthropicAgent()
        normalized = AnthropicTransport().normalize_response(SimpleNamespace(
            content=[
                SimpleNamespace(type="thinking", thinking="Readable thinking", signature="signed"),
                SimpleNamespace(type="tool_use", id="call_empty", name="web_search", input={}),
            ], stop_reason="tool_use",
        ))
        # A display seed can carry only the thinking part of the ordered response.
        normalized.provider_data["anthropic_content_blocks"] = [
            block for block in normalized.provider_data["anthropic_content_blocks"] if block["type"] == "thinking"
        ]
        normalized.tool_calls = None
        finish, field = "stop", "anthropic_content_blocks"
    else:
        from agent.codex_responses_adapter import _normalize_codex_response
        agent = _CodexAgent()
        normalized, finish = _normalize_codex_response(SimpleNamespace(status="completed", output=[
            SimpleNamespace(type="reasoning", id="rs_empty", encrypted_content="opaque", summary=[]),
            SimpleNamespace(type="message", role="assistant", id="msg_empty", status="completed",
                            phase="final_answer", content=[SimpleNamespace(type="output_text", text="Native answer.")]),
        ]), issuer_kind="codex_backend", issuer_model=agent.model)
        field = "codex_message_items"
        normalized.codex_message_items[0]["content"][0]["text"] = ""
    agent.reasoning_callback = None
    assistant = build_assistant_message(agent, normalized, finish)
    assert assistant[field]
    # The selected display bubble is nonempty, unlike its native carrier.
    assistant.update(content="Visible answer.", reasoning="Readable reasoning control")
    return agent, field, assistant


def _provider_wire_text(history, agent):
    api = _api_history(history, agent)
    if agent.provider == "bedrock":
        from agent.bedrock_adapter import convert_messages_to_converse
        _, wire = convert_messages_to_converse(api)
    elif agent.provider == "anthropic":
        from agent.transports.anthropic import AnthropicTransport
        _, wire = AnthropicTransport().convert_messages(api)
    else:
        from agent.codex_responses_adapter import _chat_messages_to_responses_input
        wire = _chat_messages_to_responses_input(
            api, current_issuer_kind="codex_backend", current_issuer_model=agent.model,
        )
        wire = [item for item in wire if item.get("type") == "message"]
    return "\n".join(part.get("text", "") for row in wire for part in (
        row["content"] if isinstance(row["content"], list) else [{"text": row["content"]}]
    ))


@pytest.mark.parametrize("provider", ["bedrock", "anthropic", "codex"])
@pytest.mark.parametrize("projection", ["branch", "seed"])
@pytest.mark.parametrize("carrier_state", ["native-empty", "absent", "none", "empty-list"])
def test_empty_native_authority_cannot_erase_visible_answer_after_sqlite_restart(
    tmp_path, provider, projection, carrier_state,
):
    from hermes_state import SessionDB
    from tui_gateway import server

    agent, field, assistant = _empty_native_assistant(provider)
    if carrier_state == "absent":
        assistant.pop(field)
    elif carrier_state == "none":
        assistant[field] = None
    elif carrier_state == "empty-list":
        assistant[field] = []
    original = deepcopy(assistant)
    controls = {key: value for key, value in assistant.items() if key in (
        "reasoning", "reasoning_content", "reasoning_details", "_reasoning_route", "codex_reasoning_items",
    )}
    assert controls["_reasoning_route"]
    if provider == "codex":
        assert controls["codex_reasoning_items"][0]["encrypted_content"] == "opaque"
    path = tmp_path / "state.db"
    with SessionDB(path) as db:
        db.create_session("parent", source="desktop", model=agent.model)
        db.append_messages_batch("parent", [{"role": "user", "content": "Question."}, deepcopy(assistant)])
        _, parent = db.get_resume_conversations("parent")
        history = (server._visible_branch_history(parent) if projection == "branch"
                   else server._coerce_seed_history(server._history_to_messages(parent)))
        server._persist_branch(
            db, "child", "parent", "Branch", history, source="desktop", cwd=str(tmp_path),
            profile_name="default", model=agent.model, copy_fields=server._BRANCH_COPY_FIELDS,
        )
    with SessionDB(path) as db:
        restored, _ = db.get_resume_conversations("child")
    for replay in (restored, history):
        # Conversion, not just carrier shape, must prove the visible answer survived.
        assert "Visible answer." in _provider_wire_text(replay, agent)
        assert not replay[1].get(field)
        assert {key: replay[1][key] for key in controls} == controls
    assert assistant == original


@pytest.mark.parametrize("provider", ["bedrock", "anthropic"])
def test_reasoning_only_preservation_fixture_requires_matching_native_text(provider):
    from tui_gateway import server

    agent, field, row = _empty_native_assistant(provider)
    native_thinking = deepcopy(row[field][0])
    # Original server roundtrip fixtures called these safe despite visible content='done'.
    thinking = ({"reasoningContent": "signed"} if provider == "bedrock"
                else {"type": "thinking", "signature": "signed"})
    row.update(content="done")
    row[field] = [thinking]
    if provider == "bedrock":
        # The malformed string reasoning entry is ignored, falling back to content.
        assert "done" in _provider_wire_text([row], agent)
    else:
        assert "done" not in _provider_wire_text([row], agent)
    row[field] = [native_thinking]
    assert "done" not in _provider_wire_text([row], agent)
    text = {"text": "done"} if provider == "bedrock" else {"type": "text", "text": "done"}
    row[field] = [thinking, text]
    original = deepcopy(row)
    for replay in (server._visible_branch_history([row]), server._coerce_seed_history([row])):
        assert replay[0][field] == [thinking, text]
        assert "done" in _provider_wire_text(replay, agent)
    assert row == original


def test_seed_with_non_list_native_message_content_keeps_visible_text():
    from tui_gateway import server

    seed = [{"role": "assistant", "content": "Visible answer.",
             "codex_message_items": [{"type": "message", "content": None}]}]
    assert server._coerce_seed_history(seed)[0]["content"] == "Visible answer."
