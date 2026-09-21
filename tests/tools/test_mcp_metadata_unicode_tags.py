"""MCP server-controlled metadata must not carry invisible unicode TAG chars to the model.

Text content blocks pass through ``strip_unicode_tags``; before this fix the metadata
fields around them (uris, names, mime types, descriptions, roles, structuredContent,
_meta, isError text, dropped-block notices) did not, so a hostile server could hide
prompt-injection text in fields the model reads but users cannot see.
"""

import base64
import json
from types import SimpleNamespace

TAG = "\U000e0069\U000e0067"  # invisible TAG 'i','g' — renders as nothing, tokenizer-visible
TAGGED = f"ev{TAG}il"
# Valid emoji tag sequence (TR51): must survive stripping.
EMOJI_TAGGED = "\U0001f3f4\U000e0067\U000e0062\U000e0065\U000e006e\U000e0067\U000e007f"


def _has_tag(text: str) -> bool:
    return any("\U000e0000" <= c <= "\U000e007f" for c in str(text))


def _walk_strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for k, v in value.items():
            yield from _walk_strings(k)
            yield from _walk_strings(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from _walk_strings(v)


def _assert_no_tags(value):
    for s in _walk_strings(value):
        assert not _has_tag(s), f"tag chars survived in {s!r}"


class TestDroppedBlockNotice:
    def test_type_mime_uri_name_stripped(self):
        from tools.mcp_tool_content import _render_mcp_dropped_block_notice

        block = SimpleNamespace(
            type=TAGGED, mimeType=TAGGED, uri=TAGGED, name=TAGGED, size=42)
        out = _render_mcp_dropped_block_notice(block, TAGGED)
        assert "unsupported block" in out
        _assert_no_tags(out)
        assert "evil" in out
        assert "size=42" in out

    def test_nested_resource_uri_stripped(self):
        from tools.mcp_tool_content import _render_mcp_dropped_block_notice

        block = SimpleNamespace(type=None, resource=SimpleNamespace(uri=TAGGED))
        _assert_no_tags(_render_mcp_dropped_block_notice(block, "weird"))


class TestResourceLinkAndEmbeddedMetadata:
    def test_resource_link_fields_stripped(self):
        from tools.mcp_tool_content import _render_mcp_resource_block

        block = SimpleNamespace(
            type="resource_link", uri=TAGGED, name=TAGGED, mimeType=TAGGED)
        out = _render_mcp_resource_block(block, "srv")
        assert "resource link" in out
        _assert_no_tags(out)

    def test_embedded_blob_uri_mime_in_markers_stripped(self):
        from tools.mcp_tool_content import _render_mcp_resource_block

        res = SimpleNamespace(
            uri=TAGGED, mimeType=TAGGED,
            blob=base64.b64encode(b"x").decode("ascii"), text=None)
        out = _render_mcp_resource_block(SimpleNamespace(type="resource", resource=res), "srv")
        # cache unavailable or saved — either marker must be tag-free
        assert out
        _assert_no_tags(out)

    def test_embedded_blob_decode_fail_marker_stripped(self):
        from tools.mcp_tool_content import _render_mcp_resource_block

        res = SimpleNamespace(uri=TAGGED, mimeType="", blob="ööü", text=None)
        out = _render_mcp_resource_block(SimpleNamespace(type="resource", resource=res), "srv")
        assert "could not be decoded" in out
        _assert_no_tags(out)

    def test_emoji_tag_sequence_preserved_in_text(self):
        from tools.mcp_tool_content import _render_mcp_resource_block

        res = SimpleNamespace(uri="x://y", mimeType="text/plain", text=EMOJI_TAGGED, blob=None)
        out = _render_mcp_resource_block(SimpleNamespace(type="resource", resource=res), "srv")
        assert out == EMOJI_TAGGED


class TestUtilityRenderers:
    def test_resource_list_fields_stripped(self):
        from tools.mcp_tool_handlers import _render_resource_list

        r = SimpleNamespace(uri=TAGGED, name=TAGGED, description=TAGGED, mimeType=TAGGED)
        out = _render_resource_list([r], "srv")
        _assert_no_tags(out)
        assert out["resources"][0]["name"] == "evil"

    def test_prompt_list_fields_stripped(self):
        from tools.mcp_tool_handlers import _render_prompt_list

        p = SimpleNamespace(
            name=TAGGED, description=TAGGED,
            arguments=[SimpleNamespace(name=TAGGED, description=TAGGED, required=True)])
        out = _render_prompt_list([p], "srv")
        _assert_no_tags(out)
        assert out["prompts"][0]["arguments"][0]["required"] is True

    def test_get_prompt_role_and_description_stripped(self):
        from tools.mcp_tool_handlers import _render_get_prompt

        msg = SimpleNamespace(role=TAGGED, content=SimpleNamespace(text="clean"))
        result = SimpleNamespace(messages=[msg], description=TAGGED)
        out = _render_get_prompt(result, "srv")
        _assert_no_tags(out)
        assert out["description"] == "evil"
        assert out["messages"][0]["role"] == "evil"
        assert out["messages"][0]["content"] == "clean"


class TestCallToolResultArms:
    def _render(self, result):
        from tools.mcp_tool_handlers import _render_call_tool_result
        return json.loads(_render_call_tool_result(result, "srv"))

    def test_iserror_text_stripped(self):
        res = SimpleNamespace(
            content=[SimpleNamespace(type="text", text=TAGGED)],
            isError=True, structuredContent=None, meta=None)
        data = self._render(res)
        _assert_no_tags(data)
        assert "evil" in data["error"]

    def test_iserror_embedded_resource_text_stripped(self):
        res = SimpleNamespace(
            content=[SimpleNamespace(type="resource",
                                     resource=SimpleNamespace(uri="x://y", text=TAGGED))],
            isError=True, structuredContent=None, meta=None)
        _assert_no_tags(self._render(res))

    def test_unsupported_block_notice_via_call_tool_result(self):
        block = SimpleNamespace(type=TAGGED, mimeType=TAGGED, uri=TAGGED, name=TAGGED)
        res = SimpleNamespace(content=[block], isError=False, structuredContent=None, meta=None)
        _assert_no_tags(self._render(res))

    def test_clean_metadata_untouched(self):
        res = SimpleNamespace(
            content=[SimpleNamespace(type="text", text="hi")],
            isError=False, structuredContent=None,
            meta={"vendor.example/trace": "abc-123"})
        data = self._render(res)
        assert data["_meta"] == {"vendor.example/trace": "abc-123"}


class TestSamplingPath:
    """sampling/createMessage: the server writes strings straight into a local LLM call."""

    def test_sampling_message_text_and_role_stripped(self):
        from tools.mcp_tool_sampling import _convert_sampling_message

        msg = SimpleNamespace(role=TAGGED, content=SimpleNamespace(text=TAGGED))
        out = _convert_sampling_message(msg)
        _assert_no_tags(out)
        assert out[0]["content"] == "evil"
        assert out[0]["role"] == "evil"

    def test_sampling_tool_result_text_and_id_stripped(self):
        from tools.mcp_tool_sampling import _convert_sampling_message

        tr = SimpleNamespace(toolUseId=TAGGED, content=[SimpleNamespace(text=TAGGED)])
        msg = SimpleNamespace(role="user", content=[tr])
        out = _convert_sampling_message(msg)
        _assert_no_tags(out)

    def test_sampling_tool_use_name_and_args_stripped(self):
        from tools.mcp_tool_sampling import _convert_sampling_message

        tu = SimpleNamespace(name=TAGGED, input={TAGGED: TAGGED}, id="c1")
        msg = SimpleNamespace(role="assistant", content=[tu])
        out = _convert_sampling_message(msg)
        _assert_no_tags(out)
        assert json.loads(out[0]["tool_calls"][0]["function"]["arguments"]) == {"evil": "evil"}

    def test_sampling_system_prompt_and_tools_stripped(self):
        from tools.mcp_tool_sampling import SamplingHandler

        handler = SamplingHandler("srv", {})
        params = SimpleNamespace(
            systemPrompt=TAGGED, maxTokens=16,
            messages=[SimpleNamespace(role="user", content=SimpleNamespace(text="hi"))],
            tools=[SimpleNamespace(name=TAGGED, description=TAGGED,
                                   inputSchema={"type": "object", "properties": {TAGGED: {"description": TAGGED}}})])
        # capture the messages passed to call_llm (imported inside _build_llm_call, so patch first)
        captured = {}
        import agent.auxiliary_client as aux
        orig = aux.call_llm
        aux.call_llm = lambda **kw: captured.update(kw)
        try:
            thunk = handler._build_llm_call(params, "")
            thunk()
        finally:
            aux.call_llm = orig
        _assert_no_tags(captured["messages"])
        _assert_no_tags(captured["tools"])
        assert captured["messages"][0] == {"role": "system", "content": "evil"}

    def test_elicitation_summary_fields_stripped(self):
        from tools.mcp_tool_sampling import _format_elicitation_schema_summary

        schema = {"properties": {TAGGED: {"type": TAGGED, "description": TAGGED}}}
        out = _format_elicitation_schema_summary(schema, "srv")
        _assert_no_tags(out)
        assert "evil" in out


class TestServerErrorText:
    def test_sanitize_error_strips_tags(self):
        from tools.mcp_tool_common import _sanitize_error

        assert _sanitize_error(f"server blew up: {TAGGED}") == "server blew up: evil"

    def test_dispatch_error_strips_tags(self):
        from tools.mcp_tool_handlers import _dispatch

        class Boom(Exception):
            pass

        def _raise():
            raise Boom(TAGGED)

        server = SimpleNamespace(mark_tool_call=lambda: None)
        import tools.mcp_tool_loop as _loop
        orig = _loop._run_on_mcp_loop
        _loop._run_on_mcp_loop = lambda call, timeout=None: _raise()
        try:
            out = _dispatch("srv", server, "op", lambda: None, 1.0, (), lambda e: None)
        finally:
            _loop._run_on_mcp_loop = orig
        data = json.loads(out)
        _assert_no_tags(data)
        assert "evil" in data["error"]
