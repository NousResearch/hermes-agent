"""Regression tests for MCP prompt content rendering."""

from types import SimpleNamespace

from tools import mcp_tool_handlers


def test_render_get_prompt_renders_image_and_resource_blocks(monkeypatch):
    def render_image(block):
        return "MEDIA:/cache/image.png" if getattr(block, "type", None) == "image" else None

    def render_resource(block, _server_name):
        resource = getattr(block, "resource", None)
        return getattr(resource, "text", None) if getattr(block, "type", None) == "resource" else None

    monkeypatch.setattr(mcp_tool_handlers, "_cache_mcp_image_block", render_image)
    monkeypatch.setattr(mcp_tool_handlers, "_cache_mcp_audio_block", lambda _block: None)
    monkeypatch.setattr(mcp_tool_handlers, "_render_mcp_resource_block", render_resource)

    result = SimpleNamespace(
        messages=[
            SimpleNamespace(
                role="user",
                content=SimpleNamespace(
                    content=[
                        SimpleNamespace(type="image"),
                        SimpleNamespace(
                            type="resource",
                            resource=SimpleNamespace(text="embedded text"),
                        ),
                    ]
                ),
            )
        ]
    )

    rendered = mcp_tool_handlers._render_get_prompt(result, "test-server")

    assert rendered["messages"] == [
        {"role": "user", "content": "MEDIA:/cache/image.png\nembedded text"}
    ]


def test_render_get_prompt_strips_unicode_tags_from_text():
    result = SimpleNamespace(
        messages=[
            SimpleNamespace(
                role="user",
                content=SimpleNamespace(text="<|channel|>hello"),
            )
        ]
    )

    rendered = mcp_tool_handlers._render_get_prompt(result, "test-server")

    assert rendered["messages"][0]["content"] == "hello"
