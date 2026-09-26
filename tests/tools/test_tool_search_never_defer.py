"""``tools.tool_search.never_defer``: a plugin or MCP tool can be kept eager.

``defer`` cannot express this — plugin and MCP tools defer whatever the list says."""

from tools.registry import registry
from tools.tool_search import ToolSearchConfig, assemble_tool_defs


def _td(name):
    return {"type": "function", "function": {
        "name": name, "description": f"{name} capability", "parameters": {"type": "object", "properties": {}}}}


def test_plugin_tool_named_in_never_defer_stays_visible():
    for name, toolset in (("chat_probe_send", "plugin-chat-probe"), ("chat_probe_other", "plugin-chat-probe")):
        registry.register(name=name, handler=lambda args, **kw: "{}", schema=_td(name)["function"], toolset=toolset)
    defs = [_td("terminal"), _td("chat_probe_send"), _td("chat_probe_other")]

    config = ToolSearchConfig.from_raw({"enabled": "on", "defer": [], "never_defer": ["chat_probe_send"]})
    names = {td["function"]["name"] for td in assemble_tool_defs(defs, context_length=200_000, config=config).tool_defs}
    assert "chat_probe_send" in names
    assert "chat_probe_other" not in names  # positive control: the sibling plugin tool still defers
