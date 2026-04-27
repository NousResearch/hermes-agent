import json
from types import SimpleNamespace


def test_multimodal_recall_handler_wraps_mcp(monkeypatch):
    import tools.multimodal_recall_tool as mrt

    class DummySession:
        async def call_tool(self, tool_name, arguments=None):
            return SimpleNamespace(
                isError=False,
                content=[SimpleNamespace(text=json.dumps({"tool": tool_name, "arguments": arguments}))],
                structuredContent=None,
            )

    monkeypatch.setattr(mrt.mcp_tool, '_servers', {'local_mmrag': SimpleNamespace(session=DummySession())})
    monkeypatch.setattr(mrt.mcp_tool, '_run_on_mcp_loop', lambda coro, timeout=30: __import__('asyncio').run(coro))

    result = json.loads(mrt.multimodal_recall(action='search', query='customer waiting', top_k=3))
    assert result['tool'] == 'mm_recall_search'
    assert result['arguments']['query'] == 'customer waiting'
    assert result['arguments']['top_k'] == 3


def test_multimodal_recall_evidence_mode(monkeypatch):
    import tools.multimodal_recall_tool as mrt

    class DummySession:
        async def call_tool(self, tool_name, arguments=None):
            return SimpleNamespace(
                isError=False,
                content=[SimpleNamespace(text=json.dumps({"tool": tool_name, "arguments": arguments}))],
                structuredContent=None,
            )

    monkeypatch.setattr(mrt.mcp_tool, '_servers', {'local_mmrag': SimpleNamespace(session=DummySession())})
    monkeypatch.setattr(mrt.mcp_tool, '_run_on_mcp_loop', lambda coro, timeout=30: __import__('asyncio').run(coro))

    result = json.loads(mrt.multimodal_recall(action='evidence', query='customer waiting', source_ref='issue:123', modality='image'))
    assert result['tool'] == 'mm_recall_get_evidence_pack'
    assert result['arguments']['source_ref'] == 'issue:123'
    assert result['arguments']['modality'] == 'image'
