"""``skills.manage`` search returns each hub row's description.

``SkillMeta`` (``tools/skills_hub_models.py``) is a dataclass carrying ``description`` and it has
no ``describe()``, so calling one on a search row failed the whole RPC: ``skills.manage`` answered
``-5024 'SkillMeta' object has no attribute 'describe'`` instead of results, for every query.

The near-identical call in ``command.resolve`` is *correct*, which is why this is not a blanket
"stop calling describe()": ``CommandDef.describe()`` (``hermes_cli/commands.py``) is a localized
accessor, while these hub rows come from search sources that never defined one. This test pins the
distinction so a later sweep cannot "unify" the two and re-break search.

Regression: ``9bcbe7b5df9`` (feat(i18n), #126296) rewrote both call sites from the attribute to a
call; only the ``CommandDef`` one had a method to reach.
"""

from __future__ import annotations


def test_skills_search_returns_each_rows_description(monkeypatch):
    from tools.skills_hub_models import SkillMeta

    from tui_gateway import server

    row = SkillMeta(name="proof-skill", description="a description that must survive the trip",
                    source="official", identifier="proof/skill", trust_level="builtin")
    # Patch the real hub seams rather than the handler's module: method bodies are bound to the
    # server's globals (method_ctx.bind_module), so patching tui_gateway.methods_tools has no
    # effect on what the registered handler resolves.
    monkeypatch.setattr("tools.skills_hub_search.unified_search", lambda *a, **k: [row])
    monkeypatch.setattr("tools.skills_hub_search.create_source_router", lambda *a, **k: object())
    monkeypatch.setattr("tools.skills_hub_github.GitHubAuth", lambda *a, **k: object())

    reply = server._methods["skills.manage"](1, {"action": "search", "query": "proof"})

    assert "error" not in reply, reply.get("error")
    assert reply["result"]["results"] == [
        {"name": "proof-skill", "description": "a description that must survive the trip"}
    ]


def test_skills_search_reply_is_empty_without_rows(monkeypatch):
    """No hub rows is an empty list, not an error: the handler's ``or []`` must survive."""
    from tui_gateway import server

    monkeypatch.setattr("tools.skills_hub_search.unified_search", lambda *a, **k: None)
    monkeypatch.setattr("tools.skills_hub_search.create_source_router", lambda *a, **k: object())
    monkeypatch.setattr("tools.skills_hub_github.GitHubAuth", lambda *a, **k: object())

    reply = server._methods["skills.manage"](1, {"action": "search", "query": "nothing"})

    assert "error" not in reply, reply.get("error")
    assert reply["result"]["results"] == []
