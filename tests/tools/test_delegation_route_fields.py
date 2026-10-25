"""Route fields a delegation backend returns: ``service_tier`` is checked like ``/fast`` and survives child
construction, an unknown ``reasoning_effort`` refuses that task, and a backend-routed result reports what the
child actually ran on. Real ``delegate_task``, real credential resolution and tier gates; only the child's
construction and run are stubbed where a test reads what they were handed.

Tier behaviour ported from #140614 (per-task speed tiers, unsupportedpastels).
"""

import json
import threading
from unittest.mock import MagicMock, patch

import pytest

from agent.delegation_provider import DelegationProvider

CODEX = {"base_url": "https://chatgpt.com/backend-api/codex", "api_key": "codex-key", "model": "gpt-6-astra"}
LOCAL = {"base_url": "http://127.0.0.1:8080/v1", "api_key": "k", "model": "qwen3-local"}


class _Backend(DelegationProvider):
    """Answers task ``index`` with ``routes[index]``."""

    def __init__(self, routes):
        self.routes = routes

    @property
    def name(self):
        return "route-fields"

    def route_task(self, task, config):
        route = self.routes[task["index"]]
        return dict(config) if route is None else dict(route)


def _parent(request_overrides=None):
    parent = MagicMock()
    parent.base_url, parent.api_key, parent.provider = "https://openrouter.ai/api/v1", "parent-key", "openrouter"
    parent.api_mode, parent.model, parent.platform = "chat_completions", "anthropic/claude-sonnet-4", "cli"
    parent.request_overrides = request_overrides or {}
    parent.reasoning_config = {"enabled": True, "effort": "medium"}
    parent.providers_allowed = parent.providers_ignored = parent.providers_order = parent.provider_sort = None
    parent._session_db, parent._delegate_depth, parent._active_children = None, 0, []
    parent._active_children_lock = threading.Lock()
    parent._print_fn = parent.tool_progress_callback = parent.thinking_callback = None
    return parent


def _construct(**kwargs):
    child = MagicMock()
    child.request_overrides = dict(kwargs.get("request_overrides") or {})
    child.provider, child.model = kwargs.get("provider"), kwargs.get("model")
    child.reasoning_config, child.service_tier = kwargs.get("reasoning_config"), kwargs.get("service_tier")
    return child


def _spawn(routes, *, cfg=None, parent=None, construct=_construct, provider=None):
    """``(tool result, [AIAgent kwargs], [built children])`` for one task per route."""
    batches = []
    with (
        patch("tools.delegate_tool._load_config", return_value=cfg or {}),
        patch("tools.delegate_tool_routes._active_provider", return_value=provider or _Backend(routes)),
        patch("run_agent.AIAgent", side_effect=construct) as agent,
        patch("tools.delegate_tool._run_batch",
              side_effect=lambda batch, background: batches.append(batch) or json.dumps(
                  {"status": "built", **({"refused": batch.refused} if batch.refused else {})})),
    ):
        from tools.delegate_tool import delegate_task
        result = json.loads(delegate_task(tasks=[{"goal": f"task number {i}"} for i in range(len(routes))],
                                          parent_agent=parent or _parent()))
    children = [c for _, _, c in batches[0].children] if batches else []
    return result, [call.kwargs for call in agent.call_args_list], children


def test_route_tier_reaches_the_child_and_its_sibling_keeps_the_default():
    result, (fast, plain), _ = _spawn([{**CODEX, "service_tier": "fast"}, None])
    assert result == {"status": "built"}
    assert fast["request_overrides"]["service_tier"] == "priority" and fast["service_tier"] == "priority"
    assert "service_tier" not in plain["request_overrides"] and plain.get("service_tier") is None


@pytest.mark.parametrize("route, needle", [
    ({**LOCAL, "service_tier": "fast"}, "not available"),  # a local route has no fast tier: never a silent normal
    ({**CODEX, "service_tier": "turbo"}, "turbo"),
    ({**CODEX, "service_tier": "auto"}, "auto"),  # bounded windows are per user turn, meaningless for a child
])
def test_a_tier_the_route_does_not_offer_refuses_that_task_and_its_sibling_runs(route, needle):
    from tools.delegation_live_log import _manifest_path

    result, built, (sibling,) = _spawn([route, None])
    (refused,) = result["refused"]
    assert refused["task_index"] == 0 and needle in refused["error"] and len(built) == 1
    # Before the sibling runs (async units never carry it), the batch manifest already shows the refusal.
    manifest = json.loads(_manifest_path(sibling._delegation_id, None).read_text(encoding="utf-8"))
    assert [t["status"] for t in manifest["tasks"]] == ["error", "running"] and "completed" not in manifest


@pytest.mark.parametrize("account_offers, expected", [(False, None), (True, "ultrafast"), (None, "ultrafast")])
def test_ultrafast_follows_the_codex_account_like_the_fast_command(account_offers, expected):
    """The Codex backend serves another tier to a plan without ultrafast; refuse unless the account lists it."""
    with patch("agent.codex_service_tiers.codex_account_offers_tier", return_value=account_offers):
        result, built, _ = _spawn([{**CODEX, "model": "gpt-6.1-sol", "service_tier": "ultrafast"}])
    if expected is None:
        assert "not available" in result["error"] and built == []
    else:
        assert built[0]["request_overrides"]["service_tier"] == expected


def test_ultrafast_is_checked_against_the_routes_own_account():
    """A backend's Codex route may carry another account's key: the ultrafast entitlement is read for that key,
    never the profile's stored login."""
    seen = []

    def offers(model, tier, *, base_url=None, access_token=None):
        seen.append(access_token)
        return access_token == "routed-account-key"

    with patch("agent.codex_service_tiers.codex_account_offers_tier", side_effect=offers):
        result, built, _ = _spawn([{**CODEX, "api_key": "routed-account-key", "model": "gpt-6.1-sol",
                                    "service_tier": "ultrafast"}])
    assert seen == ["routed-account-key"]
    assert result == {"status": "built"} and built[0]["request_overrides"]["service_tier"] == "ultrafast"


def test_route_tier_survives_a_provider_extra_body_merged_at_construction():
    """Child setup can merge a custom provider's extra_body (carrying its own tier) after the route's overrides."""
    def construct(**kwargs):
        child = _construct(**kwargs)
        child.request_overrides = {**child.request_overrides, "extra_body": {"service_tier": "priority", "x": 1}}
        return child

    parent = _parent(request_overrides={"service_tier": "priority"})
    _, _, (child,) = _spawn([{**CODEX, "service_tier": "normal"}], parent=parent, construct=construct)
    assert child.request_overrides == {"extra_body": {"x": 1}}
    assert parent.request_overrides == {"service_tier": "priority"}  # the parent is untouched


def test_default_backend_never_reads_a_tier_from_the_delegation_section():
    """``service_tier`` is not a delegation-section key: the default route stays exactly today's."""
    from agent.delegation_registry import get_active_provider
    result, (child,), _ = _spawn([None], cfg={**CODEX, "service_tier": "fast"}, provider=get_active_provider({}))
    assert result == {"status": "built"}
    assert child.get("service_tier") is None and "service_tier" not in child["request_overrides"]


def test_unknown_route_reasoning_effort_refuses_the_task_never_the_parents_level():
    result, built, _ = _spawn([{**LOCAL, "reasoning_effort": "hgih"}, None])
    (refused,) = result["refused"]
    assert "hgih" in refused["error"] and "xhigh" in refused["error"] and len(built) == 1


def test_the_delegation_sections_own_effort_keeps_its_warn_and_inherit_rule():
    """A backend that falls back to ``dict(config)`` carries the config's value; config behaviour is unchanged."""
    result, (child,), _ = _spawn([None], cfg={"reasoning_effort": "hgih"})
    assert result == {"status": "built"} and child["reasoning_config"] == {"enabled": True, "effort": "medium"}


def test_labelled_route_result_reports_what_the_child_ran_on():
    from tools.delegate_tool_child_run import _SchemaOutcome, _build_result_entry, _fabricated_entry

    _, _, (deep, local) = _spawn([{**CODEX, "reasoning_effort": "high", "service_tier": "fast", "label": "deep"},
                                  {**LOCAL, "reasoning_effort": "none", "label": "local"}])
    entry = _build_result_entry(deep, {"final_response": "ok", "completed": True}, 0, 1.0,
                                _SchemaOutcome(None, None, [], 0))
    assert entry["route"] == {"backend": "route-fields", "label": "deep", "provider": "openai-codex",
                              "model": "gpt-6-astra", "reasoning_effort": "high", "service_tier": "fast"}
    assert _fabricated_entry(1, "error", "boom", local)["route"] == {
        "backend": "route-fields", "label": "local", "provider": "custom", "model": "qwen3-local",
        "reasoning_effort": "none", "service_tier": "normal"}


def test_every_backend_routed_result_names_its_backend_and_the_default_keeps_its_shape():
    """An unlabelled backend route still reports its backend and what ran; the default backend's children leave the
    result entry exactly as before."""
    from agent.delegation_registry import get_active_provider
    from tools.delegate_tool_child_run import _SchemaOutcome, _build_result_entry

    def entry_of(child):
        return _build_result_entry(child, {"final_response": "ok", "completed": True}, 0, 1.0,
                                   _SchemaOutcome(None, None, [], 0))

    _, _, (routed,) = _spawn([{**LOCAL, "reasoning_effort": "high"}])
    assert entry_of(routed)["route"] == {"backend": "route-fields", "provider": "custom", "model": "qwen3-local",
                                         "reasoning_effort": "high", "service_tier": "normal"}
    _, _, (plain,) = _spawn([None], cfg=LOCAL, provider=get_active_provider({}))
    assert "route" not in entry_of(plain)


def test_an_inherited_fast_tier_is_reported_as_what_ran():
    """An empty backend route inherits the parent's request settings, fast tier included: the report says fast."""
    from tools.delegate_tool_child_run import _SchemaOutcome, _build_result_entry

    _, _, (child,) = _spawn([{"label": "inherit"}], parent=_parent(request_overrides={"service_tier": "priority"}))
    entry = _build_result_entry(child, {"final_response": "ok", "completed": True}, 0, 1.0,
                                _SchemaOutcome(None, None, [], 0))
    assert entry["route"]["service_tier"] == "fast"
