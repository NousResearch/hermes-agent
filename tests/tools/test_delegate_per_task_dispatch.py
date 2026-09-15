#!/usr/bin/env python3
"""Per-task ``model`` / ``provider`` / ``reasoning_effort`` on ``delegate_task`` (#94887).

Extends the per-task provider/model pin (PR #107945) with a reasoning pin and cross-provider
batches. Contract under test:

* absent keys  -> byte-identical to the pre-change behaviour (batch creds, delegation.reasoning_effort),
* a pin        -> resolved through the same ``_resolve_delegation_credentials`` path as CLI/config,
                  cached per distinct ``(provider, model)`` pair within one call,
* a bad pin    -> the WHOLE batch aborts naming the task index; no child is constructed or run,
* any pin      -> the child counts as pinned, so it never borrows the parent's fallback chain,
* api_mode     -> derived from the pinned (provider, model) whenever either is pinned; a child that
                  pinned neither still inherits the parent's wire verbatim.
"""

import json
import threading
from unittest.mock import MagicMock, patch

import pytest

from hermes_cli.models import opencode_model_api_mode
from tools.delegate_tool import DELEGATE_TASK_SCHEMA, _resolve_task_credentials, delegate_task
from tools.delegate_tool_toolsets import _is_mcp_toolset_name, _resolve_child_toolsets

BATCH_CREDS = {
    "model": "deepseek-v4-flash-0731", "provider": "alibaba", "base_url": "https://alibaba/v1",
    "api_key": "alibaba-key", "api_mode": "chat_completions", "request_overrides": None,
}
GO_CREDS = {
    "model": "grok-4.6", "provider": "opencode-go", "base_url": "https://opencode.ai/v1",
    "api_key": "go-key", "api_mode": "codex_responses", "request_overrides": None,
}
INHERIT_CREDS = {
    "model": None, "provider": None, "base_url": None, "api_key": None, "api_mode": None,
    "request_overrides": None,
}
# The default batch route under test: delegation.provider is configured, so every child is already pinned.
# The operator's trust table. Alibaba is Denis's own brain provider and is granted everything the
# orchestrator holds; OpenCode Go is the cheap lane, granted one harmless toolset and nothing else.
PROVIDER_TOOLSETS = {
    "alibaba": ["memory-mcp", "mcp-scratchpad", "session_search", "skills", "todo"],
    "opencode-go": ["todo"],
}
DELEGATION_CFG = {"provider": "alibaba", "model": "deepseek-v4-flash-0731", "base_url": "https://alibaba/v1",
                  "provider_toolsets": PROVIDER_TOOLSETS}
PARENT_CHAIN = [{"provider": "openrouter", "model": "parent-fallback"}]
# The parent every test here is measured against. Seeded 2026-09-14 from forge, the live orchestrator
# profile, and deliberately NOT tracking it since: a unit test that reads a production profile breaks in
# CI, on the Mac, and for any upstream contributor. What has to stay true of this list is structural:
#   * >=2 MCP toolsets, one spelled as a bare registered alias and one canonically — the re-add loop in
#     _resolve_child_toolsets iterates EVERY parent MCP toolset, and a single entry cannot tell
#     "suppresses all of them" from "suppresses the one we happened to name";
#   * >=1 toolset blocked for every child, so the asymmetry this patch exists for stays visible;
#   * >=1 plain inheritable toolset to narrow down to.
VAULT_TOOLSET = "mcp-memory-mcp"  # canonical name of the server holding Denis's Obsidian vault
VAULT_ALIAS = "memory-mcp"  # how forge spells it: a bare alias, resolvable only through the registry
SECOND_MCP_TOOLSET = "mcp-scratchpad"  # synthetic second server, spelled the canonical way
PARENT_TOOLSETS = ["clarify", "delegation", "kanban", "memory", SECOND_MCP_TOOLSET, VAULT_ALIAS,
                   "session_search", "skills", "todo"]
# What a child ACTUALLY inherits: "delegation"/"kanban"/"memory"/"clarify" are stripped for every child,
# pinned or not — but the MCP toolsets, one of which holds the vault, ride straight through. That
# asymmetry is why per-task ``toolsets`` has to exist.
INHERITED_TOOLSETS = [SECOND_MCP_TOOLSET, VAULT_ALIAS, "session_search", "skills", "todo"]
BAD_PROVIDER_ERROR = "Cannot resolve delegation provider 'not-a-real-provider'"
LONG_GOAL = "Research the delegation topic thoroughly"


@pytest.fixture(autouse=True)
def warm_mcp_alias():
    """Register the vault's bare-name alias, as a connected MCP server does at startup.

    Without it ``_is_mcp_toolset_name("memory-mcp")`` is False offline, nothing in the parent set counts
    as an MCP toolset, and the exact=True/exact=False paths agree for the wrong reason — a suite that
    cannot show the vault arriving cannot prove it was removed.
    """
    from tools.registry import registry
    before = registry.get_registered_toolset_aliases()
    registry.register_toolset_alias(VAULT_ALIAS, VAULT_TOOLSET)
    yield
    registry.unregister_toolset_alias(VAULT_ALIAS)
    for alias, target in before.items():
        registry.register_toolset_alias(alias, target)


def _parent(**attrs):
    """Mock parent with every attribute the child-construction path actually reads."""
    parent = MagicMock()
    parent.base_url, parent.api_key = "https://openrouter.ai/api/v1", "parent-key"
    parent.provider, parent.api_mode = "openrouter", "chat_completions"
    parent.model, parent.platform = "parent-model", "cli"
    parent.providers_allowed = parent.providers_ignored = parent.providers_order = parent.provider_sort = None
    parent.provider_require_parameters, parent.provider_data_collection = False, ""
    parent.openrouter_min_coding_score = parent.max_tokens = None
    parent.reasoning_config = {"enabled": True, "effort": "medium"}
    parent.request_overrides = None
    parent._fallback_chain = list(PARENT_CHAIN)
    parent._session_db = None
    parent._delegate_depth = 0
    parent._active_children = []
    parent._active_children_lock = threading.Lock()
    parent._print_fn = parent.tool_progress_callback = parent.thinking_callback = None
    parent.acp_command, parent.acp_args = None, []
    parent.enabled_toolsets, parent.disabled_toolsets = list(PARENT_TOOLSETS), []
    for k, v in attrs.items():
        setattr(parent, k, v)
    return parent


def _fake_resolve(cfg, parent_agent):
    """Stand-in for ``_resolve_delegation_credentials``, keyed on the overlay it is handed."""
    provider = str((cfg or {}).get("provider") or "").strip()
    model = str((cfg or {}).get("model") or "").strip()
    if provider == "not-a-real-provider":
        raise ValueError(BAD_PROVIDER_ERROR)
    if provider == "opencode-go":
        return {**GO_CREDS, "model": model or GO_CREDS["model"],
                "api_mode": opencode_model_api_mode(provider, model or GO_CREDS["model"])}
    if not provider:  # pure-inherit branch: the child takes the parent's route
        return {**INHERIT_CREDS, "model": model or None}
    return {**BATCH_CREDS, "provider": provider, "model": model or BATCH_CREDS["model"]}


def _ok_child():
    child = MagicMock()
    child.run_conversation.return_value = {"final_response": "ok", "completed": True, "api_calls": 1}
    return child


def _cfg(**overrides):
    """The default delegation block with per-test overrides."""
    return {**DELEGATION_CFG, **overrides}


def _run(tasks, *, delegation_cfg=None, parent=None):
    """Dispatch a batch with credential resolution and child construction stubbed out.

    Returns ``(payload, MockAgent, resolve_calls)`` — ``resolve_calls`` is every routing config the
    credential resolver was handed, so call-count and overlay assertions are possible.
    """
    parent = parent or _parent()
    resolve_calls = []

    def tracking_resolve(cfg, parent_agent):
        resolve_calls.append(dict(cfg) if isinstance(cfg, dict) else cfg)
        return _fake_resolve(cfg, parent_agent)

    with (
        patch("tools.delegate_tool._load_config",
              return_value=dict(_cfg() if delegation_cfg is None else delegation_cfg)),
        patch("tools.delegate_tool._resolve_delegation_credentials", side_effect=tracking_resolve),
        patch("run_agent.AIAgent") as MockAgent,
    ):
        MockAgent.side_effect = lambda **kw: _ok_child()
        raw = delegate_task(tasks=tasks, parent_agent=parent)
    return json.loads(raw), MockAgent, resolve_calls


def _child_kwargs(MockAgent, index):
    return MockAgent.call_args_list[index].kwargs


# -- regression guard: no per-task keys -------------------------------------------------------


def test_no_per_task_keys_uses_batch_creds_only():
    """Every child on the batch route, one credential resolution for the whole call."""
    _, MockAgent, resolve_calls = _run([{"goal": f"{LONG_GOAL} {n}"} for n in range(3)])

    assert MockAgent.call_count == 3
    for i in range(3):
        kw = _child_kwargs(MockAgent, i)
        assert (kw["provider"], kw["model"], kw["base_url"]) == ("alibaba", "deepseek-v4-flash-0731", "https://alibaba/v1")
        assert kw["api_mode"] == "chat_completions"
    assert len(resolve_calls) == 1


def test_no_per_task_keys_inherits_parent_fallback_chain():
    """Unpinned children still borrow the parent chain (pinned=False path)."""
    _, MockAgent, _ = _run([{"goal": LONG_GOAL}], delegation_cfg={})  # no delegation.provider/model
    assert _child_kwargs(MockAgent, 0)["fallback_model"] == PARENT_CHAIN


def test_no_per_task_keys_keeps_delegation_reasoning_effort():
    _, MockAgent, _ = _run([{"goal": LONG_GOAL}], delegation_cfg=_cfg(reasoning_effort="high"))
    assert _child_kwargs(MockAgent, 0)["reasoning_config"] == {"enabled": True, "effort": "high"}


# -- per-task model, within the batch provider ------------------------------------------------


def test_per_task_model_only_keeps_batch_provider():
    _, MockAgent, resolve_calls = _run([
        {"goal": f"{LONG_GOAL} strong", "model": "qwen3.8-max"},
        {"goal": f"{LONG_GOAL} cheap"},
    ])

    pinned, inherited = _child_kwargs(MockAgent, 0), _child_kwargs(MockAgent, 1)
    assert (pinned["model"], pinned["provider"]) == ("qwen3.8-max", "alibaba")
    assert (inherited["model"], inherited["provider"]) == ("deepseek-v4-flash-0731", "alibaba")
    # The overlay carries the pinned model, so model-derived routing sees the right target.
    assert resolve_calls[-1]["model"] == "qwen3.8-max"


def test_blank_pins_inherit_the_batch_route():
    """Empty / whitespace-only / non-string pins are not pins."""
    _, MockAgent, resolve_calls = _run([{"goal": LONG_GOAL, "provider": "  ", "model": ""}])
    kw = _child_kwargs(MockAgent, 0)
    assert (kw["provider"], kw["model"]) == ("alibaba", "deepseek-v4-flash-0731")
    assert len(resolve_calls) == 1


# -- cross-provider batches -------------------------------------------------------------------


def test_cross_provider_batch_routes_each_child_separately():
    """Two providers in one call: each child gets its own credentials, endpoint and wire."""
    _, MockAgent, _ = _run([
        {"goal": f"{LONG_GOAL} on go", "provider": "opencode-go", "model": "grok-4.6", "toolsets": ["todo"]},
        {"goal": f"{LONG_GOAL} on alibaba", "provider": "alibaba", "model": "qwen3.8-max", "toolsets": ["todo"]},
    ])

    go, ali = _child_kwargs(MockAgent, 0), _child_kwargs(MockAgent, 1)
    assert (go["provider"], go["model"], go["base_url"]) == ("opencode-go", "grok-4.6", "https://opencode.ai/v1")
    assert (ali["provider"], ali["model"], ali["base_url"]) == ("alibaba", "qwen3.8-max", "https://alibaba/v1")


def test_provider_pin_drops_a_configured_batch_endpoint():
    """A leftover delegation.base_url must not follow the child onto a different provider."""
    calls = []

    def capture(cfg, parent_agent):
        calls.append(dict(cfg))
        return _fake_resolve(cfg, parent_agent)

    with patch("tools.delegate_tool._resolve_delegation_credentials", side_effect=capture):
        routing = {"provider": "alibaba", "base_url": "https://alibaba/v1", "model": "deepseek-v4-flash-0731"}
        _resolve_task_credentials({"goal": LONG_GOAL, "provider": "opencode-go"}, dict(BATCH_CREDS),
                                  routing, _parent(), {})
    assert calls[0]["base_url"] == ""
    assert calls[0]["provider"] == "opencode-go"


def test_credentials_resolve_once_per_distinct_route():
    """N tasks sharing a route resolve once; the batch route is resolved once more, up front."""
    _, MockAgent, resolve_calls = _run([
        {"goal": f"{LONG_GOAL} a", "provider": "opencode-go", "model": "grok-4.6", "toolsets": ["todo"]},
        {"goal": f"{LONG_GOAL} b", "provider": "opencode-go", "model": "grok-4.6", "toolsets": ["todo"]},
        {"goal": f"{LONG_GOAL} c", "provider": "opencode-go", "model": "grok-4.6", "toolsets": ["todo"]},
        {"goal": f"{LONG_GOAL} d"},
    ])

    assert MockAgent.call_count == 4
    # 1 batch-level + 1 for the single distinct pin, NOT one per pinned task.
    assert len(resolve_calls) == 2


def test_distinct_models_on_one_provider_resolve_separately():
    """Cache key is (provider, model): a shared provider must not share a model-derived wire."""
    _, MockAgent, resolve_calls = _run([
        {"goal": f"{LONG_GOAL} codex", "provider": "opencode-go", "model": "grok-4.6", "toolsets": ["todo"]},
        {"goal": f"{LONG_GOAL} messages", "provider": "opencode-go", "model": "qwen3.8-max", "toolsets": ["todo"]},
    ])

    assert len(resolve_calls) == 3  # batch + two distinct (provider, model) pairs
    assert _child_kwargs(MockAgent, 0)["api_mode"] == "codex_responses"
    assert _child_kwargs(MockAgent, 1)["api_mode"] == "anthropic_messages"


# -- api_mode per provider/model --------------------------------------------------------------


@pytest.mark.parametrize("model,expected", [
    ("grok-4.6", "codex_responses"),        # gpt-/grok-/muse-spark family
    ("muse-spark", "codex_responses"),
    ("qwen3.8-max", "anthropic_messages"),  # minimax-/qwen family
    ("minimax-m3", "anthropic_messages"),
    ("claude-opus-4.8", "chat_completions"),  # everything else on opencode-go
])
def test_opencode_api_mode_is_derived_per_pinned_model(model, expected):
    """All three OpenCode wire families; a wrong mode here is a silent 404 in production."""
    _, MockAgent, _ = _run([{"goal": LONG_GOAL, "provider": "opencode-go", "model": model, "toolsets": ["todo"]}])
    assert _child_kwargs(MockAgent, 0)["api_mode"] == expected


def test_pinned_api_mode_does_not_leak_to_an_unpinned_sibling():
    _, MockAgent, _ = _run([
        {"goal": f"{LONG_GOAL} pinned", "provider": "opencode-go", "model": "qwen3.8-max", "toolsets": ["todo"]},
        {"goal": f"{LONG_GOAL} plain"},
    ])
    assert _child_kwargs(MockAgent, 0)["api_mode"] == "anthropic_messages"
    assert _child_kwargs(MockAgent, 1)["api_mode"] == "chat_completions"


def _runtime(parent, **overrides):
    """``_resolve_child_runtime`` kwargs for a child, with nothing pinned unless the test pins it."""
    from tools.delegate_tool_config import _resolve_child_runtime
    kwargs = dict(
        model=None, override_provider=None, override_base_url=None, override_api_key=None,
        override_api_mode=None, override_acp_command=None, override_acp_args=None,
    )
    kwargs.update(overrides)
    return _resolve_child_runtime(parent, {}, "parent-key", **kwargs)


def test_model_only_pin_rederives_the_wire_on_a_multi_wire_provider():
    """OpenCode picks the wire from the MODEL prefix, so staying on the parent's provider is not enough
    to stay on the parent's wire. Inheriting it sends a codex_responses model to /v1/messages — a 404
    with no hint that a pin caused it."""
    parent = _parent(provider="opencode-go", model="qwen3-max", api_mode="anthropic_messages")
    assert opencode_model_api_mode("opencode-go", "qwen3-max") == "anthropic_messages", "fixture guard"
    assert _runtime(parent, model="grok-4.6")["api_mode"] == "codex_responses"


def test_model_only_pin_leaves_a_single_wire_provider_alone():
    parent = _parent(provider="alibaba", model="deepseek-v4-flash-0731", api_mode="chat_completions")
    assert _runtime(parent, model="qwen3.8-max")["api_mode"] == "chat_completions"


def test_no_pin_inherits_the_parents_wire_verbatim_even_on_a_multi_wire_provider():
    """The gate on the re-derivation: a child that pinned nothing must behave exactly as before, which
    includes keeping an api_mode an operator set explicitly against the prefix table."""
    parent = _parent(provider="opencode-go", model="grok-4.6", api_mode="anthropic_messages")
    assert _runtime(parent)["api_mode"] == "anthropic_messages"


# -- what a provider pin must NOT carry across -------------------------------------------------


def test_provider_pin_drops_the_batch_request_overrides():
    """``request_overrides`` is merged OVER the pinned provider's own, so a batch-level extra_body tuned
    for one provider's wire would silently override the new provider's. Dropped with the base_url."""
    _, _, calls = _run_toolsets(
        [{"goal": LONG_GOAL, "provider": "opencode-go", "model": "grok-4.6", "toolsets": ["todo"]}],
        delegation_cfg=_cfg(request_overrides={"extra_body": {"enable_thinking": True}}),
    )
    pinned = [c for c in calls if c.get("provider") == "opencode-go"]
    assert pinned, "the pinned route never reached the resolver"
    assert all(c.get("request_overrides") is None and c.get("base_url") == "" for c in pinned)


def test_model_only_pin_keeps_the_batch_request_overrides():
    """Nothing crossed a provider boundary, so the batch's own tuning still applies."""
    overrides = {"extra_body": {"enable_thinking": True}}
    _, _, calls = _run_toolsets([{"goal": LONG_GOAL, "model": "qwen3.8-max"}],
                                delegation_cfg=_cfg(request_overrides=overrides))
    assert all(c.get("request_overrides") == overrides for c in calls)


# -- reasoning_effort precedence --------------------------------------------------------------


def test_task_reasoning_effort_beats_config():
    _, MockAgent, _ = _run(
        [{"goal": f"{LONG_GOAL} deep", "reasoning_effort": "xhigh"}, {"goal": f"{LONG_GOAL} plain"}],
        delegation_cfg=_cfg(reasoning_effort="low"),
    )
    assert _child_kwargs(MockAgent, 0)["reasoning_config"] == {"enabled": True, "effort": "xhigh"}
    assert _child_kwargs(MockAgent, 1)["reasoning_config"] == {"enabled": True, "effort": "low"}


def test_config_reasoning_effort_beats_parent():
    _, MockAgent, _ = _run([{"goal": LONG_GOAL}], delegation_cfg=_cfg(reasoning_effort="low"))
    assert _child_kwargs(MockAgent, 0)["reasoning_config"] == {"enabled": True, "effort": "low"}


def test_parent_reasoning_inherited_when_nothing_pins_it():
    _, MockAgent, _ = _run([{"goal": LONG_GOAL}])
    assert _child_kwargs(MockAgent, 0)["reasoning_config"] == {"enabled": True, "effort": "medium"}


@pytest.mark.parametrize("value", [False, "false", "none", "disabled"])
def test_task_reasoning_effort_false_disables_thinking(value):
    """A task-level false must DISABLE thinking, never coerce to empty and inherit."""
    _, MockAgent, _ = _run([{"goal": LONG_GOAL, "reasoning_effort": value}],
                           delegation_cfg=_cfg(reasoning_effort="high"))
    assert _child_kwargs(MockAgent, 0)["reasoning_config"] == {"enabled": False}


def test_unknown_task_reasoning_effort_aborts_the_batch():
    """A per-task pin is a claim about THIS task; silently running it at another level is a wrong answer
    delivered confidently. Unlike the config default, it aborts."""
    payload, MockAgent, _ = _run(
        [{"goal": f"{LONG_GOAL} fine"}, {"goal": f"{LONG_GOAL} typo", "reasoning_effort": "turbo"}],
        delegation_cfg=_cfg(reasoning_effort="low"),
    )
    assert "Task 1" in payload["error"] and "turbo" in payload["error"]
    assert "xhigh" in payload["error"]  # the message lists what it would have accepted
    MockAgent.assert_not_called()


@pytest.mark.parametrize("value", [0, 0.0, [], {}, 7])
def test_non_string_task_reasoning_effort_aborts_the_batch(value):
    """0 / [] / {} are falsy but are not blank: treating them as "nothing was pinned" would inherit a
    thinking level the caller never asked for, which is the exact failure the abort exists to prevent."""
    payload, MockAgent, _ = _run([{"goal": LONG_GOAL, "reasoning_effort": value}],
                                 delegation_cfg=_cfg(reasoning_effort="low"))
    assert "Task 0" in payload["error"] and "reasoning_effort" in payload["error"]
    MockAgent.assert_not_called()


def test_blank_task_reasoning_effort_is_not_a_pin():
    """Absent/blank still falls back to delegation.reasoning_effort — only a real value is a promise."""
    _, MockAgent, _ = _run([{"goal": LONG_GOAL, "reasoning_effort": ""}],
                           delegation_cfg=_cfg(reasoning_effort="low"))
    assert _child_kwargs(MockAgent, 0)["reasoning_config"] == {"enabled": True, "effort": "low"}


def test_unknown_config_reasoning_effort_still_warns_and_inherits():
    """Scope guard: delegation.reasoning_effort is a batch default, not a per-task claim. Unchanged."""
    _, MockAgent, _ = _run([{"goal": LONG_GOAL}], delegation_cfg=_cfg(reasoning_effort="turbo"))
    assert _child_kwargs(MockAgent, 0)["reasoning_config"] == {"enabled": True, "effort": "medium"}


def test_reasoning_pin_is_independent_of_route_pin():
    """reasoning_effort alone must not change the route; model alone must not change reasoning."""
    _, MockAgent, resolve_calls = _run(
        [{"goal": f"{LONG_GOAL} think", "reasoning_effort": "high"},
         {"goal": f"{LONG_GOAL} route", "model": "qwen3.8-max"}],
        delegation_cfg=_cfg(reasoning_effort="low"),
    )
    think, route = _child_kwargs(MockAgent, 0), _child_kwargs(MockAgent, 1)
    assert (think["model"], think["reasoning_config"]) == ("deepseek-v4-flash-0731", {"enabled": True, "effort": "high"})
    assert (route["model"], route["reasoning_config"]) == ("qwen3.8-max", {"enabled": True, "effort": "low"})
    assert len(resolve_calls) == 2  # only the model pin needed its own resolution


# -- fail loud --------------------------------------------------------------------------------


def test_bad_provider_aborts_the_batch_naming_the_task():
    payload, MockAgent, _ = _run([
        {"goal": f"{LONG_GOAL} good"},
        {"goal": f"{LONG_GOAL} bad", "provider": "not-a-real-provider"},
        {"goal": f"{LONG_GOAL} also good"},
    ])

    assert "error" in payload
    assert "Task 1" in payload["error"]
    assert "not-a-real-provider" in payload["error"]
    assert BAD_PROVIDER_ERROR in payload["error"]
    MockAgent.assert_not_called()  # no partial batch, no silent substitution


def test_bad_pin_at_the_end_still_builds_nothing():
    """Routes are preflighted before construction, so a late bad pin leaves no orphaned child agents."""
    payload, MockAgent, _ = _run([
        {"goal": f"{LONG_GOAL} first", "provider": "opencode-go", "model": "grok-4.6", "toolsets": ["todo"]},
        {"goal": f"{LONG_GOAL} second"},
        {"goal": f"{LONG_GOAL} third", "provider": "not-a-real-provider"},
    ])
    assert "error" in payload and "Task 2" in payload["error"]
    MockAgent.assert_not_called()


def test_config_level_pin_failure_keeps_its_bare_message():
    """An unpinned task must not gain a 'Task N' prefix — existing error text is unchanged."""
    parent = _parent()

    def always_fail(cfg, parent_agent):
        raise ValueError("Delegation provider 'x' resolved but has no API key.")

    with (
        patch("tools.delegate_tool._load_config", return_value={}),
        patch("tools.delegate_tool._resolve_delegation_credentials", side_effect=always_fail),
        patch("run_agent.AIAgent") as MockAgent,
    ):
        payload = json.loads(delegate_task(tasks=[{"goal": LONG_GOAL}], parent_agent=parent))

    assert payload["error"] == "Delegation provider 'x' resolved but has no API key."
    MockAgent.assert_not_called()


# -- pinned children never borrow the parent fallback chain -----------------------------------


@pytest.mark.parametrize("task", [
    {"goal": LONG_GOAL, "model": "qwen3.8-max"},
    {"goal": LONG_GOAL, "provider": "opencode-go", "model": "grok-4.6", "toolsets": ["todo"]},
])
def test_per_task_pin_counts_as_pinned_for_fallback(task):
    _, MockAgent, _ = _run([task])
    assert _child_kwargs(MockAgent, 0)["fallback_model"] is None


def test_reasoning_only_pin_does_not_make_a_child_pinned():
    """reasoning_effort is not a route pin: the chain stays inherited."""
    _, MockAgent, _ = _run([{"goal": LONG_GOAL, "reasoning_effort": "high"}], delegation_cfg={})
    assert _child_kwargs(MockAgent, 0)["fallback_model"] == PARENT_CHAIN


# -- schema -----------------------------------------------------------------------------------


def test_schema_advertises_the_three_per_task_keys():
    props = DELEGATE_TASK_SCHEMA["parameters"]["properties"]["tasks"]["items"]["properties"]
    for key in ("provider", "model", "reasoning_effort"):
        assert props[key]["type"] == "string", key
    assert props["goal"]["type"] == "string"
    assert DELEGATE_TASK_SCHEMA["parameters"]["properties"]["tasks"]["items"]["required"] == ["goal"]


# -- per-task toolsets ------------------------------------------------------------------------
#
# The security case: forge, the orchestrator, carries memory-mcp — Denis's Obsidian vault — and can now
# dispatch a child onto OpenCode Go, which is allowed unsensitive data only. Children inherit the
# parent's toolsets by default, so a route pin on its own would hand Go the vault. Two things close
# that: an explicit ``toolsets`` list is EXACT (no MCP re-add), and a cross-provider pin without one is
# refused outright.


def _run_toolsets(tasks, **kw):
    """Dispatch with the leaf role forced, so role-granted 'delegation' cannot blur the assertions."""
    with patch("tools.delegate_tool._get_orchestrator_enabled", return_value=False):
        return _run(tasks, **kw)


def test_absent_toolsets_inherits_everything_including_the_vault():
    """Characterization, and the baseline every other test here is measured against: inheritance is
    unchanged by this patch, and it really does pass the vault down."""
    assert _is_mcp_toolset_name(VAULT_ALIAS) is True, "vault alias cold — this suite would prove nothing"
    _, MockAgent, _ = _run_toolsets([{"goal": LONG_GOAL}])
    kw = _child_kwargs(MockAgent, 0)
    assert kw["enabled_toolsets"] == INHERITED_TOOLSETS
    assert VAULT_ALIAS in kw["enabled_toolsets"]
    # The builtin memory toolset is blocked for every child; the MCP server holding the same vault is not.
    assert "memory" not in kw["enabled_toolsets"] and "memory" in kw["disabled_toolsets"]


def test_mcp_reinheritance_is_what_exact_suppresses():
    """The mechanism at the resolver: without exact=True the vault is handed back to a narrowed child."""
    assert _is_mcp_toolset_name(VAULT_ALIAS) is True, "vault alias cold — this suite would prove nothing"
    parent = _parent()
    with patch("tools.delegate_tool_toolsets._get_inherit_mcp_toolsets", return_value=True):
        inherited, _ = _resolve_child_toolsets(parent, ["todo"], "leaf", exact=False)
        pinned, _ = _resolve_child_toolsets(parent, ["todo"], "leaf", exact=True)
    assert inherited == ["todo", SECOND_MCP_TOOLSET, VAULT_ALIAS], \
        "guard: the default re-adds EVERY parent MCP toolset, not just the one this test names"
    assert pinned == ["todo"]


def test_prefixed_mcp_toolset_is_recognised_without_the_alias_registry():
    """Covers the other half of _is_mcp_toolset_name: the ``mcp-`` prefix, with zero global state.

    Every other test here carries the BARE name, as forge does, which is only recognised through the
    registry. This one empties the alias map so a broken prefix check cannot hide behind a registered
    alias — the two branches are then independently proven.
    """
    from tools.registry import registry
    for alias in registry.get_registered_toolset_aliases():
        registry.unregister_toolset_alias(alias)
    assert registry.get_registered_toolset_aliases() == {}, "alias map not actually empty"
    assert _is_mcp_toolset_name(VAULT_ALIAS) is False
    assert _is_mcp_toolset_name(VAULT_TOOLSET) is True
    parent = _parent(enabled_toolsets=["todo", VAULT_TOOLSET])
    with patch("tools.delegate_tool_toolsets._get_inherit_mcp_toolsets", return_value=True):
        inherited, _ = _resolve_child_toolsets(parent, ["todo"], "leaf", exact=False)
        pinned, _ = _resolve_child_toolsets(parent, ["todo"], "leaf", exact=True)
    assert inherited == ["todo", VAULT_TOOLSET], "guard: the prefixed name really is re-added too"
    assert pinned == ["todo"]


def test_explicit_toolsets_drop_the_parents_mcp_vault_toolset():
    """The property Denis's hard limit rests on: a child routed to OpenCode Go cannot reach the vault."""
    assert _is_mcp_toolset_name(VAULT_ALIAS) is True, "vault alias cold — this suite would prove nothing"
    _, MockAgent, _ = _run_toolsets([
        {"goal": LONG_GOAL, "provider": "opencode-go", "model": "grok-4.6", "toolsets": ["todo"]},
    ])
    kw = _child_kwargs(MockAgent, 0)
    assert kw["provider"] == "opencode-go"
    assert kw["enabled_toolsets"] == ["todo"]
    # Both spellings: the registry carries the canonical name and the bare alias, and either would reach it.
    assert VAULT_ALIAS not in kw["enabled_toolsets"] and VAULT_TOOLSET not in kw["enabled_toolsets"]
    # And the parent's other MCP server goes too — suppression is not special-cased to the vault.
    assert SECOND_MCP_TOOLSET not in kw["enabled_toolsets"]


def test_explicit_toolsets_replace_the_parent_set():
    _, MockAgent, _ = _run_toolsets([{"goal": LONG_GOAL, "toolsets": ["session_search", "todo"]}])
    assert _child_kwargs(MockAgent, 0)["enabled_toolsets"] == ["session_search", "todo"]


def test_explicit_toolsets_drop_toolsets_the_child_would_have_inherited():
    """A narrowed child provably loses toolsets inheritance would have given it."""
    _, MockAgent, _ = _run_toolsets([{"goal": LONG_GOAL, "toolsets": ["todo"]}])
    kw = _child_kwargs(MockAgent, 0)
    for lost in ("session_search", "skills", VAULT_ALIAS, SECOND_MCP_TOOLSET):
        assert lost in INHERITED_TOOLSETS and lost not in kw["enabled_toolsets"], lost


def test_empty_toolsets_means_no_tools_not_inherit():
    """[] is a pure reasoning child. An empty-list-means-default bug would silently restore the vault."""
    _, MockAgent, _ = _run_toolsets([{"goal": LONG_GOAL, "toolsets": []}])
    assert _child_kwargs(MockAgent, 0)["enabled_toolsets"] == []
    assert INHERITED_TOOLSETS  # the distinction is real: inheriting would have given it tools


def test_mcp_toolset_names_are_accepted_as_known():
    """An MCP toolset is a legitimate pin; validation must not reject the alias as an unknown name."""
    _, MockAgent, _ = _run_toolsets([{"goal": LONG_GOAL, "toolsets": ["todo", VAULT_ALIAS]}])
    assert _child_kwargs(MockAgent, 0)["enabled_toolsets"] == ["todo", VAULT_ALIAS]


def test_toolsets_pin_is_per_task_not_batch_wide():
    """One narrowed child must not narrow its siblings, nor a sibling widen it."""
    _, MockAgent, _ = _run_toolsets([
        {"goal": f"{LONG_GOAL} restricted", "provider": "opencode-go", "model": "grok-4.6", "toolsets": ["todo"]},
        {"goal": f"{LONG_GOAL} trusted"},
    ])
    restricted, trusted = _child_kwargs(MockAgent, 0), _child_kwargs(MockAgent, 1)
    assert restricted["enabled_toolsets"] == ["todo"]
    assert trusted["enabled_toolsets"] == INHERITED_TOOLSETS
    assert trusted["provider"] == "alibaba"


# -- toolset pins fail loud -------------------------------------------------------------------


def test_unknown_toolset_aborts_the_batch_naming_the_task():
    payload, MockAgent, _ = _run_toolsets([
        {"goal": f"{LONG_GOAL} fine"},
        {"goal": f"{LONG_GOAL} typo", "toolsets": ["todo", "flie", "nonsense"]},
    ])
    assert "Task 1" in payload["error"]
    assert "flie" in payload["error"] and "nonsense" in payload["error"]
    assert "unknown toolset" in payload["error"]
    MockAgent.assert_not_called()  # nothing built, nothing run


def test_toolset_the_parent_lacks_aborts_with_its_own_diagnosis():
    """A real name the parent cannot delegate is a different mistake from a typo, and says so.

    _resolve_child_toolsets drops these silently — it is shared with other callers — so the check lives
    in delegate_task's own preflight, against the same expanded parent set the resolver intersects with.
    """
    assert "web" not in PARENT_TOOLSETS
    payload, MockAgent, _ = _run_toolsets([
        {"goal": f"{LONG_GOAL} fine"},
        {"goal": f"{LONG_GOAL} greedy", "toolsets": ["todo", "web"]},
    ])
    assert "Task 1" in payload["error"] and "web" in payload["error"]
    assert "does not have" in payload["error"]
    assert "unknown toolset" not in payload["error"]
    MockAgent.assert_not_called()


def test_non_list_toolsets_aborts_the_batch():
    payload, MockAgent, _ = _run_toolsets([{"goal": LONG_GOAL, "toolsets": "todo"}])
    assert "Task 0" in payload["error"] and "list of toolset names" in payload["error"]
    MockAgent.assert_not_called()


# -- a cross-provider child carries only what the operator granted that provider ----------------
#
# Denis's hard limit — nothing may give an OpenCode Go child access to his vault — cannot rest on the
# orchestrator choosing a good toolset list, because the orchestrator is the untrusted party here. The
# operator writes delegation.provider_toolsets; the model may only narrow within it. Three routes reach
# a cross-provider child and all three are gated on the FINAL resolved list.


def test_cross_provider_explicit_vault_pin_aborts():
    """Route 1: the model names the vault outright. A present 'toolsets' key is not consent."""
    assert _is_mcp_toolset_name(VAULT_ALIAS) is True, "vault alias cold — this suite would prove nothing"
    payload, MockAgent, _ = _run_toolsets([
        {"goal": f"{LONG_GOAL} fine"},
        {"goal": f"{LONG_GOAL} leak", "provider": "opencode-go", "model": "grok-4.6",
         "toolsets": ["todo", VAULT_ALIAS]},
    ])
    assert "Task 1" in payload["error"] and VAULT_ALIAS in payload["error"]
    assert "not granted" in payload["error"]
    MockAgent.assert_not_called()


def test_cross_provider_inheritance_aborts():
    """Route 2: no 'toolsets' key at all, so inheritance hands the child the parent's whole set."""
    assert _is_mcp_toolset_name(VAULT_ALIAS) is True, "vault alias cold — this suite would prove nothing"
    payload, MockAgent, _ = _run_toolsets([
        {"goal": LONG_GOAL, "provider": "opencode-go", "model": "grok-4.6"},
    ])
    assert "Task 0" in payload["error"] and VAULT_ALIAS in payload["error"]
    MockAgent.assert_not_called()


def test_batch_level_provider_aborts_with_no_per_task_key_at_all():
    """Route 3: delegation.provider sends every child across; no per-task key is involved, so a guard
    living only in the per-task preflight would never even run."""
    assert _is_mcp_toolset_name(VAULT_ALIAS) is True, "vault alias cold — this suite would prove nothing"
    payload, MockAgent, _ = _run_toolsets(
        [{"goal": LONG_GOAL}], delegation_cfg=_cfg(provider="opencode-go", model="grok-4.6"))
    assert "Task 0" in payload["error"] and VAULT_ALIAS in payload["error"]
    MockAgent.assert_not_called()


def test_session_search_is_gated_like_any_other_ungranted_toolset():
    """MCP-ness is not the property being gated. session_search reads the whole profile's history,
    which holds whatever the parent read this turn — and it is denied for the same reason."""
    payload, MockAgent, _ = _run_toolsets([
        {"goal": LONG_GOAL, "provider": "opencode-go", "model": "grok-4.6",
         "toolsets": ["todo", "session_search"]},
    ])
    assert "Task 0" in payload["error"] and "session_search" in payload["error"]
    MockAgent.assert_not_called()


def test_the_gate_vets_the_list_the_child_is_actually_built_with():
    """The invariant behind the grant gate, stated directly: vetted list == constructed list.

    Both ``_task_toolsets`` and ``_resolve_child_toolsets`` read the global MCP alias registry, which
    ``mcp_tool_health`` deregisters on its own thread when a server's last tool goes away. Resolving in
    the preflight and again at construction let a drop between the two turn a vetted narrow list back
    into full inheritance — the widest outcome available, reached by a check that ran and passed. The
    fix is structural (resolve once, carry forward), so this test is meant to be hard to break.
    """
    from tools.registry import registry
    import tools.delegate_tool as dt
    vetted = []
    real_gate = dt._cross_provider_toolset_error

    def gate_then_drop_the_alias(index, provider, child_toolsets):
        vetted.append(list(child_toolsets))
        err = real_gate(index, provider, child_toolsets)
        # Exactly the window the defect lived in: after the gate has passed, before the child is built.
        registry.unregister_toolset_alias(VAULT_ALIAS)
        return err

    grant = {"opencode-go": ["todo", VAULT_ALIAS], "alibaba": PROVIDER_TOOLSETS["alibaba"]}
    with patch.object(dt, "_cross_provider_toolset_error", side_effect=gate_then_drop_the_alias):
        _, MockAgent, _ = _run_toolsets(
            [{"goal": LONG_GOAL, "provider": "opencode-go", "model": "grok-4.6",
              "toolsets": ["todo", VAULT_ALIAS]}],
            delegation_cfg=_cfg(provider_toolsets=grant),
        )
    assert vetted == [["todo", VAULT_ALIAS]], "the gate did not see the list this test assumes"
    assert _child_kwargs(MockAgent, 0)["enabled_toolsets"] == vetted[0]


def test_a_role_flip_between_preflight_and_construction_changes_nothing():
    """The same defect class through the other global: the orchestrator kill switch. A flip to
    orchestrator re-adds 'delegation' to the child — after the gate vetted a list without it."""
    import tools.delegate_tool as dt
    # False for the preflight's single call, True for every _build_child_agent call after it. The depth
    # budget has to be wide enough for the flip to reach the role at all, or the test proves nothing.
    flip = [False] + [True] * 8
    with (
        patch.object(dt, "_get_orchestrator_enabled", side_effect=flip),
        patch.object(dt, "_get_max_spawn_depth", return_value=5),
    ):
        _, MockAgent, _ = _run([{"goal": LONG_GOAL}])
    kw = _child_kwargs(MockAgent, 0)
    assert kw["enabled_toolsets"] == INHERITED_TOOLSETS
    assert "delegation" not in kw["enabled_toolsets"]


def test_the_provider_grant_abort_is_its_own_third_diagnosis():
    """Three failures, three fixes: fix the name, widen the parent, or ask the operator to grant it.
    A caller cannot pick the right one if the messages blur, so the grant abort borrows neither wording."""
    payload, _, _ = _run_toolsets([
        {"goal": LONG_GOAL, "provider": "opencode-go", "model": "grok-4.6",
         "toolsets": ["todo", VAULT_ALIAS]},
    ])
    assert "not granted" in payload["error"]
    assert "delegation.provider_toolsets" in payload["error"]
    assert "unknown toolset" not in payload["error"]  # the name is real
    assert "does not have" not in payload["error"]  # and the parent does have it


def test_undeclared_provider_fails_closed():
    """A provider the operator never wrote down receives nothing, not everything."""
    payload, MockAgent, _ = _run_toolsets(
        [{"goal": LONG_GOAL, "provider": "opencode-go", "model": "grok-4.6", "toolsets": ["todo"]}],
        delegation_cfg=_cfg(provider_toolsets={"some-other-provider": ["todo"]}),
    )
    assert "Task 0" in payload["error"] and "opencode-go" in payload["error"]
    assert "no delegation.provider_toolsets entry" in payload["error"]
    assert "granted nothing" in payload["error"]
    MockAgent.assert_not_called()


def test_toolless_child_on_an_undeclared_provider_is_allowed():
    """The one case an undeclared provider still runs: it carries nothing, so it can leak nothing.
    Keeps the gate from aborting batches that were never at risk."""
    _, MockAgent, _ = _run_toolsets(
        [{"goal": LONG_GOAL, "provider": "opencode-go", "model": "grok-4.6", "toolsets": []}],
        delegation_cfg=_cfg(provider_toolsets={}),
    )
    kw = _child_kwargs(MockAgent, 0)
    assert (kw["provider"], kw["enabled_toolsets"]) == ("opencode-go", [])


def test_granted_toolsets_pass_the_gate():
    _, MockAgent, _ = _run_toolsets([
        {"goal": LONG_GOAL, "provider": "opencode-go", "model": "grok-4.6", "toolsets": ["todo"]},
    ])
    kw = _child_kwargs(MockAgent, 0)
    assert (kw["provider"], kw["enabled_toolsets"]) == ("opencode-go", ["todo"])


def test_empty_toolsets_passes_any_grant():
    """[] carries nothing, so it satisfies even an empty operator grant."""
    _, MockAgent, _ = _run_toolsets(
        [{"goal": LONG_GOAL, "provider": "opencode-go", "model": "grok-4.6", "toolsets": []}],
        delegation_cfg=_cfg(provider_toolsets={"opencode-go": []}),
    )
    kw = _child_kwargs(MockAgent, 0)
    assert (kw["provider"], kw["enabled_toolsets"]) == ("opencode-go", [])


def test_same_provider_child_is_not_gated():
    """'per-task keys absent -> today's behaviour exactly' survives: no boundary crossed, no gate,
    and no provider_toolsets entry needed."""
    _, MockAgent, _ = _run_toolsets([{"goal": LONG_GOAL}], delegation_cfg={})
    assert _child_kwargs(MockAgent, 0)["enabled_toolsets"] == INHERITED_TOOLSETS


def test_model_only_pin_is_gated_by_the_batch_provider_not_the_model():
    """The model pin changes no trust boundary; the batch provider it rides on is what is gated."""
    _, MockAgent, _ = _run_toolsets([{"goal": LONG_GOAL, "model": "qwen3.8-max"}])
    kw = _child_kwargs(MockAgent, 0)
    assert (kw["model"], kw["enabled_toolsets"]) == ("qwen3.8-max", INHERITED_TOOLSETS)


def test_provider_toolsets_is_a_registered_config_key():
    """An unregistered key is invisible to anyone reading the schema, and is exactly what a future
    strict-validation pass drops silently. Registered empty: every provider starts granted nothing."""
    from hermes_cli.config_defaults import DEFAULT_CONFIG
    assert DEFAULT_CONFIG["delegation"]["provider_toolsets"] == {}


def test_a_provider_absent_from_the_table_is_undeclared_not_empty():
    """Which of the two aborts fires depends on this distinction surviving: None = no entry at all (an
    omission), [] = an entry granting nothing (a deliberate denial)."""
    from tools.delegate_tool_config import _get_provider_toolsets
    # Patched on delegate_tool, not delegate_tool_config: _cfg() imports _load_config from the origin
    # module at call time, precisely so a patch there reaches every config reader.
    with patch("tools.delegate_tool._load_config", return_value={"provider_toolsets": {"alibaba": []}}):
        assert _get_provider_toolsets("opencode-go") is None
        assert _get_provider_toolsets("alibaba") == []


def test_schema_advertises_toolsets_as_a_string_array():
    props = DELEGATE_TASK_SCHEMA["parameters"]["properties"]["tasks"]["items"]["properties"]
    assert props["toolsets"]["type"] == "array"
    assert props["toolsets"]["items"] == {"type": "string"}
    # The requirement must be stated where an orchestrator prompt author will see it.
    assert "toolsets" in props["provider"]["description"]
    assert "delegation.provider_toolsets" in props["provider"]["description"]
