#!/usr/bin/env python3
"""Quota-wall descent for ``delegate_task``: walk a declared ladder, one rung per wall, and nothing else.

A child that dies on a quota wall re-runs on the next model down ``delegation.descent_order``, repeating
per wall until the order runs out. Contract under test:

* quota wall (billing / rate_limit / upstream_rate_limit) -> descend ONE rung, on a child built from the
  same task with only its credentials changed,
* any other ``failure_reason``, or none at all -> returned as-is, never descended: a task failure masked
  as a transient one is the expensive direction of wrong,
* ladder exhausted, model absent from the order, or opted out -> fail loud, and SAY WHICH (``descent_halted``),
* no wrap-around, no revisiting: rungs are consumed left to right,
* the walk is legible after the fact -> ``switched_from`` / ``switched_to`` / ``switched_reason`` and an
  ordered ``route_history`` carrying every hop's route, outcome and cost,
* per-task override -> ``fallback: "<model>"`` is one rung instead of the ladder, ``fallback: "none"`` opts out,
* malformed ``fallback`` or ``descent_order`` -> the WHOLE batch aborts at validation, before any child,
* a wall on one task leaves its siblings' results untouched.

The harness (parent double, fake credential resolver, config block) is shared with
``test_delegate_per_task_dispatch`` so the two cannot drift about what a route even is.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from tools.delegate_tool import (
    DELEGATE_TASK_SCHEMA, _normalize_descent_order, _task_fallback_mode, delegate_task,
)
from tools.delegate_tool_dispatch import _QUOTA_WALL_REASONS

from tests.tools.test_delegate_per_task_dispatch import (
    DELEGATION_CFG, LONG_GOAL, _fake_resolve, _parent,
)

# Denis's descent order for the opencode-go lane, revised 2026-09-14. Deliberately a fixture, not a read
# of the live profile: a unit test that reads production config breaks in CI and on anyone else's machine.
# What the tests need from it is structural — 7 rungs, so mid-ladder starts and exhaustion stay distinct.
DESCENT_ORDER = ["grok-4.6", "kimi-k2.7-code", "kimi-k3", "gpt-5.6-luna",
                 "minimax-m2.7", "glm-5.3", "glm-5.3-flash"]
GO = "opencode-go"
# Every route granted every toolset the parent can delegate, so the cross-provider gate is satisfied and
# these tests measure the router rather than re-testing the grant table. One test below un-grants on purpose.
OPEN_GRANTS = {"alibaba": list(_parent().enabled_toolsets), GO: list(_parent().enabled_toolsets)}
GO_CFG = {**DELEGATION_CFG, "provider": GO, "model": "grok-4.6", "base_url": "",
          "provider_toolsets": OPEN_GRANTS, "descent_order": list(DESCENT_ORDER)}


def _go_cfg(**overrides):
    return {**GO_CFG, **overrides}


def _rung(model):
    """The ``(provider, model)`` key a ladder child is constructed under."""
    return (GO, model)


def _ok(text="done"):
    return {"final_response": text, "completed": True, "api_calls": 1}


def _walled(reason="rate_limit"):
    """A child that died on a provider wall — the shape ``_build_result_entry`` turns into
    ``status=failed`` + ``failure_reason``."""
    return {"final_response": "", "completed": False, "failed": True,
            "error": f"429 {reason}", "failure_reason": reason, "api_calls": 1}


def _failed(reason=None, error="the tool call was malformed"):
    out = {"final_response": "", "completed": False, "failed": True, "error": error, "api_calls": 1}
    if reason:
        out["failure_reason"] = reason
    return out


def _run(tasks, results_by_route, *, delegation_cfg=None, parent=None):
    """Dispatch a batch, handing each child the result queued for the ROUTE it was built on.

    Returns ``(payload, built_routes)`` — every ``(provider, model)`` a child was actually constructed
    for, in construction order, which is how "how far did it descend?" is answered without counting mocks.

    Keying on the route rather than on call order is deliberate: ladder children are constructed LATE,
    from the worker thread that hit the wall, so construction order is a batch-scheduling detail.
    """
    parent = parent or _parent()
    built_routes = []

    def _make(**kw):
        route = (kw.get("provider"), kw.get("model"))
        built_routes.append(route)
        child = MagicMock()
        child.run_conversation.return_value = dict(results_by_route[route])
        return child

    cfg = _go_cfg() if delegation_cfg is None else delegation_cfg
    with (
        patch("tools.delegate_tool._load_config", return_value=dict(cfg)),
        patch("tools.delegate_tool._resolve_delegation_credentials", side_effect=_fake_resolve),
        patch("run_agent.AIAgent") as MockAgent,
    ):
        MockAgent.side_effect = _make
        raw = delegate_task(tasks=tasks, parent_agent=parent)
    return json.loads(raw), built_routes


def _entries(payload):
    """Result entries keyed by task_index, whatever envelope the batch came back in."""
    for key in ("results", "tasks", "children"):
        if isinstance(payload, dict) and isinstance(payload.get(key), list):
            return {e["task_index"]: e for e in payload[key] if isinstance(e, dict) and "task_index" in e}
    if isinstance(payload, list):
        return {e["task_index"]: e for e in payload if isinstance(e, dict) and "task_index" in e}
    raise AssertionError(f"no result entries in payload: {json.dumps(payload)[:400]}")


def _walled_everywhere(*models, reason="rate_limit"):
    return {_rung(m): _walled(reason) for m in models}


# ── one rung per wall ────────────────────────────────────────────────────────


@pytest.mark.parametrize("reason", ["billing", "rate_limit", "upstream_rate_limit"])
def test_a_quota_wall_descends_exactly_one_rung(reason):
    """All three rate-limit-family reasons are walls: the route is exhausted, the task is not at fault."""
    payload, built = _run(
        [{"goal": LONG_GOAL}],
        {_rung("grok-4.6"): _walled(reason), _rung("kimi-k2.7-code"): _ok("recovered")},
    )
    assert built == [_rung("grok-4.6"), _rung("kimi-k2.7-code")]
    entry = _entries(payload)[0]
    assert entry["status"] == "completed" and entry["summary"] == "recovered"


def test_upstream_rate_limit_is_treated_as_a_wall():
    """An aggregator's upstream 429 leaves the key healthy — the classifier's own answer is to switch
    models, which is exactly what a descent does. Guard against it being dropped from the set."""
    assert _QUOTA_WALL_REASONS == {"billing", "rate_limit", "upstream_rate_limit"}


def test_each_wall_moves_exactly_one_rung_not_to_the_bottom():
    """Three walls, three rungs — not a jump to the end of the order, and not a repeat of a walled rung."""
    payload, built = _run(
        [{"goal": LONG_GOAL}],
        {**_walled_everywhere("grok-4.6", "kimi-k2.7-code", "kimi-k3"),
         _rung("gpt-5.6-luna"): _ok("fourth time lucky")},
    )
    assert built == [_rung(m) for m in DESCENT_ORDER[:4]]
    assert _entries(payload)[0]["summary"] == "fourth time lucky"


def test_a_walk_starting_mid_ladder_descends_from_there():
    """A task pinned to rung 3 has rungs 4-7 below it, and rungs 1-2 are NOT reachable: no wrap-around,
    and no climbing back to a model the operator ranked above this one."""
    payload, built = _run(
        [{"goal": LONG_GOAL, "provider": GO, "model": "kimi-k3"}],
        {**_walled_everywhere("kimi-k3", "gpt-5.6-luna"), _rung("minimax-m2.7"): _ok()},
    )
    assert built == [_rung("kimi-k3"), _rung("gpt-5.6-luna"), _rung("minimax-m2.7")]
    entry = _entries(payload)[0]
    assert entry["switched_from"] == f"{GO}/kimi-k3"
    assert entry["switched_to"] == f"{GO}/minimax-m2.7"


def test_a_successful_child_never_builds_a_rung():
    payload, built = _run([{"goal": LONG_GOAL}], {_rung("grok-4.6"): _ok()})
    assert built == [_rung("grok-4.6")]
    assert _entries(payload)[0]["status"] == "completed"


# ── the walk is bounded, and ends loud ───────────────────────────────────────


def test_the_full_ladder_is_walked_then_fails_loud():
    """Seven rungs, seven children, no eighth: the order is the bound, consumed left to right."""
    payload, built = _run([{"goal": LONG_GOAL}], _walled_everywhere(*DESCENT_ORDER, reason="billing"))
    assert built == [_rung(m) for m in DESCENT_ORDER]
    entry = _entries(payload)[0]
    assert entry["status"] == "failed"
    assert entry["failure_reason"] == "billing"
    assert entry["descent_halted"] == "delegation.descent_order is exhausted below this rung"
    assert len(entry["route_history"]) == 7


def test_the_bottom_rung_has_nowhere_to_descend_and_says_so():
    payload, built = _run(
        [{"goal": LONG_GOAL, "provider": GO, "model": "glm-5.3-flash"}],
        {_rung("glm-5.3-flash"): _walled()},
    )
    assert built == [_rung("glm-5.3-flash")]
    entry = _entries(payload)[0]
    assert entry["status"] == "failed"
    assert "last rung" in entry["descent_halted"]
    assert "switched_to" not in entry


def test_a_model_absent_from_the_order_has_no_position_and_fails_loud():
    """Rule: do not guess a position. An unlisted model is not silently dropped at rung 1."""
    payload, built = _run(
        [{"goal": LONG_GOAL, "provider": GO, "model": "mimo-v2.5-pro"}],
        {_rung("mimo-v2.5-pro"): _walled("billing")},
    )
    assert built == [_rung("mimo-v2.5-pro")]
    entry = _entries(payload)[0]
    assert entry["status"] == "failed"
    assert "not in delegation.descent_order" in entry["descent_halted"]
    assert "mimo-v2.5-pro" in entry["descent_halted"]


def test_no_descent_order_configured_means_no_descent():
    """Today's behaviour, byte for byte, whenever the operator has declared nothing."""
    payload, built = _run(
        [{"goal": LONG_GOAL}], {_rung("grok-4.6"): _walled("billing")},
        delegation_cfg=_go_cfg(descent_order=[]),
    )
    assert built == [_rung("grok-4.6")]
    entry = _entries(payload)[0]
    assert entry["failure_reason"] == "billing"
    assert entry["descent_halted"] == "delegation.descent_order is not configured"
    assert not {"switched_from", "switched_to", "route_history"} & set(entry)


def test_a_halt_reason_is_only_stamped_on_an_actual_wall():
    """A task that succeeded, or failed for its own reasons, is not annotated about a ladder it never
    needed — that noise would make descent_halted meaningless where it matters."""
    payload, _ = _run(
        [{"goal": LONG_GOAL, "provider": GO, "model": "glm-5.3-flash"}],
        {_rung("glm-5.3-flash"): _ok()},
    )
    assert "descent_halted" not in _entries(payload)[0]


# ── every failure that is NOT a wall ─────────────────────────────────────────


@pytest.mark.parametrize("reason", ["format_error", "model_not_found", "auth_permanent",
                                    "context_overflow", "timeout", "unknown"])
def test_a_non_quota_failure_never_descends(reason):
    """These all set should_fallback=True or look transient, and none is a quota wall. Descending on them
    would report a real task failure as a route problem."""
    payload, built = _run([{"goal": LONG_GOAL}], {_rung("grok-4.6"): _failed(reason=reason)})
    assert built == [_rung("grok-4.6")]
    entry = _entries(payload)[0]
    assert entry["status"] == "failed" and entry["failure_reason"] == reason
    assert not {"switched_from", "switched_to", "descent_halted"} & set(entry)


def test_a_failure_with_no_classified_reason_never_descends():
    payload, built = _run([{"goal": LONG_GOAL}], {_rung("grok-4.6"): _failed()})
    assert built == [_rung("grok-4.6")]
    assert "switched_to" not in _entries(payload)[0]


def test_a_mid_walk_non_quota_failure_stops_the_walk_where_it_stands():
    """The rung that failed for its OWN reasons is final, even with five rungs still below it."""
    payload, built = _run(
        [{"goal": LONG_GOAL}],
        {_rung("grok-4.6"): _walled(), _rung("kimi-k2.7-code"): _failed(reason="format_error")},
    )
    assert built == [_rung("grok-4.6"), _rung("kimi-k2.7-code")]
    entry = _entries(payload)[0]
    assert entry["failure_reason"] == "format_error"
    assert entry["switched_to"] == f"{GO}/kimi-k2.7-code"
    assert "descent_halted" not in entry  # it stopped by choice, not by running out


# ── legibility of a completed walk ───────────────────────────────────────────


def test_route_history_records_every_hop_in_order():
    payload, _ = _run(
        [{"goal": LONG_GOAL}],
        {_rung("grok-4.6"): _walled("billing"), _rung("kimi-k2.7-code"): _walled("rate_limit"),
         _rung("kimi-k3"): _ok("third time lucky")},
    )
    entry = _entries(payload)[0]
    assert [h["route"] for h in entry["route_history"]] == [
        f"{GO}/grok-4.6", f"{GO}/kimi-k2.7-code", f"{GO}/kimi-k3"]
    assert [h.get("failure_reason") for h in entry["route_history"]] == ["billing", "rate_limit", None]
    assert [h["status"] for h in entry["route_history"]] == ["failed", "failed", "completed"]


def test_the_switch_stamps_name_the_ends_of_the_walk_not_the_last_hop():
    """A three-rung walk that reported only its final hop would be indistinguishable from a one-rung one."""
    payload, _ = _run(
        [{"goal": LONG_GOAL}],
        {**_walled_everywhere("grok-4.6", "kimi-k2.7-code"), _rung("kimi-k3"): _ok()},
    )
    entry = _entries(payload)[0]
    assert entry["switched_from"] == f"{GO}/grok-4.6"
    assert entry["switched_to"] == f"{GO}/kimi-k3"
    # The reason that STARTED the descent, not the one that ended it.
    assert entry["switched_reason"] == "rate_limit"


def test_the_switch_is_stamped_even_when_the_last_rung_also_fails():
    """"Never silent" is not conditional on the descent working."""
    payload, _ = _run(
        [{"goal": LONG_GOAL}],
        {_rung("grok-4.6"): _walled(), _rung("kimi-k2.7-code"): _failed(error="model refused the schema")},
    )
    entry = _entries(payload)[0]
    assert entry["status"] == "failed"
    assert (entry["switched_from"], entry["switched_to"]) == (f"{GO}/grok-4.6", f"{GO}/kimi-k2.7-code")


def test_every_abandoned_attempt_s_cost_is_folded_in():
    """A seven-rung walk costs seven children; reporting only the survivor's spend hides six of them."""
    payload, _ = _run([{"goal": LONG_GOAL}], _walled_everywhere(*DESCENT_ORDER))
    entry = _entries(payload)[0]
    # Children are MagicMocks whose cost coerces to 0.0 via _build_result_entry's isinstance guard, so what
    # survives as an assertion is that the fold ran over every hop and produced a number.
    assert isinstance(entry["cost_usd"], (int, float))
    assert all("cost_usd" in h and "duration_seconds" in h for h in entry["route_history"])


# ── per-task override ────────────────────────────────────────────────────────


def test_an_explicit_fallback_replaces_the_ladder_with_one_rung():
    payload, built = _run(
        [{"goal": LONG_GOAL, "fallback": "glm-5.3"}],
        {_rung("grok-4.6"): _walled(), _rung("glm-5.3"): _ok("explicit route")},
    )
    # kimi-k2.7-code is rung 2 and was NOT taken: the task's own route wins over the declared order.
    assert built == [_rung("grok-4.6"), _rung("glm-5.3")]
    assert _entries(payload)[0]["summary"] == "explicit route"


def test_an_explicit_fallback_does_not_then_continue_down_the_ladder():
    """It is an override, not an entry point: one rung, then loud."""
    payload, built = _run(
        [{"goal": LONG_GOAL, "fallback": "glm-5.3"}],
        {_rung("grok-4.6"): _walled(), _rung("glm-5.3"): _walled("billing")},
    )
    assert built == [_rung("grok-4.6"), _rung("glm-5.3")]
    entry = _entries(payload)[0]
    assert entry["status"] == "failed"
    assert entry["descent_halted"] == "delegation.descent_order is exhausted below this rung"


def test_an_explicit_fallback_may_cross_to_another_provider():
    """The credential bundle is resolved per rung, so a ladder can end somewhere else entirely."""
    payload, built = _run(
        [{"goal": LONG_GOAL, "fallback": {"model": "qwen3.8-max", "provider": "alibaba"}}],
        {_rung("grok-4.6"): _walled(), ("alibaba", "qwen3.8-max"): _ok("metered lane")},
    )
    assert built == [_rung("grok-4.6"), ("alibaba", "qwen3.8-max")]
    entry = _entries(payload)[0]
    assert entry["switched_to"] == "alibaba/qwen3.8-max"
    assert entry["summary"] == "metered lane"


@pytest.mark.parametrize("opt_out", ["none", "NONE", "  none  "])
def test_fallback_none_opts_the_task_out_of_descent_entirely(opt_out):
    """For work that must not silently move provider. The wall is final, and says why."""
    payload, built = _run(
        [{"goal": LONG_GOAL, "fallback": opt_out}], {_rung("grok-4.6"): _walled("billing")})
    assert built == [_rung("grok-4.6")]
    entry = _entries(payload)[0]
    assert entry["status"] == "failed" and entry["failure_reason"] == "billing"
    assert entry["descent_halted"] == "the task set fallback: 'none'"
    assert not {"switched_from", "switched_to", "route_history"} & set(entry)


def test_opting_one_task_out_does_not_opt_out_its_siblings():
    payload, built = _run([
        {"goal": f"{LONG_GOAL} sensitive", "fallback": "none"},
        {"goal": f"{LONG_GOAL} ordinary"},
    ], {_rung("grok-4.6"): _walled(), _rung("kimi-k2.7-code"): _ok("descended")})

    assert built.count(_rung("kimi-k2.7-code")) == 1
    entries = _entries(payload)
    assert entries[0]["status"] == "failed" and "descent_halted" in entries[0]
    assert entries[1]["summary"] == "descended"


# ── malformed input: abort at validation, before anything runs ───────────────


@pytest.mark.parametrize("bad, fragment", [
    (["glm-5.3"], "must be a model name or an object"),
    (42, "must be a model name or an object"),
    ("   ", "is blank"),
    ({}, "missing a 'model'"),
    ({"provider": "alibaba"}, "missing a 'model'"),
    ({"model": "   "}, "missing a 'model'"),
    ({"model": "glm-5.3", "toolsets": ["web"]}, "unsupported key"),
    ({"model": "glm-5.3", "reasoning_effort": "high"}, "unsupported key"),
])
def test_a_malformed_fallback_aborts_the_batch_before_any_child_is_built(bad, fragment):
    payload, built = _run([{"goal": LONG_GOAL, "fallback": bad}], {})
    assert built == [], "a validation abort must construct no children"
    blob = json.dumps(payload)
    assert fragment in blob and "Task 0" in blob


def test_a_malformed_fallback_names_the_opt_out_spelling():
    """The error has to teach the one value that is not a route, or 'none' is undiscoverable."""
    payload, _ = _run([{"goal": LONG_GOAL, "fallback": 42}], {})
    assert "'none'" in json.dumps(payload)


@pytest.mark.parametrize("bad, fragment", [
    ("grok-4.6", "must be a list of models"),
    ({"opencode-go": ["grok-4.6"]}, "must be a list of models"),
    ([{"provider": "opencode-go"}], "missing a 'model'"),
    ([{"model": "grok-4.6", "effort": "high"}], "unsupported key"),
    ([""], "is blank"),
    (["grok-4.6", "grok-4.6"], "repeats"),
    (["grok-4.6", {"model": "grok-4.6"}], "repeats"),
])
def test_a_malformed_descent_order_aborts_the_batch(bad, fragment):
    """Operator config, and the ONLY place the order exists — degrading to no ladder would be
    indistinguishable from every rung being walled."""
    payload, built = _run([{"goal": LONG_GOAL}], {}, delegation_cfg=_go_cfg(descent_order=bad))
    assert built == []
    assert fragment in json.dumps(payload)


def test_a_fallback_naming_the_task_s_own_route_aborts():
    payload, built = _run(
        [{"goal": LONG_GOAL, "provider": GO, "model": "glm-5.3", "fallback": "glm-5.3"}], {})
    assert built == []
    assert "same route" in json.dumps(payload)


def test_an_unresolvable_rung_aborts_the_batch_at_validation():
    """Found now, not three rungs into a walk at 2am."""
    payload, built = _run(
        [{"goal": LONG_GOAL, "fallback": {"model": "m", "provider": "not-a-real-provider"}}], {})
    assert built == []
    assert "not-a-real-provider" in json.dumps(payload)


def test_a_rung_crossing_to_an_ungranted_provider_aborts():
    """The cross-provider toolset grant is not a hole the ladder can be routed around."""
    payload, built = _run(
        [{"goal": LONG_GOAL, "toolsets": ["todo"],
          "fallback": {"model": "qwen3.8-max", "provider": "alibaba"}}], {},
        delegation_cfg=_go_cfg(provider_toolsets={GO: ["todo"]}),  # alibaba: undeclared
    )
    assert built == []
    blob = json.dumps(payload)
    assert "descent rung" in blob and "alibaba" in blob


def test_one_bad_fallback_aborts_the_whole_batch_including_good_siblings():
    payload, built = _run([
        {"goal": f"{LONG_GOAL} one"},
        {"goal": f"{LONG_GOAL} two", "fallback": {"provider": "alibaba"}},
    ], {})
    assert built == []
    assert "Task 1" in json.dumps(payload)


# ── siblings ─────────────────────────────────────────────────────────────────


def test_a_walk_on_one_task_leaves_its_siblings_intact():
    """The descent runs on the walled task's own worker; nothing about it reaches the batch."""
    payload, built = _run([
        {"goal": f"{LONG_GOAL} walled"},
        {"goal": f"{LONG_GOAL} sibling", "provider": GO, "model": "glm-5.3"},
    ], {_rung("grok-4.6"): _walled("billing"), _rung("kimi-k2.7-code"): _ok("recovered"),
        _rung("glm-5.3"): _ok("sibling fine")})

    assert sorted(built) == sorted([_rung("grok-4.6"), _rung("glm-5.3"), _rung("kimi-k2.7-code")])
    entries = _entries(payload)
    assert entries[0]["summary"] == "recovered"
    assert entries[0]["switched_to"] == f"{GO}/kimi-k2.7-code"
    assert entries[1]["status"] == "completed" and entries[1]["summary"] == "sibling fine"
    assert not {"switched_from", "switched_to", "route_history"} & set(entries[1])


def test_two_siblings_may_walk_the_same_ladder_independently():
    """A shared order is not shared state: each task consumes its own rungs from its own position."""
    payload, built = _run([
        {"goal": f"{LONG_GOAL} one"},
        {"goal": f"{LONG_GOAL} two", "provider": GO, "model": "minimax-m2.7"},
    ], {_rung("grok-4.6"): _walled(), _rung("kimi-k2.7-code"): _ok("one recovered"),
        _rung("minimax-m2.7"): _walled(), _rung("glm-5.3"): _ok("two recovered")})

    assert sorted(built) == sorted([_rung("grok-4.6"), _rung("minimax-m2.7"),
                                    _rung("kimi-k2.7-code"), _rung("glm-5.3")])
    entries = _entries(payload)
    assert entries[0]["switched_to"] == f"{GO}/kimi-k2.7-code"
    assert entries[1]["switched_to"] == f"{GO}/glm-5.3"


# ── validation units, config registration, model-facing surface ──────────────


def test_absent_fallback_means_ladder():
    assert _task_fallback_mode({"goal": LONG_GOAL}, 0) == ("ladder", None, None)


@pytest.mark.parametrize("spelling", ["none", "None", "NONE", " none "])
def test_the_opt_out_is_case_and_space_insensitive(spelling):
    assert _task_fallback_mode({"fallback": spelling}, 0)[0] == "none"


def test_a_bare_model_name_is_a_route_on_the_task_s_own_provider():
    mode, pins, err = _task_fallback_mode({"fallback": "qwen3.8-max"}, 0)
    assert (mode, err) == ("explicit", None)
    assert pins == {"provider": None, "model": "qwen3.8-max"}


def test_descent_order_normalizes_both_spellings_to_one_shape():
    order, err = _normalize_descent_order(
        {"descent_order": ["grok-4.6", {"model": "qwen3.8-max", "provider": "alibaba"}]})
    assert err is None
    assert order == [{"provider": None, "model": "grok-4.6"},
                     {"provider": "alibaba", "model": "qwen3.8-max"}]


def test_an_absent_descent_order_is_not_an_error():
    assert _normalize_descent_order({}) == ([], None)
    assert _normalize_descent_order({"descent_order": []}) == ([], None)


def test_descent_order_is_a_registered_config_key():
    """An unregistered key is invisible to anyone reading the schema, and is exactly what a future
    strict-validation pass drops silently. Registered empty: no descent until an operator declares one."""
    from hermes_cli.config_defaults import DEFAULT_CONFIG
    assert DEFAULT_CONFIG["delegation"]["descent_order"] == []


def test_schema_advertises_fallback_as_a_plain_string():
    """No union type: this repo's schemas never use one, and a strict provider may reject it."""
    props = DELEGATE_TASK_SCHEMA["parameters"]["properties"]["tasks"]["items"]["properties"]
    assert props["fallback"]["type"] == "string"
    desc = props["fallback"]["description"]
    # The four facts an orchestrator prompt author must not have to read the source for.
    assert "quota" in desc
    assert "descent_order" in desc
    assert "'none'" in desc
    assert "never retried" in desc


def test_a_fallback_that_only_LOOKS_different_from_the_task_route_aborts():
    """Regression: the guard compares RESOLVED routes. Here the task pins nothing (inheriting the batch
    route) and the fallback spells that same route out in full — as pins the two share not one value, and
    a pin-level compare waves it through onto the wall it was supposed to avoid."""
    payload, built = _run(
        [{"goal": LONG_GOAL, "fallback": {"provider": GO, "model": "grok-4.6"}}], {})
    assert built == []
    assert "same route the task already runs on" in json.dumps(payload)
