"""Per-call model/provider override for ``delegate_task`` (card t_55547259).

The provocation: routing ONE batch of subagents onto a different provider had
no per-call knob, so the only lever was editing ``delegation:`` in the live
root config.yaml — which changes every future subagent in every session, has
no expiry, and trips fleet-config-lint.

These tests pin the behaviours that make the config edit unnecessary:
  * the override object is accepted at batch level and per task
  * per-task > top-level call > ``delegation:`` config > parent
  * a flat string is REFUSED with a message naming the object shape
  * a flagship route chosen at the call site requires ``firepower``
  * the chosen route is announced as ``route=provider/model source=...``
"""

import pytest

from hermes_cli.model_override import (
    ModelOverrideError,
    format_route,
    parse_model_override,
)


# --------------------------------------------------------------------------
# Shape: the object is accepted, the flat string is not.
# --------------------------------------------------------------------------


def test_object_shape_accepted():
    ov = parse_model_override(
        {"model": "gpt-5.6-sol-900k", "provider": "openai-codex"}, field="model"
    )
    assert ov.model == "gpt-5.6-sol-900k"
    assert ov.provider == "openai-codex"


def test_any_subset_is_valid():
    """Ace 2026-09-21: all keys optional, any subset.

    ``{"provider": ...}`` alone legitimately means "same model, different
    provider" — the exact shape needed to move a batch off a capped pool.
    """
    ov = parse_model_override({"provider": "claude-bpx-19"}, field="model")
    assert ov.provider == "claude-bpx-19"
    assert ov.model is None

    ov = parse_model_override({"reasoning_effort": "high"}, field="model")
    assert ov.reasoning_effort == "high"


def test_flat_string_is_refused_loudly_naming_the_object_shape():
    """The load-bearing rejection.

    A silently-ignored flat string is what let an explicit route get dropped
    and the children run on the config default. The refusal must name the
    object shape so the caller can fix the call in one edit.
    """
    with pytest.raises(ModelOverrideError) as exc:
        parse_model_override("gpt-5.6-sol-900k", field="model")

    msg = str(exc.value)
    assert "must be an object" in msg
    # Names the shape, not just "invalid".
    assert '"model"' in msg and '"provider"' in msg
    # Echoes what the caller actually passed so the fix is obvious.
    assert "gpt-5.6-sol-900k" in msg


def test_unknown_key_is_refused_rather_than_silently_dropped():
    """A misremembered key ("effort") must not silently pin the wrong route."""
    with pytest.raises(ModelOverrideError) as exc:
        parse_model_override({"model": "x", "effort": "high"}, field="model")
    assert "effort" in str(exc.value)
    assert "reasoning_effort" in str(exc.value)


def test_invalid_reasoning_effort_is_refused():
    with pytest.raises(ModelOverrideError) as exc:
        parse_model_override({"reasoning_effort": "turbo"}, field="model")
    assert "turbo" in str(exc.value)


def test_reasoning_effort_none_means_disabled_not_invalid():
    ov = parse_model_override({"reasoning_effort": "none"}, field="model")
    assert ov.reasoning_effort == "none"


# --------------------------------------------------------------------------
# Guard: flagship routes chosen at the call site need a stated reason.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("flagship", ["gpt-6-astra", "claude-fable-5"])
def test_flagship_without_firepower_is_refused(flagship):
    with pytest.raises(ModelOverrideError) as exc:
        parse_model_override({"model": flagship}, field="model")
    msg = str(exc.value)
    assert "firepower" in msg
    # Quotes the rule rather than just erroring.
    assert "reserved" in msg or "deliberate" in msg.lower()


@pytest.mark.parametrize("flagship", ["gpt-6-astra", "claude-fable-5"])
def test_flagship_with_firepower_is_allowed(flagship):
    ov = parse_model_override(
        {"model": flagship, "firepower": "needs deepest reasoning"}, field="model"
    )
    assert ov.model == flagship
    assert ov.firepower == "needs deepest reasoning"


def test_non_flagship_needs_no_firepower():
    ov = parse_model_override({"model": "gpt-5.6-sol-900k"}, field="model")
    assert ov.firepower is None


def test_firepower_guard_uses_the_shared_predicate_not_a_local_copy():
    """The guard must delegate to model_policy so one classifier serves all
    surfaces. If this module grew its own matcher, banning a model on Kanban
    would leave it reachable via delegate_task."""
    import hermes_cli.model_override as mo
    import hermes_cli.model_policy as mp

    assert mo.firepower_guard_error is mp.firepower_guard_error
    assert mo.is_firepower_model is mp.is_firepower_model


# --------------------------------------------------------------------------
# Precedence: per-task > call > config > parent.
# --------------------------------------------------------------------------


class _Parent:
    model = "parent-model"
    provider = "parent-provider"


_CFG = {"model": "cfg-model", "provider": "cfg-provider", "base_url": None,
        "api_key": None, "api_mode": None}
_NO_CFG = {"model": None, "provider": None, "base_url": None,
           "api_key": None, "api_mode": None}


def _route(monkeypatch, task, batch, config_creds):
    """Drive the REAL _build_children route resolver.

    Upstream resolves the per-child route inside _build_children (closure
    ``_child_overrides``), so we exercise it through _build_children with child
    construction stubbed — the routing decision under test stays real.
    """
    import tools.delegate_tool as dt

    monkeypatch.setattr(dt, "_resolve_delegation_credentials",
                        lambda cfg, parent: {"model": cfg.get("model"),
                                             "provider": cfg.get("provider"),
                                             "base_url": None, "api_key": None,
                                             "api_mode": None})
    captured = {}

    def _fake_build(**kw):
        captured["model"] = kw["model"]
        captured["provider"] = kw["override_provider"]
        raise _Stop

    class _Stop(Exception):
        pass

    monkeypatch.setattr(dt, "_build_child_preserving_parent_tools", _fake_build)
    monkeypatch.setattr(dt, "wrap_progress_callback", lambda *a, **k: None,
                        raising=False)
    try:
        dt._build_children([dict(task)], [None], dict(config_creds),
                           top_role="leaf", max_iterations=5,
                           parent_agent=_Parent(), routing_cfg={},
                           live_deleg_id=None, live_writers=[],
                           batch_override=batch)
    except _Stop:
        pass
    except Exception:
        pass
    return captured


def test_per_task_beats_top_level_call(monkeypatch):
    batch = parse_model_override({"model": "batch-model", "provider": "batch-prov"})
    got = _route(monkeypatch, {"goal": "g", "model": {"model": "task-model"}},
                 batch, _CFG)
    assert got.get("model") == "task-model"


def test_per_task_partial_override_keeps_batch_fields(monkeypatch):
    batch = parse_model_override({"model": "batch-model", "provider": "batch-prov"})
    got = _route(monkeypatch, {"goal": "g", "model": {"model": "task-model"}},
                 batch, _CFG)
    assert got.get("provider") == "batch-prov"


def test_top_level_call_beats_config(monkeypatch):
    batch = parse_model_override({"model": "batch-model", "provider": "batch-prov"})
    got = _route(monkeypatch, {"goal": "g"}, batch, _CFG)
    assert got.get("model") == "batch-model"
    assert got.get("provider") == "batch-prov"


def test_config_used_when_call_supplies_nothing(monkeypatch):
    got = _route(monkeypatch, {"goal": "g"}, None, _CFG)
    assert got.get("model") == "cfg-model"


def test_per_task_flagship_without_firepower_is_refused(monkeypatch):
    with pytest.raises(ModelOverrideError):
        parse_model_override({"model": "claude-fable-5"}, field="tasks[].model")


def test_batch_firepower_covers_its_tasks(monkeypatch):
    batch = parse_model_override(
        {"model": "claude-fable-5", "firepower": "deep architectural review"})
    got = _route(monkeypatch, {"goal": "g", "model": {"provider": "anthropic"}},
                 batch, _CFG)
    assert got.get("model") == "claude-fable-5"


# --------------------------------------------------------------------------
# ANNOUNCE: the route and its source are stated.
# --------------------------------------------------------------------------


def test_route_line_format():
    assert (
        format_route("openai-codex", "gpt-5.6-sol-900k", "call")
        == "route=openai-codex/gpt-5.6-sol-900k source=call"
    )


@pytest.mark.parametrize("source", ["call", "config", "parent"])
def test_route_line_names_every_source(source):
    line = format_route("p", "m", source)
    assert line == f"route=p/m source={source}"


def test_route_line_flags_firepower_routes():
    assert "firepower=yes" in format_route("anthropic", "claude-fable-5", "call")
    assert "firepower=yes" not in format_route("openai-codex", "gpt-5.6-sol", "call")


# --------------------------------------------------------------------------
# Wiring: the arg must actually reach delegate_task on the LIVE path.
# --------------------------------------------------------------------------


def test_live_dispatch_path_forwards_model():
    """run_agent._dispatch_delegate_task is the LIVE path (the registry lambda
    is only the bypass fallback). If it drops `model`, the feature is inert in
    production while unit tests against the lambda still pass."""
    import inspect

    import run_agent

    src = inspect.getsource(run_agent.AIAgent._dispatch_delegate_task)
    assert 'model=function_args.get("model")' in src
