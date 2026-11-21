"""A fallback must route vision through the model that actually consumes the tool result.

Regression for NousResearch/hermes-agent#156017. The route-swap fix is carried
by #155848; Xiaomi's independent tool-message capability fix is #121833.
"""

from types import SimpleNamespace

from agent.auxiliary_client import (
    _read_main_model,
    _read_main_provider,
    clear_runtime_main,
    set_runtime_main,
)
from agent.route_binding import RouteEntryClient, bind_route_entry
from hermes_cli.config import load_config
from tools.vision_tools import _native_tool_result_images, _should_use_native_vision_fast_path


def test_route_fallback_rebinds_native_image_gate(monkeypatch):
    """The gate must not keep the primary route after an actual fallback bind."""
    from agent import image_routing

    primary = ("glm-codingplan", "GLM-5.3-Flash")
    fallback = ("xiaomi", "mimo-v2.6-flash")
    agent = SimpleNamespace(
        provider=primary[0],
        model=primary[1],
        requested_provider=primary[0],
        base_url="https://example.invalid/v1",
        api_mode="chat_completions",
        api_key="",
        auth_mode="",
        session_id="vision-fallback",
        _transport_cache={},
        _anthropic_prompt_cache_policy=lambda **_kw: (False, False),
        _ensure_lmstudio_runtime_loaded=lambda: None,
    )
    route = RouteEntryClient(
        SimpleNamespace(base_url="https://api.xiaomimimo.com/v1"),
        fallback[1], True, "chat_completions", None,
    )
    monkeypatch.setattr(
        "agent.route_binding.resolve_route_entry_client", lambda *_a, **_kw: route
    )
    monkeypatch.setattr(
        "agent.prompt_cache_scope.resolve_prompt_cache_scope_safe", lambda _agent: ""
    )
    monkeypatch.setattr(
        "hermes_cli.providers.is_actual_route", lambda *_a, **_kw: True
    )

    # Keep the real route binder and the real native-vision decision. Isolate
    # credential pools, transport construction and state persistence only.
    def noop(*_args, **_kwargs):
        return None

    for name in (
        "agent.route_verdicts.save_primary_route_verdicts",
        "agent.route_verdicts.reset_route_verdicts",
        "agent.chat_completion_helpers._rebind_fallback_credential_pool",
        "agent.chat_completion_helpers._update_fallback_context_compressor",
        "agent.chat_completion_helpers._reresolve_fallback_reasoning_config",
        "agent.chat_completion_helpers._rescope_fallback_extra_body",
        "agent.client_lifecycle._swap_fallback_clients",
        "agent.agent_runtime_helpers.sync_credential_pool_entry_id",
    ):
        monkeypatch.setattr(name, noop)
    monkeypatch.setattr(
        image_routing, "decide_image_input_mode", lambda *_args: "native"
    )
    monkeypatch.setattr(
        image_routing, "_lookup_supports_vision", lambda *_args: True
    )

    set_runtime_main(*primary)
    try:
        assert bind_route_entry(agent, {}, *fallback) == (primary[1], primary[0])
        assert (_read_main_provider(), _read_main_model()) == fallback

        # The separate Xiaomi capability decision may change when #121833
        # lands. The fast-path gate must always use the SAME consuming route.
        expected = _native_tool_result_images(*fallback, load_config())
        assert _should_use_native_vision_fast_path() is expected
    finally:
        clear_runtime_main()
