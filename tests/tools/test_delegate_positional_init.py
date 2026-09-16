"""ACP options must preserve the child builder's positional routing tail."""


def test_legacy_positional_routing_config_reaches_real_child():
    from run_agent import AIAgent
    from tools.delegate_tool import _build_child_agent

    fallback_chain = [{"provider": "custom", "model": "fallback-model"}]
    routing_cfg = {"fallback_providers": fallback_chain}
    parent = AIAgent(
        base_url="http://127.0.0.1:9/v1", api_key="test-only",
        provider="custom", model="parent-model", max_iterations=3,
        quiet_mode=True, enabled_toolsets=[], skip_context_files=True,
        skip_memory=True, skip_background_review=True,
    )
    try:
        # Include both legacy tail arguments: routing_cfg then role.
        child = _build_child_agent(
            0, "verify positional routing", None, [], None, 2, 1, parent,
            None, None, None, None, None, None, None, routing_cfg, "leaf",
        )
        try:
            assert child.model == parent.model
            assert child.max_iterations == 2
            assert child._fallback_chain == fallback_chain
            assert child.acp_cwd is None
        finally:
            child.close()
    finally:
        parent.close()
