from tools.delegate_tool import _build_observability


def test_build_observability_treats_pareto_code_as_router_resolution():
    obs = _build_observability([
        {
            "model_request": "openrouter/pareto-code",
            "model_response": "anthropic/claude-sonnet-4.6",
            "match": False,
            "tokens_in": 10,
            "tokens_out": 5,
            "duration_s": 0.25,
        }
    ])

    assert obs is not None
    assert obs["pareto_router_resolutions"] == {
        "openrouter/pareto-code": {"anthropic/claude-sonnet-4.6": 1}
    }
    assert "override_mismatches" not in obs


def test_build_observability_still_warns_on_real_override_mismatch():
    obs = _build_observability([
        {
            "model_request": "anthropic/claude-opus-4.7",
            "model_response": "google/gemini-2.5-flash",
            "match": False,
            "tokens_in": 10,
            "tokens_out": 5,
            "duration_s": 0.25,
        }
    ])

    assert obs is not None
    assert obs["override_mismatches"] == [
        {"requested": "anthropic/claude-opus-4.7", "actual": "google/gemini-2.5-flash"}
    ]
    assert "pareto_router_resolutions" not in obs
