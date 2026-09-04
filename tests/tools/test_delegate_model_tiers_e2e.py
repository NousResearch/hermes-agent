"""End-to-end: a tiered task actually reaches _build_child_agent with the
tier's model, while an untiered sibling in the SAME batch keeps the default.

Guards the wiring (schema -> _resolve_tier_credentials -> per-task creds ->
_build_child_agent), which the unit tests do not cover: a mixed batch is the
case where a single batch-level `creds` object would silently win.
"""

from unittest.mock import MagicMock, patch

import tools.delegate_tool as dt


CFG = {
    "max_iterations": 5,
    "base_url": "https://example.test/v1",
    "api_key": "test-key-1234567890",
    "model": "default-model",
    "model_tiers": {
        "fast": {"model": "cheap-model"},
        "deep": {"model": "expensive-model"},
    },
}


def _parent():
    parent = MagicMock()
    parent._delegate_depth = 0
    parent.request_overrides = None
    parent.session_id = "sess-test"
    return parent


def _dispatch(tasks):
    """Run delegate_task far enough to capture per-child construction."""
    built = []

    def _fake_build(**kwargs):
        built.append(kwargs)
        return MagicMock()

    with patch.object(dt, "_load_config", return_value=CFG), \
         patch.object(dt, "_build_child_preserving_parent_tools", side_effect=_fake_build), \
         patch("tools.async_delegation.dispatch_async_delegation_batch",
               return_value={"status": "dispatched", "delegation_id": "d1"}), \
         patch("tools.delegation_live_log.create_live_transcripts",
               return_value=(None, [None] * len(tasks), [])):
        dt.delegate_task(tasks=tasks, parent_agent=_parent(), background=True)
    return built


def test_mixed_batch_routes_each_child_to_its_own_model():
    built = _dispatch([
        {"goal": "Reformat the changelog entries into the house style."},
        {"goal": "Redesign the retry policy for the ingest pipeline.",
         "model_tier": "deep"},
        {"goal": "Extract every TODO comment under src/ into a list.",
         "model_tier": "fast"},
    ])
    assert [b["model"] for b in built] == [
        "default-model",    # untiered -> global delegation pin
        "expensive-model",  # deep
        "cheap-model",      # fast
    ]


def test_unknown_tier_refuses_before_any_child_is_built():
    """No child may spawn when one task names a bad tier — a partially
    dispatched batch would leave orphaned children on a typo."""
    built = []
    with patch.object(dt, "_load_config", return_value=CFG), \
         patch.object(dt, "_build_child_preserving_parent_tools",
                      side_effect=lambda **kw: built.append(kw)):
        result = dt.delegate_task(
            tasks=[
                {"goal": "A perfectly valid first task goal here."},
                {"goal": "Second task naming a tier that does not exist.",
                 "model_tier": "turbo"},
            ],
            parent_agent=_parent(),
            background=True,
        )
    assert built == []
    assert "turbo" in str(result)
