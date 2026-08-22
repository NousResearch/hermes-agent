from research.evidence_fabric_tool_gate import MODEL_TOOL_DEFERRED, model_tool_go_no_go


def test_model_tool_is_deferred_without_trusted_runtime_scope_context():
    go, decision = model_tool_go_no_go()
    assert go is False
    assert decision == MODEL_TOOL_DEFERRED
