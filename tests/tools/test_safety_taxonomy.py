from tools.safety_taxonomy import classify_tool_action


def test_classify_read_only_tool():
    c = classify_tool_action("read_file")
    assert c.risk_class == "low"
    assert c.action_type == "read_only"
    assert c.approval_required is False


def test_classify_reversible_tool_requires_approval():
    c = classify_tool_action("patch")
    assert c.risk_class == "medium"
    assert c.approval_required is True
    assert c.rollback_expected is True


def test_classify_irreversible_tool_is_high_risk():
    c = classify_tool_action("terminal")
    assert c.risk_class == "high"
    assert c.action_type == "irreversible_side_effect"
