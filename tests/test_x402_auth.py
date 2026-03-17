from hermes_cli import x402_auth


def test_usdc_to_units_uses_two_dollar_default_shape():
    assert x402_auth.usdc_to_units("2") == "2000000"


def test_should_invalidate_permit_for_cap_exhausted():
    assert x402_auth.should_invalidate_permit({"code": "cap_exhausted"}) is True


def test_should_invalidate_permit_for_session_closed_message():
    assert x402_auth.should_invalidate_permit({"message": "session closed"}) is True
