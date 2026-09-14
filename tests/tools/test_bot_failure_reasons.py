"""Tests: typed failure-reason codes (tools/bot_failure_reasons.py, #93091).

Pins the closed reason vocabulary, the ordered classifier (incl. the
auth-beats-quota precedence seen in real Anthropic 401 bodies), the three
real-world fixtures from live bot runs, and the auto-retryable set.
"""

import pytest

from tools import bot_failure_reasons as fr

# Real error text captured from live bot turns.
FIXTURE_ANTHROPIC_401 = (
    "Error code: 401 - {'type': 'error', 'error': {'type': 'authentication_error', "
    "'message': 'Your API key is invalid, blocked or out of funds...'}}"
)
FIXTURE_NO_PROVIDER = (
    "agent init failed: No LLM provider configured. Run `hermes model` to select "
    "a provider, or run `hermes setup` for first-time configuration."
)
FIXTURE_NO_TOKEN = "agent init failed: No access token found for Nous Portal login."


def test_closed_vocabulary_contains_every_code():
    assert fr.ALL_REASONS == {
        value for name, value in vars(fr).items()
        if name.isupper() and isinstance(value, str) and value == name.lower()
    }
    # constants match their string values
    assert fr.RUNTIME_OFFLINE == "runtime_offline"
    assert fr.PROVIDER_AUTH_OR_ACCESS == "provider_auth_or_access"
    assert fr.UNKNOWN == "unknown"


@pytest.mark.parametrize(
    ("text", "code"),
    [
        ("Error code: 403 - forbidden", fr.PROVIDER_AUTH_OR_ACCESS),
        ("Invalid API key provided", fr.PROVIDER_AUTH_OR_ACCESS),
        ("Error code: 402 - payment required", fr.PROVIDER_QUOTA_LIMIT),
        ("insufficient balance, top up your account", fr.PROVIDER_QUOTA_LIMIT),
        ("You exceeded your current quota", fr.PROVIDER_QUOTA_LIMIT),
        ("Error code: 429 - Too Many Requests", fr.PROVIDER_RATE_LIMIT),
        ("Rate limit reached for gpt-4o", fr.PROVIDER_RATE_LIMIT),
        ("Error code: 500 - internal server error", fr.PROVIDER_SERVER_ERROR),
        ("Error code: 529 - overloaded_error: Overloaded", fr.PROVIDER_SERVER_ERROR),
        ("This model's maximum context length is 128000 tokens", fr.CONTEXT_OVERFLOW),
        ("context_overflow: prompt too large", fr.CONTEXT_OVERFLOW),
        ("missing config: no provider block in config.yaml", fr.MISSING_CONFIG),
        ("model 'gpt-9' not found", fr.MODEL_UNAVAILABLE),
        ("The model `foo-bar` does not exist", fr.MODEL_UNAVAILABLE),
        ("model_not_found", fr.MODEL_UNAVAILABLE),
        ("status: 401 unauthorized", fr.PROVIDER_AUTH_OR_ACCESS),
        ("upstream server error", fr.PROVIDER_SERVER_ERROR),
        # bare numbers WITHOUT a status-code context must not classify —
        # they feed AUTO_RETRYABLE and a misfire could auto-retry a
        # permanent local failure (review finding on #93101).
        ("gate check failed: error at line 502 of module", fr.UNKNOWN),
        ("took 429 ms to fail", fr.UNKNOWN),
        ("something inexplicable happened", fr.UNKNOWN),
        ("", fr.UNKNOWN),
        (None, fr.UNKNOWN),
    ],
)
def test_classify_agent_error_rules(text, code):
    assert fr.classify_agent_error(text) == code


def test_fixture_anthropic_401_auth_beats_quota():
    # The live 401 body ALSO says "out of funds" — auth wins by precedence.
    assert "out of funds" in FIXTURE_ANTHROPIC_401
    assert fr.classify_agent_error(FIXTURE_ANTHROPIC_401) == fr.PROVIDER_AUTH_OR_ACCESS


def test_precedence_authentication_error_type_alone_beats_quota_words():
    text = "authentication_error: account is out of funds"
    assert fr.classify_agent_error(text) == fr.PROVIDER_AUTH_OR_ACCESS
    # but plain quota text without any auth marker classifies as quota
    assert fr.classify_agent_error("account is out of funds") == fr.PROVIDER_QUOTA_LIMIT


def test_fixture_no_provider_configured_is_missing_config():
    assert fr.classify_agent_error(FIXTURE_NO_PROVIDER) == fr.MISSING_CONFIG


def test_fixture_no_access_token_is_missing_config():
    assert fr.classify_agent_error(FIXTURE_NO_TOKEN) == fr.MISSING_CONFIG


def test_auto_retryable_set_and_predicate():
    assert fr.AUTO_RETRYABLE == {
        fr.RUNTIME_OFFLINE,
        fr.DELIVERY_TIMEOUT,
        fr.PROVIDER_RATE_LIMIT,
        fr.PROVIDER_SERVER_ERROR,
    }
    for code in fr.AUTO_RETRYABLE:
        assert fr.is_auto_retryable(code)
    for code in fr.ALL_REASONS - fr.AUTO_RETRYABLE:
        assert not fr.is_auto_retryable(code)
    assert not fr.is_auto_retryable("")
    assert not fr.is_auto_retryable("nonsense")
