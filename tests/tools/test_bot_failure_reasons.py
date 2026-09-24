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
# The two spellings of a target-scope spawn refusal, captured verbatim from the relay ledger after a
# relayed DM into a PEER's default profile (the raising process is the TARGET's, not the sender's).
FIXTURE_TARGET_SCOPE = (
    "Hermes could not read this profile's API key (an internal profile-scoping bug on the "
    "multiplexed gateway, not your configuration). Run `hermes gateway restart`; if it keeps "
    "happening, report it with `hermes debug share`."
)
FIXTURE_TARGET_SCOPE_DETAILED = (
    "served_profile_child_env(inherit_credentials=True) called with no target home and no profile "
    "secret scope bound while multiplexing is on; the child would inherit the launch profile's "
    "credentials. Bind the profile scope (or pass target_home) at the spawn site."
)




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
        (FIXTURE_TARGET_SCOPE, fr.TARGET_SCOPE_UNRESOLVED),
        (FIXTURE_TARGET_SCOPE_DETAILED, fr.TARGET_SCOPE_UNRESOLVED),
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


def test_target_scope_refusal_wins_over_the_api_key_wording_and_never_auto_retries():
    """The copy says "API key" and names a gateway restart, but nothing was ever sent to a provider:
    the delivery turn was never created because a spawn site could not resolve the target profile's
    secret scope. Reading it as auth would put it on the wrong ladder and send the reader to a
    restart on the wrong machine; it must classify as its own class and never auto-retry."""
    assert "api key" in FIXTURE_TARGET_SCOPE.lower()  # the word that must not win
    for text in (FIXTURE_TARGET_SCOPE, FIXTURE_TARGET_SCOPE_DETAILED):
        code = fr.classify_agent_error(text)
        assert code == fr.TARGET_SCOPE_UNRESOLVED
        assert code in fr.ALL_REASONS
        assert not fr.is_auto_retryable(code)
        assert fr.retry_action(code) == fr.RETRY_NONE


def test_auto_retryable_set_and_predicate():
    for code in fr.AUTO_RETRYABLE:
        assert fr.is_auto_retryable(code)
    for code in fr.ALL_REASONS - fr.AUTO_RETRYABLE:
        assert not fr.is_auto_retryable(code)
    assert not fr.is_auto_retryable("")
    assert not fr.is_auto_retryable("nonsense")
