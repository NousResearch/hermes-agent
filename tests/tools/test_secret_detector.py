"""Tests for tools/secret_detector.py — credential detection in model context.

These pin the invariant: "Persistent context intended for injection into an
LLM must not contain probable authentication credentials."

All canary values are unmistakably fake (CANARY_*, hunter2, ghp_..., etc.).

The detector must:
  * reject prose and direct-assignment credential forms,
  * keep prompt-injection detection conceptually separate (this module finds
    *values*, not *instructions*),
  * never return or echo the matched secret value,
  * accept harmless sentences that merely mention credentials.
"""

import pytest

from tools.secret_detector import (
    SecretFinding,
    contains_secret,
    first_secret_message,
    scan_for_secrets,
    scan_memory_for_secrets,
)

# Synthetic, obviously-fake canaries only.
CANARY_PROSE = "Password for Test WebUI on this machine: CANARY_PASSWORD_7F39C1_A91D2E"
CANARY_ASSIGN = 'password = "CANARY_PASSWORD_7F39C1_A91D2E"'
CANARY_API_PROSE = "The api key is CANARY_APIKEY_ABC123DEF456GHI789"
CANARY_BEARER = "Authorization: Bearer CANARYeyJabc123def456ghi789"
CANARY_BASIC = "Authorization: Basic CANARYdXNlcjpwYXNzd29yZA=="
CANARY_PRIVKEY = (
    "-----BEGIN RSA PRIVATE KEY-----\n"
    "CANARYMIIEogIBAAKCAQEAexamplekeyvalue1234567890\n"
    "-----END RSA PRIVATE KEY-----"
)
CANARY_CONNSTR = "postgres://admin:CANARY_Sup3rSecret@db.example.com:5432/app"
CANARY_URL_USERINFO = "https://user:CANARY_FancyPassw0rd@api.example.com/v1"
CANARY_GH_PAT = "token: ghp_CANARYabcdefghijklmnopqrstuvwxyz0123"


class TestDetectsCredentials:
    def test_prose_password_rejected(self):
        assert contains_secret(CANARY_PROSE)

    def test_direct_assignment_rejected(self):
        assert contains_secret(CANARY_ASSIGN)

    def test_api_token_prose_rejected(self):
        assert contains_secret(CANARY_API_PROSE)

    def test_authorization_header_rejected(self):
        assert contains_secret(CANARY_BEARER)
        assert contains_secret(CANARY_BASIC)

    def test_private_key_material_rejected(self):
        assert contains_secret(CANARY_PRIVKEY)

    def test_connection_string_rejected(self):
        assert contains_secret(CANARY_CONNSTR)

    def test_url_userinfo_rejected(self):
        assert contains_secret(CANARY_URL_USERINFO)

    def test_github_pat_rejected(self):
        assert contains_secret(CANARY_GH_PAT)

    def test_first_secret_message_explains_and_does_not_echo(self):
        msg = first_secret_message(CANARY_PROSE)
        assert msg is not None
        assert "MEMORY.md" in msg or "memory" in msg.lower()
        # The actual secret value must never appear in the rejection message.
        assert "CANARY_PASSWORD_7F39C1_A91D2E" not in msg


class TestAcceptsHarmless:
    @pytest.mark.parametrize("text", [
        "User rotates passwords monthly for security hygiene.",
        "The project has password authentication enabled.",
        "Never store API keys in source control or memory.",
        "The password is required to log in to the VPN.",
        "The password field is on the left of the form.",
        "An API key is needed for the service; obtain one from the portal.",
        "We are organizing a secret santa gift exchange this year.",
        "The token economy of the platform rewards active users.",
        "The meeting is at noon and the report is due Friday.",
        "Set a password.",
        # 'password' followed by a normal word, not a value
        "My password manager suggested a long passphrase.",
    ])
    def test_no_false_positive_on_credential_mention(self, text):
        assert not contains_secret(text)

    def test_first_secret_message_clean_is_none(self):
        assert first_secret_message("User rotates passwords monthly.") is None


class TestFindingIsValueFree:
    def test_finding_has_no_secret_value(self):
        findings = scan_for_secrets(CANARY_PROSE)
        assert findings
        for f in findings:
            assert isinstance(f, SecretFinding)
            assert "CANARY_PASSWORD_7F39C1_A91D2E" not in f.category
            assert "CANARY_PASSWORD_7F39C1_A91D2E" not in f.marker

    def test_finding_repr_is_safe(self):
        findings = scan_for_secrets(CANARY_PROSE)
        for f in findings:
            assert "CANARY_PASSWORD_7F39C1_A91D2E" not in str(f)


class TestEmptyInput:
    def test_empty_returns_empty(self):
        assert scan_for_secrets("") == []
        assert not contains_secret("")
        assert first_secret_message("") is None
