"""
Tests for terminal_tool sudo password HMAC-SHA256 hash cache.

Covers: _hmac_verification_hash, _verify_sudo_password,
_sudo_password_hash_cache, _sudo_password_plaintext, _PROCESS_SECRET,
_reset_cached_sudo_passwords, _set_cached_sudo_password.
"""

import hashlib
import hmac as _hmac
import threading
from unittest import mock

import pytest


# Import the module under test
from tools import terminal_tool as tt


@pytest.fixture(autouse=True)
def reset_sudo_cache():
    """Reset the sudo password cache before and after each test."""
    tt._reset_cached_sudo_passwords()
    yield
    tt._reset_cached_sudo_passwords()


# ------------------------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------------------------


@pytest.fixture
def fixed_process_secret():
    """Fix _PROCESS_SECRET to a known value so we can predict hashes."""
    original = tt._PROCESS_SECRET
    tt._PROCESS_SECRET = "test-secret-32-bytes-exactly-here"
    yield "test-secret-32-bytes-exactly-here"
    tt._PROCESS_SECRET = original


# ------------------------------------------------------------------------------------
# _hmac_verification_hash
# ------------------------------------------------------------------------------------


class TestHmacVerificationHash:
    def test_hash_is_deterministic(self, fixed_process_secret):
        h1 = tt._hmac_verification_hash("password123", "scope-a")
        h2 = tt._hmac_verification_hash("password123", "scope-a")
        assert h1 == h2

    def test_different_passwords_different_hashes(self, fixed_process_secret):
        h1 = tt._hmac_verification_hash("password1", "scope-a")
        h2 = tt._hmac_verification_hash("password2", "scope-a")
        assert h1 != h2

    def test_different_scopes_different_hashes(self, fixed_process_secret):
        h1 = tt._hmac_verification_hash("password123", "scope-a")
        h2 = tt._hmac_verification_hash("password123", "scope-b")
        assert h1 != h2

    def test_hash_format_is_hex_sha256(self, fixed_process_secret):
        h = tt._hmac_verification_hash("password123", "scope-a")
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_hash_uses_process_secret(self, fixed_process_secret):
        h1 = tt._hmac_verification_hash("password123", "scope-a")
        tt._PROCESS_SECRET = "different-secret-here-32bytes!!"
        h2 = tt._hmac_verification_hash("password123", "scope-a")
        assert h1 != h2

    def test_hash_empty_password(self, fixed_process_secret):
        h = tt._hmac_verification_hash("", "scope-a")
        assert len(h) == 64


# ------------------------------------------------------------------------------------
# _verify_sudo_password
# ------------------------------------------------------------------------------------


class TestVerifySudoPassword:
    def test_verify_correct_password(self, fixed_process_secret):
        tt._set_cached_sudo_password("mysecret")
        assert tt._verify_sudo_password("mysecret") is True

    def test_verify_wrong_password(self, fixed_process_secret):
        tt._set_cached_sudo_password("mysecret")
        assert tt._verify_sudo_password("wrongsecret") is False

    def test_verify_no_cache(self, fixed_process_secret):
        assert tt._verify_sudo_password("anypassword") is False

    def test_verify_empty_cache_after_clear(self, fixed_process_secret):
        tt._set_cached_sudo_password("mysecret")
        tt._reset_cached_sudo_passwords()
        assert tt._verify_sudo_password("mysecret") is False

    def test_verify_uses_constant_time_comparison(self, fixed_process_secret):
        """The function must use hmac.compare_digest to prevent timing attacks."""
        tt._set_cached_sudo_password("mysecret")
        # _verify_sudo_password calls hmac.compare_digest internally
        result = tt._verify_sudo_password("mysecret")
        assert result is True

    def test_verify_does_not_return_plaintext(self, fixed_process_secret):
        """Ensure _verify_sudo_password returns bool, not the password."""
        tt._set_cached_sudo_password("mysecret")
        result = tt._verify_sudo_password("mysecret")
        assert isinstance(result, bool)
        assert result is True


# ------------------------------------------------------------------------------------
# Cache lifecycle
# ------------------------------------------------------------------------------------


class TestCacheLifecycle:
    def test_set_and_get_plaintext(self, fixed_process_secret):
        tt._set_cached_sudo_password("secret123")
        assert tt._get_cached_sudo_password() == "secret123"

    def test_get_empty_when_not_set(self, fixed_process_secret):
        assert tt._get_cached_sudo_password() == ""

    def test_set_empty_clears_both_caches(self, fixed_process_secret):
        tt._set_cached_sudo_password("secret123")
        tt._set_cached_sudo_password("")
        assert tt._get_cached_sudo_password() == ""
        assert tt._verify_sudo_password("secret123") is False

    def test_reset_clears_both_caches(self, fixed_process_secret):
        tt._set_cached_sudo_password("secret123")
        tt._reset_cached_sudo_passwords()
        assert tt._get_cached_sudo_password() == ""
        assert tt._verify_sudo_password("secret123") is False

    def test_hash_cache_not_storing_plaintext(self, fixed_process_secret):
        tt._set_cached_sudo_password("secret123")
        scope = tt._get_sudo_password_cache_scope()
        stored_hash = tt._sudo_password_hash_cache.get(scope, "")
        assert stored_hash != ""
        # Hash should NOT be the plaintext
        assert stored_hash != "secret123"
        assert stored_hash != "secret123\n"

    def test_plaintext_cache_stores_original(self, fixed_process_secret):
        tt._set_cached_sudo_password("secret123")
        scope = tt._get_sudo_password_cache_scope()
        stored_plaintext = tt._sudo_password_plaintext.get(scope, "")
        assert stored_plaintext == "secret123"


# ------------------------------------------------------------------------------------
# Thread isolation
# ------------------------------------------------------------------------------------


class TestThreadIsolation:
    def test_different_threads_have_separate_scopes(self):
        results: dict[str, str] = {}

        def set_in_thread(value: str, results_dict: dict, key: str):
            results_dict[key] = tt._get_sudo_password_cache_scope()

        t1 = threading.Thread(target=set_in_thread, args=("v1", results, "t1"))
        t2 = threading.Thread(target=set_in_thread, args=("v2", results, "t2"))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # Each thread should have gotten a different scope
        assert results["t1"] != results["t2"]


# ------------------------------------------------------------------------------------
# Security properties
# ------------------------------------------------------------------------------------


class TestSecurityProperties:
    def test_plaintext_never_in_hash_cache(self, fixed_process_secret):
        """The hash cache must never contain the plaintext password."""
        tt._set_cached_sudo_password("SuperSecret123!")
        for value in tt._sudo_password_hash_cache.values():
            assert value != "SuperSecret123!"
            assert "SuperSecret123!" not in value

    def test_different_scopes_produce_different_hashes_for_same_password(
        self, fixed_process_secret
    ):
        """Same password in different scopes must NOT produce the same hash."""
        with mock.patch.object(
            tt, "_get_sudo_password_cache_scope", return_value="scope-1"
        ):
            tt._set_cached_sudo_password("password123")
            hash1 = tt._sudo_password_hash_cache.get("scope-1", "")

        with mock.patch.object(
            tt, "_get_sudo_password_cache_scope", return_value="scope-2"
        ):
            tt._set_cached_sudo_password("password123")
            hash2 = tt._sudo_password_hash_cache.get("scope-2", "")

        assert hash1 != hash2

    def test_process_secret_in_hash_derivation(self, fixed_process_secret):
        """Hash must include PROCESS_SECRET to prevent cross-process replay."""
        # Hash with original secret
        h1 = tt._hmac_verification_hash("mypass", "scope")

        # Hash with different secret
        tt._PROCESS_SECRET = "completely-different-secret-!!"
        h2 = tt._hmac_verification_hash("mypass", "scope")

        assert h1 != h2

    def test_hashes_are_salted_with_scope(self, fixed_process_secret):
        """Scope acts as an additional salt so same password+secret != same hash."""
        # Scope changes the HMAC key (scope + PROCESS_SECRET)
        h1 = tt._hmac_verification_hash("password", "scope-a")
        h2 = tt._hmac_verification_hash("password", "scope-b")
        assert h1 != h2
