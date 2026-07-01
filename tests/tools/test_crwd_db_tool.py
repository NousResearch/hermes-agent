"""Tests for the crwd_db tool module (no live database required)."""

import json
from unittest.mock import MagicMock, patch

import pytest

from tools import crwd_db_tool as t


class TestAvailability:
    def test_unavailable_without_uri(self, monkeypatch):
        monkeypatch.delenv("CRWD_MONGO_URI", raising=False)
        assert t.check_crwd_db_requirements() is False

    def test_available_with_uri(self, monkeypatch):
        monkeypatch.setenv("CRWD_MONGO_URI", "mongodb://localhost:27017/")
        assert t.check_crwd_db_requirements() is True

    def test_handler_errors_without_uri(self, monkeypatch):
        monkeypatch.delenv("CRWD_MONGO_URI", raising=False)
        out = json.loads(t.crwd_db_tool({"action": "list_active_gigs"}))
        assert "error" in out


class TestNormalizeAndScore:
    def test_normalize_strips_noise_words(self):
        assert t._normalize("The Self Obsessed Supplement Gig") == "self obsessed"

    def test_normalize_falls_back_when_all_noise(self):
        # If every token is a noise word, keep the original tokens rather than "".
        assert t._normalize("the a an") == "the a an"

    def test_exact_name_scores_high(self):
        q = t._normalize("self obsessed")
        assert t._score(q, "Self Obsessed - Supplement") >= 0.9

    def test_partial_name_beats_unrelated(self):
        q = t._normalize("self obsessed")
        strong = t._score(q, "Self Obsessed Maxed - Supplement")
        weak = t._score(q, "Review a Gym on Yelp")
        assert strong > weak

    def test_garbage_scores_below_floor(self):
        q = t._normalize("zzzqqq nonexistent xyzzy")
        assert t._score(q, "Self Obsessed - Supplement") < t._MATCH_FLOOR

    def test_description_match_boosts(self):
        q = t._normalize("testosterone")
        assert t._score(q, "Random Gig", "boosts testosterone support") >= 0.5

    def test_empty_query_scores_zero(self):
        assert t._score("", "anything") == 0.0


class TestGuardHelpers:
    def test_has_where_top_level(self):
        assert t._has_where({"$where": "1==1"}) is True

    def test_has_where_nested(self):
        assert t._has_where({"a": {"b": {"$where": "x"}}}) is True

    def test_has_where_in_list(self):
        assert t._has_where({"$or": [{"x": 1}, {"$where": "y"}]}) is True

    def test_has_where_absent(self):
        assert t._has_where({"city": "Austin", "isDeleted": {"$ne": True}}) is False

    def test_redact_secrets_drops_secrets(self):
        doc = {
            "email": "a@b.com", "password": "hash", "emailOTP": "123",
            "emailForgotPasswordVerifyToken": "tok", "resetSecret": "s",
        }
        red = t._redact_secrets(doc)
        assert red == {"email": "a@b.com"}

    def test_redact_secrets_recurses(self):
        doc = {"nested": {"token": "x", "keep": 1}}
        assert t._redact_secrets(doc) == {"nested": {"keep": 1}}

    def test_redact_secrets_drops_notification_tokens(self):
        doc = {"title": "hi", "deviceToken": "d", "webDeviceToken": "w", "chat_token": "c"}
        assert t._redact_secrets(doc) == {"title": "hi"}

    def test_id_values_objectid_and_string(self):
        vals = t._id_values("69a72d9b2109705cc0224a35")
        assert len(vals) == 2 and "69a72d9b2109705cc0224a35" in vals

    def test_id_values_plain_string_only(self):
        assert t._id_values("6a33bb6003b1c0cc31a7baa5x") == ["6a33bb6003b1c0cc31a7baa5x"]

    def test_effective_payout_prefers_top_level(self):
        assert t._effective_payout({"payout": 25, "gig_stores": [{"payout_amount": 5}]}) == 25

    def test_effective_payout_falls_back_to_stores(self):
        gig = {"payout": 0, "gig_stores": [{"payout_amount": 3}, {"payout_amount": 7}]}
        assert t._effective_payout(gig) == 7

    def test_effective_payout_no_stores(self):
        assert t._effective_payout({"payout": 0}) == 0


class TestOid:
    def test_valid_24_hex(self):
        assert t._oid("69a72d9b2109705cc0224a35") is not None

    def test_invalid_returns_none(self):
        assert t._oid("not-an-id") is None
        assert t._oid("") is None


def _fake_db(collections):
    """Return a fake db mapping so _db()[name] yields the given collection mock."""
    db = MagicMock()
    db.__getitem__.side_effect = lambda name: collections[name]
    return db


class TestCustomQueryGuardrails:
    def test_disallowed_collection(self, monkeypatch):
        monkeypatch.setenv("CRWD_MONGO_URI", "mongodb://x/")
        out = json.loads(t.crwd_db_tool({
            "action": "custom_query", "collection": "orders", "operation": "find",
        }))
        assert "error" in out and "collection" in out["error"]

    def test_bad_operation(self, monkeypatch):
        monkeypatch.setenv("CRWD_MONGO_URI", "mongodb://x/")
        out = json.loads(t.crwd_db_tool({
            "action": "custom_query", "collection": "crwds", "operation": "aggregate",
        }))
        assert "error" in out

    def test_where_rejected(self, monkeypatch):
        monkeypatch.setenv("CRWD_MONGO_URI", "mongodb://x/")
        out = json.loads(t.crwd_db_tool({
            "action": "custom_query", "collection": "crwds", "operation": "find",
            "filter": {"$where": "1==1"},
        }))
        assert out["error"] == "$where is not allowed"

    def test_limit_capped_at_20(self, monkeypatch):
        monkeypatch.setenv("CRWD_MONGO_URI", "mongodb://x/")
        coll = MagicMock()
        cursor = MagicMock()
        coll.find.return_value = cursor
        cursor.limit.return_value = []
        with patch.object(t, "_db", return_value=_fake_db({"crwds": coll})):
            t.crwd_db_tool({
                "action": "custom_query", "collection": "crwds", "operation": "find",
                "limit": 9999,
            })
        cursor.limit.assert_called_once_with(t._HARD_LIMIT)

    def test_users_projection_redacted(self, monkeypatch):
        monkeypatch.setenv("CRWD_MONGO_URI", "mongodb://x/")
        coll = MagicMock()
        cursor = MagicMock()
        coll.find.return_value = cursor
        cursor.limit.return_value = [
            {"email": "a@b.com", "password": "hash", "emailOTP": "1"}
        ]
        with patch.object(t, "_db", return_value=_fake_db({"users": coll})):
            out = json.loads(t.crwd_db_tool({
                "action": "custom_query", "collection": "users", "operation": "find",
                "projection": {"email": 1, "password": 1},
            }))
        assert out["items"] == [{"email": "a@b.com"}]


class TestNewUserActions:
    @pytest.mark.parametrize("action", [
        "get_user_products", "get_user_receipts", "get_user_notifications",
    ])
    def test_require_user_id(self, monkeypatch, action):
        monkeypatch.setenv("CRWD_MONGO_URI", "mongodb://x/")
        out = json.loads(t.crwd_db_tool({"action": action, "user_id": ""}))
        assert "error" in out and "user_id" in out["error"]

    def test_new_collections_in_allowlist(self):
        assert {"user_product_purchases", "receipt_upload_history", "notifications"} <= t._ALLOWED_COLLECTIONS

    def test_get_user_products_shape(self, monkeypatch):
        monkeypatch.setenv("CRWD_MONGO_URI", "mongodb://x/")
        coll = MagicMock()
        cursor = MagicMock()
        coll.find.return_value = cursor
        cursor.sort.return_value = cursor
        cursor.limit.return_value = [{"product_name": "X", "product_url": "http://u"}]
        with patch.object(t, "_db", return_value=_fake_db({"user_product_purchases": coll})):
            out = json.loads(t.crwd_db_tool({"action": "get_user_products", "user_id": "abc"}))
        assert out["_type"] == "user_products"
        assert out["items"][0]["product_name"] == "X"

    def test_notifications_custom_query_redacts_tokens(self, monkeypatch):
        monkeypatch.setenv("CRWD_MONGO_URI", "mongodb://x/")
        coll = MagicMock()
        cursor = MagicMock()
        coll.find.return_value = cursor
        cursor.limit.return_value = [{"title": "hi", "deviceToken": "d", "chat_token": "c"}]
        with patch.object(t, "_db", return_value=_fake_db({"notifications": coll})):
            out = json.loads(t.crwd_db_tool({
                "action": "custom_query", "collection": "notifications", "operation": "find",
            }))
        assert out["items"] == [{"title": "hi"}]


class TestRouter:
    def test_unknown_action(self, monkeypatch):
        monkeypatch.setenv("CRWD_MONGO_URI", "mongodb://x/")
        out = json.loads(t.crwd_db_tool({"action": "frobnicate"}))
        assert "error" in out

    def test_unexpected_exception_generic_error(self, monkeypatch):
        monkeypatch.setenv("CRWD_MONGO_URI", "mongodb://x/")
        with patch.object(t, "_db", side_effect=Exception("driver boom")):
            out = json.loads(t.crwd_db_tool({"action": "list_active_gigs"}))
        # Raw driver error must not leak to the model.
        assert out == {"error": "query failed"}
