from tui_gateway import prompt_intents
from tui_gateway.prompt_intents import PromptIntentClaim, PromptIntentLedger


def test_prompt_intent_ledger_deduplicates_and_rejects_conflicting_reuse():
    ledger = PromptIntentLedger()
    claim = {
        "profile_scope": "/profiles/default",
        "request_id": "intent-1",
        "route_identity": "stored-a",
        "text": "hello",
        "truncate_ordinal": None,
    }

    assert ledger.claim(**claim) is PromptIntentClaim.ACCEPTED
    assert ledger.claim(**claim) is PromptIntentClaim.DUPLICATE
    assert ledger.claim(**{**claim, "text": "different"}) is PromptIntentClaim.CONFLICT
    assert (
        ledger.claim(**{**claim, "route_identity": "stored-b"})
        is PromptIntentClaim.CONFLICT
    )
    assert ledger.claim(**{**claim, "request_id": "x" * 257}) is PromptIntentClaim.INVALID
    assert len(ledger) == 1

    ledger.abort(profile_scope=claim["profile_scope"], request_id=claim["request_id"])
    assert len(ledger) == 0
    assert ledger.claim(**claim) is PromptIntentClaim.ACCEPTED


def test_prompt_intent_ledger_admits_beyond_the_previous_process_ceiling():
    ledger = PromptIntentLedger()

    for index in range(8193):
        assert (
            ledger.claim(
                profile_scope="profile",
                request_id=f"intent-{index}",
                route_identity="stored-a",
                text=f"prompt-{index}",
                truncate_ordinal=None,
            )
            is PromptIntentClaim.ACCEPTED
        )

    assert len(ledger) == 8193


def test_prompt_intent_ledger_persists_and_prunes_only_after_ttl(
    monkeypatch, tmp_path
):
    now = {"value": 100.0}
    monkeypatch.setattr(prompt_intents, "_MIN_TTL_S", 10)
    monkeypatch.setattr(prompt_intents.time, "time", lambda: now["value"])
    db_path = tmp_path / "prompt-intents.sqlite3"
    ledger = PromptIntentLedger(db_path=db_path)

    claim = {
        "profile_scope": "profile",
        "request_id": "first",
        "route_identity": "stored-a",
        "text": "one",
        "truncate_ordinal": None,
    }

    assert ledger.claim(**claim) is PromptIntentClaim.ACCEPTED
    ledger.close()

    now["value"] = 105.0
    reopened = PromptIntentLedger(db_path=db_path)
    assert reopened.claim(**claim) is PromptIntentClaim.DUPLICATE
    assert (
        reopened.claim(**{**claim, "request_id": "second", "text": "two"})
        is PromptIntentClaim.ACCEPTED
    )
    assert len(reopened) == 2

    now["value"] = 116.0
    assert reopened.claim(**claim) is PromptIntentClaim.ACCEPTED
    assert len(reopened) == 1
    reopened.close()
