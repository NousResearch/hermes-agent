from __future__ import annotations

from wisdom.commands import WisdomCommandContext, handle_wisdom_command


def test_status_help_on_off_commands(wisdom_db, wisdom_config):
    assert "Wisdom status" in handle_wisdom_command("status", config=wisdom_config, db=wisdom_db)
    assert "Wisdom commands" in handle_wisdom_command("help", config=wisdom_config, db=wisdom_db)
    assert "Wisdom is off" in handle_wisdom_command("off", config=wisdom_config, db=wisdom_db)
    assert "Use /wisdom on" in handle_wisdom_command("inbox", config=wisdom_config, db=wisdom_db)
    assert "Wisdom is on" in handle_wisdom_command("on", config=wisdom_config, db=wisdom_db)


def test_capture_inbox_search_original_interpret_apply_archive_review(wisdom_db, wisdom_config):
    context = WisdomCommandContext(channel="test", source_kind="command", session_key="s1", message_ref="m1")
    response = handle_wisdom_command(
        "capture Clients need windshields, not rear-view mirrors.",
        context=context,
        config=wisdom_config,
        db=wisdom_db,
    )
    assert response.startswith("Captured #1")
    assert "Clients need" in handle_wisdom_command("inbox", config=wisdom_config, db=wisdom_db)
    assert "Clients need" in handle_wisdom_command("search windshields", config=wisdom_config, db=wisdom_db)
    assert handle_wisdom_command("original 1", config=wisdom_config, db=wisdom_db) == "Clients need windshields, not rear-view mirrors."
    assert "Counterpoint:" in handle_wisdom_command("interpret 1", config=wisdom_config, db=wisdom_db)
    assert "Application proposals for #1" in handle_wisdom_command("apply 1", config=wisdom_config, db=wisdom_db)
    assert "Wisdom Review" in handle_wisdom_command("review", config=wisdom_config, db=wisdom_db)
    assert "Archived #1" in handle_wisdom_command("archive 1", config=wisdom_config, db=wisdom_db)
    assert "No captures found" in handle_wisdom_command("inbox", config=wisdom_config, db=wisdom_db)


def test_review_related_accept_and_dismiss_commands(wisdom_db, wisdom_config):
    handle_wisdom_command(
        "capture Reports are rear-view mirrors when clients need windshields.",
        config=wisdom_config,
        db=wisdom_db,
    )
    handle_wisdom_command(
        "capture Client reports should show the road ahead, not just last quarter.",
        config=wisdom_config,
        db=wisdom_db,
    )

    review = handle_wisdom_command("review high-potential", config=wisdom_config, db=wisdom_db)
    assert "Wisdom Review" in review
    assert "High-potential" in review

    related = handle_wisdom_command("related 1", config=wisdom_config, db=wisdom_db)
    assert "Related captures for #1" in related
    assert "#2" in related

    accepted = handle_wisdom_command("accept 1", config=wisdom_config, db=wisdom_db)
    assert "Accepted #1" in accepted
    assert wisdom_db.get_capture(1).review_status == "accepted"

    dismissed = handle_wisdom_command("dismiss 2", config=wisdom_config, db=wisdom_db)
    assert "Dismissed #2" in dismissed
    assert wisdom_db.get_capture(2).review_status == "dismissed"


def test_secret_command_capture_blocked(wisdom_db, wisdom_config):
    response = handle_wisdom_command(
        "capture Authorization: Bearer abcdefghijklmnopqrstuvwxyz",
        config=wisdom_config,
        db=wisdom_db,
    )
    assert "Capture blocked" in response
    assert wisdom_db.counts()["captures"] == 0


def test_unknown_subcommand_returns_help(wisdom_db, wisdom_config):
    assert "Wisdom commands" in handle_wisdom_command("wat", config=wisdom_config, db=wisdom_db)


def test_result_limits_enforced(wisdom_db, wisdom_config):
    limited = wisdom_config.__class__(
        enabled=wisdom_config.enabled,
        db_path=wisdom_config.db_path,
        capture_mode=wisdom_config.capture_mode,
        max_results=2,
        interpret_timeout_seconds=wisdom_config.interpret_timeout_seconds,
        interpretation_mode=wisdom_config.interpretation_mode,
    )
    for idx in range(4):
        handle_wisdom_command(f"capture note {idx}", config=limited, db=wisdom_db)
    response = handle_wisdom_command("inbox", config=limited, db=wisdom_db)
    assert response.count("#") == 2
