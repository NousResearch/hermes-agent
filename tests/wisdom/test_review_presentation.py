from hermes_wisdom.review_presentation import (
    aggregate_review_text,
    full_review_text,
    review_status_text,
    review_check_line,
)
import pytest


@pytest.mark.parametrize("status,expected", [
    ("pass", "✅ Security check"),
    ("advisory", "⚠️ Security check: Advisory"),
    ("blocked", "❌ Security check: Blocked"),
    ("pending", "⏳ Security check: Pending"),
    ("unavailable", "➖ Security check: Unavailable"),
])
def test_card_status_leads_with_icon_and_keeps_nonpass_meaning(status, expected):
    assert review_check_line("Security check", status) == expected


def test_expanded_card_rows_keep_summary_and_findings_without_pass_labels():
    text = full_review_text({
        "status": "advisory", "summary": "Review a policy match.",
        "checks": [{"label": "Private keys", "status": "pass"},
                   {"label": "Organization policy", "status": "advisory",
                    "finding_count": 1, "details": ["Policy match details"]}],
    }, {"status": "pass"}, status_first=True)
    assert "✅ Private keys" in text
    assert "⚠️ Organization policy: Advisory (1 finding)" in text
    assert "Review a policy match." in text
    assert "Policy match details" in text
    assert "Pass" not in text


def test_review_status_text_covers_every_presented_state():
    assert review_status_text("pass") == "✅ Pass"
    assert review_status_text("advisory") == "⚠️ Advisory"
    assert review_status_text("blocked") == "❌ Blocked"
    assert review_status_text("pending") == "⏳ Pending"
    assert review_status_text("running") == "⏳ Pending"
    assert review_status_text("retry") == "⏳ Pending"
    assert review_status_text("unavailable") == "➖ Unavailable"
    assert review_status_text("future-status") == "➖ Unavailable"


def test_full_review_text_renders_each_check_as_its_own_labeled_row():
    rendered = full_review_text(
        {
            "status": "advisory",
            "summary": "One review item needs attention.",
            "checks": [
                {
                    "key": "private_keys",
                    "label": "Private keys",
                    "status": "pass",
                    "finding_count": 0,
                    "details": [],
                },
                {
                    "key": "organization_policy",
                    "label": "Organization policy",
                    "status": "advisory",
                    "finding_count": 1,
                    "details": ["Review a bounded policy match."],
                },
            ],
        },
        {
            "status": "blocked",
            "checks": [
                {
                    "key": "profanity_or_abuse",
                    "status": "blocked",
                    "finding_count": 1,
                    "details": ["Potential abusive language."],
                }
            ],
        },
    )

    assert rendered.splitlines() == [
        "Security check: ⚠️ Advisory",
        "One review item needs attention.",
        "Private keys: ✅ Pass",
        "Organization policy: ⚠️ Advisory (1 finding)",
        "  Review a bounded policy match.",
        "No known matches detected is not a security certification.",
        "",
        "Professionalism check (agent-assessed, advisory): ❌ Blocked",
        "Profanity or abusive language: ❌ Blocked (1 finding)",
        "  Potential abusive language.",
    ]


def test_public_review_projection_remains_aggregate_only():
    rendered = aggregate_review_text(
        {"status": "pass", "checks": [{"label": "Private keys"}]},
        {"status": "unavailable", "checks": [{"label": "Hate or harassment"}]},
    )

    assert rendered == "Security: ✅ Pass · Professionalism: ➖ Unavailable"
    assert "Private keys" not in rendered
    assert "Hate or harassment" not in rendered
