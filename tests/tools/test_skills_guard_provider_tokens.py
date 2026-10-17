"""Provider-format credential detection in the skill-install security scanner.

Synthetic values are assembled from fragments to avoid committing usable or
push-protection-triggering token-shaped literals.
"""

import pytest

from tools.skills_guard import scan_file, scan_skill, should_allow_install


# The formats intentionally require plausible *complete* shapes, not short
# documentation examples such as "xoxb-YOUR-TOKEN" or "npm_...".
SAMPLES = (
    ("slack_bot_token_leaked",
     "xoxb-" + "123456789012-" + "123456789012-" + "Ab3" * 9),
    ("npm_token_leaked", "npm_" + "Ab3" * 12),
    ("google_api_key_leaked", "AIza" + "Ab3" * 11 + "q2"),
    ("stripe_live_key_leaked", "sk_live_" + "Ab3" * 10),
)


@pytest.mark.parametrize(("pattern_id", "token"), SAMPLES)
def test_skill_scanner_detects_unlabeled_provider_token(tmp_path, pattern_id, token):
    path = tmp_path / "sample.md"
    path.write_text("Authentication value: " + token + "\n", encoding="utf-8")
    findings = scan_file(path)
    hits = [f for f in findings if f.pattern_id == pattern_id]
    assert len(hits) == 1
    assert hits[0].severity == "critical"
    assert hits[0].category == "credential_exposure"


@pytest.mark.parametrize(("pattern_id", "token"), SAMPLES)
def test_community_install_blocks_credential_shaped_skill(tmp_path, pattern_id, token):
    skill = tmp_path / "sample-skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text(
        "---\nname: sample-skill\ndescription: sample\n---\n"
        "Login value: " + token + "\n", encoding="utf-8",
    )
    result = scan_skill(skill, source="community")
    assert any(f.pattern_id == pattern_id for f in result.findings)
    assert result.verdict == "dangerous"
    allowed, _message = should_allow_install(result)
    assert allowed is False


@pytest.mark.parametrize("example", [
    "Set SLACK_BOT_TOKEN before using this skill",
    "Use xoxb-YOUR-EXAMPLE in the setup guide",
    "npm_... is the access-token prefix",
    "npm_" + "Ab3" * 11,  # short token, not the vendor format
    "NPM_" + "Ab3" * 12,  # wrong case
    "AIza" + "Ab3" * 11 + "q2more",  # longer than one key
    "aiza" + "Ab3" * 11 + "q2",  # wrong case
    "sk_test_" + "Ab3" * 10,  # a Stripe test key, not a production secret
    "SK_LIVE_" + "Ab3" * 10,  # wrong case
])
def test_non_secret_examples_do_not_trigger_added_provider_detectors(tmp_path, example):
    path = tmp_path / "examples.md"
    path.write_text(example + "\n", encoding="utf-8")
    new_ids = {item[0] for item in SAMPLES}
    findings = scan_file(path)
    assert not (new_ids & {f.pattern_id for f in findings})
