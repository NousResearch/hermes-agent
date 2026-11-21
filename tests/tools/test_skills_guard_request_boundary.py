"""HTTP-call exfiltration rules read the call's arguments, not prose after the call."""
import pytest

from tools.skills_guard import scan_skill, should_allow_install

CALLS = [("requests.get", "env_exfil_requests"), ("requests.post", "env_exfil_requests"),
         ("httpx.get", "env_exfil_httpx"), ("http.post", "env_exfil_httpx")]


def _scan(tmp_path, text):
    skill = tmp_path / "example"
    skill.mkdir()
    (skill / "SKILL.md").write_text(text, encoding="utf-8")
    return scan_skill(skill, source="community")


@pytest.mark.parametrize("call,rule", CALLS)
@pytest.mark.parametrize("suffix", [
    " with no timeout keyword",
    "; API_KEY = value",
    ' (the "timeout keyword")',
    "  # retries on a token bucket",
])
def test_prose_after_a_plain_call_is_not_a_credential(tmp_path, call, rule, suffix):
    result = _scan(tmp_path, f"{call}(url){suffix}")
    assert not any(f.pattern_id == rule for f in result.findings)
    assert result.verdict == "safe"
    assert should_allow_install(result)[0] is True


@pytest.mark.parametrize("call,rule", CALLS)
@pytest.mark.parametrize("argument", [
    "prefix_KEY_suffix", "tokenValue", "keyword",
    "wrap(API_KEY)", 'os.environ["API_KEY"]',
    'f") {API_KEY}"', '{"Authorization": TOKEN}',
])
def test_credential_arguments_stay_hard_blocked(tmp_path, call, rule, argument):
    result = _scan(tmp_path, f"{call}(url, data={argument})")
    assert any(f.pattern_id == rule and f.severity == "critical" for f in result.findings)
    assert result.verdict == "dangerous"
    assert should_allow_install(result, force=True)[0] is False
