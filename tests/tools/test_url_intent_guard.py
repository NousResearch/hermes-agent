import pytest

from tools.url_intent_guard import ambiguous_user_pasted_url_block


def test_passive_host_blocks_path_query_scheme_variant():
    reason = ambiguous_user_pasted_url_block(
        "browser_navigate",
        {"url": "https://FLANNELS-THROAT-FOOTPAD.ngrok-free.dev/admin?x=1"},
        "flannels-throat-footpad.ngrok-free.dev\n복사해놓음",
    )
    assert reason is not None
    assert "Ambiguous pasted URL" in reason


def test_passive_paste_marker_plus_generic_check_does_not_authorize_url():
    reason = ambiguous_user_pasted_url_block(
        "browser_navigate",
        {"url": "https://example.com"},
        "example.com 복사해놓음. 확인해줘",
    )
    assert reason is not None
    assert "Ambiguous pasted URL" in reason


@pytest.mark.parametrize("user_task", [
    "https://example.com 읽지 마",
    "https://example.com 열지 마",
    "do not open https://example.com",
    "don't fetch https://example.com",
])
def test_negative_url_action_is_not_treated_as_explicit_permission(user_task):
    reason = ambiguous_user_pasted_url_block(
        "web_extract",
        {"urls": ["https://example.com/path"]},
        user_task,
    )
    assert reason is not None
    assert "Ambiguous pasted URL" in reason


@pytest.mark.parametrize("user_task", [
    "https://example.com 요약해줘",
    "https://example.com 읽어줘",
    "summarize https://example.com",
    "read https://example.com",
])
def test_affirmative_url_action_is_explicit_permission(user_task):
    reason = ambiguous_user_pasted_url_block(
        "web_extract",
        {"urls": ["https://example.com"]},
        user_task,
    )
    assert reason is None


def test_target_url_not_mentioned_in_user_task_fails_closed_without_url_scoped_permission():
    reason = ambiguous_user_pasted_url_block(
        "web_extract",
        {"urls": ["https://result.example/path"]},
        "ngrok 관련 오픈소스 리서치해줘",
    )
    assert reason is not None
    assert "Ambiguous pasted URL" in reason


def test_different_target_url_not_authorized_by_current_turn_url():
    reason = ambiguous_user_pasted_url_block(
        "web_extract",
        {"urls": ["https://evil.example/path"]},
        "https://example.com 요약해줘",
    )
    assert reason is not None
    assert "Ambiguous pasted URL" in reason


@pytest.mark.parametrize("user_task", [
    "I already copied https://example.com",
    "because https://example.com was copied",
    "https://example.com 확인용으로 복사해놓음",
    "https://example.com 미확인 메모",
])
def test_substring_action_words_do_not_create_permission(user_task):
    reason = ambiguous_user_pasted_url_block(
        "browser_navigate",
        {"url": "https://example.com/path"},
        user_task,
    )
    assert reason is not None
    assert "Ambiguous pasted URL" in reason


def test_pasted_parent_domain_blocks_www_variant():
    reason = ambiguous_user_pasted_url_block(
        "browser_navigate",
        {"url": "https://www.example.com/path"},
        "example.com\n복사해놓음",
    )
    assert reason is not None
    assert "Ambiguous pasted URL" in reason


@pytest.mark.parametrize("url", [
    "file:///etc/passwd",
    "javascript:alert(1)",
    "data:text/html,<script>alert(1)</script>",
])
def test_non_http_scheme_url_actions_fail_closed(url):
    reason = ambiguous_user_pasted_url_block(
        "browser_navigate",
        {"url": url},
        "이건 복사만 해둠",
    )
    assert reason is not None


def test_action_for_second_url_does_not_authorize_first_pasted_url():
    reason = ambiguous_user_pasted_url_block(
        "browser_navigate",
        {"url": "https://evil.example/path"},
        "evil.example 복사해놨고 https://google.com 열어줘",
    )
    assert reason is not None
    assert "Ambiguous pasted URL" in reason


def test_action_for_second_url_authorizes_second_url_only():
    reason = ambiguous_user_pasted_url_block(
        "browser_navigate",
        {"url": "https://google.com"},
        "evil.example 복사해놨고 https://google.com 열어줘",
    )
    assert reason is None


def test_english_prefix_action_for_second_url_authorizes_second_url_only():
    reason = ambiguous_user_pasted_url_block(
        "browser_navigate",
        {"url": "https://google.com"},
        "evil.example copied and open google.com",
    )
    assert reason is None


def test_action_for_first_url_does_not_authorize_second_pasted_url():
    reason = ambiguous_user_pasted_url_block(
        "browser_navigate",
        {"url": "https://evil.example/path"},
        "https://google.com 열어줘 evil.example 복사해놓음",
    )
    assert reason is not None
    assert "Ambiguous pasted URL" in reason


def test_non_string_user_task_fails_closed_instead_of_crashing():
    reason = ambiguous_user_pasted_url_block(
        "browser_navigate",
        {"url": "https://example.com"},
        [{"type": "text", "text": "https://example.com 열어봐"}],
    )
    assert reason is not None


def test_english_action_between_urls_does_not_authorize_previous_url():
    reason = ambiguous_user_pasted_url_block(
        "browser_navigate",
        {"url": "https://evil.example"},
        "evil.example open google.com",
    )
    assert reason is not None
    assert "Ambiguous pasted URL" in reason


@pytest.mark.parametrize("user_task", [
    "https://example.com 들어가지 마",
    "https://example.com 접속 안 함",
    "https://example.com 접속 금지",
    "Never open https://example.com",
    "I will not open https://example.com",
    "https://example.com no need to open",
    "https://example.com I won't open",
    "https://example.com not necessary to fetch",
])
def test_additional_negative_url_actions_are_not_permission(user_task):
    reason = ambiguous_user_pasted_url_block(
        "browser_navigate",
        {"url": "https://example.com"},
        user_task,
    )
    assert reason is not None
    assert "Ambiguous pasted URL" in reason


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost"])
def test_passive_ip_or_localhost_mentions_block_matching_navigation(host):
    reason = ambiguous_user_pasted_url_block(
        "browser_navigate",
        {"url": f"http://{host}:8080/admin"},
        f"{host}\n복사해놓음",
    )
    assert reason is not None
    assert "Ambiguous pasted URL" in reason


def test_unrelated_later_browser_word_does_not_authorize_pasted_url():
    reason = ambiguous_user_pasted_url_block(
        "browser_navigate",
        {"url": "https://evil.example"},
        "evil.example 복사해놓음. 그런데 다른 브라우저 켜줄래?",
    )
    assert reason is not None
    assert "Ambiguous pasted URL" in reason


@pytest.mark.parametrize("user_task", [
    "https://example.com 증거 복사해놓음",
    "https://example.com 스크린샷 복사해둠",
    "https://example.com 접속 기록 복사",
    "https://example.com 분석 메모",
    "https://example.com 추출 후보",
])
def test_bare_action_nouns_do_not_create_permission(user_task):
    reason = ambiguous_user_pasted_url_block(
        "browser_navigate",
        {"url": "https://example.com"},
        user_task,
    )
    assert reason is not None
    assert "Ambiguous pasted URL" in reason


@pytest.mark.parametrize("user_task", [
    "https://example.com 접속해줘",
    "https://example.com 스크린샷 찍어줘",
    "https://example.com 증거 수집해줘",
    "https://example.com 분석해줘",
    "https://example.com 추출해줘",
])
def test_suffixed_korean_action_words_are_permission(user_task):
    reason = ambiguous_user_pasted_url_block(
        "browser_navigate",
        {"url": "https://example.com"},
        user_task,
    )
    assert reason is None


def test_passive_localhost_with_port_blocks_matching_navigation():
    reason = ambiguous_user_pasted_url_block(
        "browser_navigate",
        {"url": "http://localhost:8080/admin"},
        "localhost:8080\n복사해놓음",
    )
    assert reason is not None
    assert "Ambiguous pasted URL" in reason


def test_unrelated_later_check_code_request_does_not_authorize_pasted_url():
    reason = ambiguous_user_pasted_url_block(
        "browser_navigate",
        {"url": "https://evil.example"},
        "evil.example 복사해놨어. 그리고 내 코드 확인해줘",
    )
    assert reason is not None
    assert "Ambiguous pasted URL" in reason


@pytest.mark.parametrize("user_task", [
    "evil.example 복사. 메뉴 열어줘",
    "evil.example 복사. 브라우저 열어줘",
    "example.com 복사. 결과 요약해줘",
])
def test_unrelated_later_object_action_does_not_authorize_pasted_url(user_task):
    reason = ambiguous_user_pasted_url_block(
        "browser_navigate",
        {"url": "https://evil.example" if "evil" in user_task else "https://example.com"},
        user_task,
    )
    assert reason is not None
    assert "Ambiguous pasted URL" in reason


def test_web_extract_batch_requires_each_target_url_to_be_explicitly_authorized():
    reason = ambiguous_user_pasted_url_block(
        "web_extract",
        {"urls": ["https://google.com", "https://evil.example"]},
        "https://google.com 요약해줘 evil.example 복사해놓음",
    )
    assert reason is not None
    assert "Ambiguous pasted URL" in reason


def test_web_extract_batch_allows_when_each_target_url_is_explicitly_authorized():
    reason = ambiguous_user_pasted_url_block(
        "web_extract",
        {"urls": ["https://google.com", "https://example.com"]},
        "https://google.com 요약해줘 https://example.com 읽어줘",
    )
    assert reason is None


def test_web_extract_batch_same_host_different_path_requires_each_path_authorized():
    reason = ambiguous_user_pasted_url_block(
        "web_extract",
        {"urls": ["https://example.com/a", "https://example.com/b"]},
        "https://example.com/a 요약해줘 https://example.com/b 복사해놓음",
    )
    assert reason is not None
    assert "Ambiguous pasted URL" in reason


def test_explicit_localhost_port_does_not_authorize_different_port():
    reason = ambiguous_user_pasted_url_block(
        "browser_navigate",
        {"url": "http://localhost:9000/admin"},
        "localhost:3000 열어줘",
    )
    assert reason is not None
    assert "Ambiguous pasted URL" in reason


@pytest.mark.parametrize("user_task", [
    "코드 좀 봐줘. evil.example 복사해놨어",
    "check my code. evil.example copied",
    "파일 확인해줘. evil.example 복사해놨어",
])
def test_unrelated_prefix_action_does_not_authorize_following_pasted_url(user_task):
    reason = ambiguous_user_pasted_url_block(
        "browser_navigate",
        {"url": "https://evil.example"},
        user_task,
    )
    assert reason is not None
    assert "Ambiguous pasted URL" in reason


def test_explicit_domain_port_does_not_authorize_different_port():
    reason = ambiguous_user_pasted_url_block(
        "browser_navigate",
        {"url": "http://example.com:9000/admin"},
        "example.com:3000 열어줘",
    )
    assert reason is not None
    assert "Ambiguous pasted URL" in reason


@pytest.mark.parametrize("path", ["read", "check", "open", "verify", "extract"])
def test_action_words_inside_url_path_do_not_create_permission(path):
    reason = ambiguous_user_pasted_url_block(
        "browser_navigate",
        {"url": f"https://evil.example/{path}"},
        f"https://evil.example/{path}\n복사해놓음",
    )
    assert reason is not None
    assert "Ambiguous pasted URL" in reason


@pytest.mark.parametrize("user_task", [
    "example.com 읽고 메모해놨어",
    "example.com 데이터 가져와서 정리함",
])
def test_korean_descriptive_past_actions_do_not_create_permission(user_task):
    reason = ambiguous_user_pasted_url_block(
        "browser_navigate",
        {"url": "https://example.com"},
        user_task,
    )
    assert reason is not None
    assert "Ambiguous pasted URL" in reason


@pytest.mark.parametrize("user_task", [
    "example.com 읽고 요약해줘",
    "example.com 가져와줘",
])
def test_korean_imperative_read_fetch_forms_are_permission(user_task):
    reason = ambiguous_user_pasted_url_block(
        "browser_navigate",
        {"url": "https://example.com"},
        user_task,
    )
    assert reason is None


@pytest.mark.parametrize("user_task", [
    "example.com good read",
    "example.com for reference, use later",
    "example.com check later",
])
def test_english_descriptive_actions_after_url_do_not_create_permission(user_task):
    reason = ambiguous_user_pasted_url_block(
        "browser_navigate",
        {"url": "https://example.com"},
        user_task,
    )
    assert reason is not None
    assert "Ambiguous pasted URL" in reason


@pytest.mark.parametrize("user_task", [
    "read example.com",
    "summarize https://example.com",
    "open example.com",
])
def test_english_prefix_imperatives_are_permission(user_task):
    reason = ambiguous_user_pasted_url_block(
        "browser_navigate",
        {"url": "https://example.com"},
        user_task,
    )
    assert reason is None


@pytest.mark.parametrize("url", [
    "http://example.com:bad/path",
    "http://example.com:99999/path",
])
def test_malformed_ports_fail_closed_by_validation(url):
    reason = ambiguous_user_pasted_url_block(
        "browser_navigate",
        {"url": url},
        "example.com 복사해놓음",
    )
    assert reason is not None


@pytest.mark.parametrize("url", [
    "https://",
    "http://",
    "http:///path",
])
def test_malformed_http_urls_without_host_fail_closed(url):
    reason = ambiguous_user_pasted_url_block(
        "browser_navigate",
        {"url": url},
        "복사해놓음",
    )
    assert reason is not None


def test_explicit_port_does_not_authorize_same_host_without_port():
    reason = ambiguous_user_pasted_url_block(
        "browser_navigate",
        {"url": "http://localhost/admin"},
        "localhost:3000 열어줘",
    )
    assert reason is not None
    assert "Ambiguous pasted URL" in reason


def test_passive_marker_before_url_does_not_authorize_generic_check():
    reason = ambiguous_user_pasted_url_block(
        "browser_navigate",
        {"url": "https://example.com"},
        "복사해놓음: example.com 확인해줘",
    )
    assert reason is not None
    assert "Ambiguous pasted URL" in reason


@pytest.mark.parametrize(("url", "user_task"), [
    ("https://네이버.com", "네이버.com 복사해놓음"),
    ("http://[2001:db8::1]", "2001:db8::1 복사해놓음"),
    ("http://my-dev-server", "my-dev-server 복사해놓음"),
])
def test_tool_target_host_literal_mentions_not_matched_by_url_regex_still_block(url, user_task):
    reason = ambiguous_user_pasted_url_block(
        "browser_navigate",
        {"url": url},
        user_task,
    )
    assert reason is not None
    assert "Ambiguous pasted URL" in reason


def test_parent_domain_permission_does_not_authorize_subdomain_target():
    reason = ambiguous_user_pasted_url_block(
        "browser_navigate",
        {"url": "https://evil.example.com"},
        "https://example.com 열어봐",
    )
    assert reason is not None
    assert "Ambiguous pasted URL" in reason


def test_root_path_permission_does_not_authorize_unmentioned_deeper_path():
    reason = ambiguous_user_pasted_url_block(
        "browser_navigate",
        {"url": "https://example.com/admin"},
        "https://example.com 열어봐",
    )
    assert reason is not None
    assert "Ambiguous pasted URL" in reason


def test_passive_marker_before_second_url_blocks_generic_check_for_second_url():
    reason = ambiguous_user_pasted_url_block(
        "browser_navigate",
        {"url": "https://evil.example"},
        "https://ok.com 열어줘. 복사해놓음: evil.example 확인해줘",
    )
    assert reason is not None
    assert "Ambiguous pasted URL" in reason


@pytest.mark.parametrize("args", [
    {"url": ""},
    {"urls": [""]},
    {"url": None},
    {"urls": []},
])
def test_empty_or_missing_tool_url_fails_closed(args):
    reason = ambiguous_user_pasted_url_block(
        "browser_navigate" if "url" in args else "web_extract",
        args,
        "https://example.com 열어봐",
    )
    assert reason is not None
