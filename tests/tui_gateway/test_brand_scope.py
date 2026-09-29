"""Brand app-id prompt: a login pin scopes the chat, a missing pin adds nothing."""

from tui_gateway.brand_scope import build_brand_scope_prompt


def test_brand_pin_names_only_the_allowed_app_id():
    text = build_brand_scope_prompt("foundrly", ["foundrly", "toba"], "ada@foundrly.com", False)
    assert 'app id "foundrly"' in text
    assert "foundrly, toba" in text
    assert "ada@foundrly.com" in text
    assert "Do not invent a send." in text
    assert "ivx-foundrly-firecrawl" in text
    assert "name is exactly firecrawl" in text
    assert "count only the tools registered" in text
    assert "firecrawl_scrape, firecrawl_search, and firecrawl_parse" in text


def test_brand_pin_rejects_an_app_id_outside_the_allowlist():
    text = build_brand_scope_prompt("other", ["foundrly"], "ada@foundrly.com", False)
    assert 'app id "foundrly"' in text
    assert 'app id "other"' not in text


def test_super_admin_without_a_brand_stays_unscoped():
    text = build_brand_scope_prompt("", [], "root@intelli-verse-x.ai", True)
    assert "super admin (all brands)" in text


def test_inbox_account_id_is_named_for_chatwoot_calls():
    text = build_brand_scope_prompt("foundrly", ["foundrly"], "ada@foundrly.com", False, "233")
    assert "Inbox Studio account id for this brand is 233." in text
    assert "Pass accountId 233 on every chatwoot tool call." in text


def test_missing_login_adds_no_brand_block():
    assert build_brand_scope_prompt("", [], "", False) == ""
    assert build_brand_scope_prompt("Not A Brand!", ["../etc"], "", False) == ""
