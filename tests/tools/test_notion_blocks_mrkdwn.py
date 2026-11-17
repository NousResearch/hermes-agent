"""render_blocks_mrkdwn — a Release Assets row's body as Slack mrkdwn, safe by
construction (escaped text, links only for safe schemes, unknown blocks
reported rather than passed through)."""
from tools.notion_blocks_html import render_blocks_mrkdwn


def rt(text, href=None, **ann):
    return {"type": "text", "plain_text": text, "href": href, "annotations": ann}


def blk(kind, *rich, children=None, **extra):
    body = {"rich_text": list(rich)}
    body.update(extra)
    b = {"type": kind, kind: body}
    if children is not None:
        b["children"] = children
    return b


def test_paragraphs_and_formatting():
    out = render_blocks_mrkdwn([
        blk("heading_2", rt("New: points")),
        blk("paragraph", rt("Earn "), rt("bonus ", bold=True), rt("points", italic=True),
            rt(" now", strikethrough=True), rt(" x", code=True)),
    ])
    assert out.text == "*New: points*\nEarn *bonus* _points_ ~now~ `x`"
    assert out.unsupported == [] and out.images == []


def test_escapes_and_neutralises_mentions():
    out = render_blocks_mrkdwn([blk("paragraph", rt("<!channel> & <@U123> <https://evil|x>"))])
    assert out.text == "&lt;!channel&gt; &amp; &lt;@U123&gt; &lt;https://evil|x&gt;"


def test_links_only_safe_schemes():
    out = render_blocks_mrkdwn([blk("paragraph", rt("docs", href="https://help.example.com/a?b=1&c=2"),
                                    rt(" bad", href="javascript:alert(1)"),
                                    rt(" pipe", href="https://x.com/a|b"))])
    assert out.text == "<https://help.example.com/a?b=1&amp;c=2|docs> bad pipe"


def test_lists_nesting_numbering_todo():
    out = render_blocks_mrkdwn([
        blk("bulleted_list_item", rt("one"), children=[blk("bulleted_list_item", rt("one.a"))]),
        blk("bulleted_list_item", rt("two")),
        blk("numbered_list_item", rt("first")),
        blk("numbered_list_item", rt("second")),
        blk("to_do", rt("done"), checked=True),
    ])
    assert out.text.split("\n") == ["• one", "    • one.a", "• two", "1. first", "2. second", "☑ done"]


def test_quote_code_divider():
    out = render_blocks_mrkdwn([
        blk("quote", rt("note")),
        {"type": "code", "code": {"rich_text": [rt("a < b")]}},
        {"type": "divider", "divider": {}},
    ])
    assert out.text == "> note\n```\na &lt; b\n```\n───"


def test_images_and_unsupported_reported():
    out = render_blocks_mrkdwn([
        {"type": "image", "image": {"caption": [rt("Settings page")]}},
        {"type": "table", "table": {}},
        {"type": "embed", "embed": {}},
    ])
    assert out.images == ["Settings page"]
    assert out.unsupported == ["table", "embed"]
