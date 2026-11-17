"""notion_blocks_html — Notion block tree → Zendesk article HTML. Safe by
construction: text is escaped, only allowlisted tags are emitted, links need
an allowlisted scheme, and anything unknown is reported, not passed through."""
from tools.notion_blocks_html import render_blocks


def rt(text, href=None, **ann):
    a = {"bold": False, "italic": False, "strikethrough": False, "underline": False,
         "code": False, "color": "default"}
    a.update(ann)
    return {"type": "text", "plain_text": text, "href": href, "annotations": a}


def blk(kind, rich=None, children=None, **extra):
    body = {"rich_text": rich or []}
    body.update(extra)
    b = {"type": kind, kind: body, "has_children": bool(children)}
    if children:
        b["children"] = children
    return b


def test_paragraphs_headings_and_annotations():
    out = render_blocks([
        blk("heading_2", [rt("Overview")]),
        blk("paragraph", [rt("Use "), rt("points", bold=True), rt(" and "),
                          rt("code", code=True), rt(" or "), rt("it", italic=True)]),
        blk("paragraph", []),  # empty paragraphs are dropped
        blk("heading_3", [rt("Steps")]),
    ])
    assert out.html == ("<h2>Overview</h2><p>Use <strong>points</strong> and "
                        "<code>code</code> or <em>it</em></p><h3>Steps</h3>")
    assert out.images == [] and out.unsupported == []


def test_text_is_escaped_never_raw_html():
    out = render_blocks([
        blk("paragraph", [rt('<script>alert(1)</script><img src=x onerror=alert(1)>')]),
        blk("code", [rt("<iframe src=//evil>")], language="html"),
    ])
    assert "<script" not in out.html and "<iframe" not in out.html and "<img" not in out.html
    assert "&lt;script&gt;" in out.html
    assert out.html.endswith("<pre><code>&lt;iframe src=//evil&gt;</code></pre>")


def test_links_need_safe_scheme():
    out = render_blocks([blk("paragraph", [
        rt("ok", href="https://help.example.com/a?b=1&c=2"),
        rt(" bad", href="javascript:alert(1)"),
        rt(" quote", href='https://x.com/"onmouseover="alert(1)'),
        rt(" mail", href="mailto:support@example.com"),
    ])])
    assert '<a href="https://help.example.com/a?b=1&amp;c=2">ok</a>' in out.html
    assert "javascript" not in out.html and " bad" in out.html
    assert 'onmouseover="' not in out.html and "&quot;" in out.html
    assert '<a href="mailto:support@example.com"> mail</a>' in out.html


def test_lists_are_grouped_and_nested():
    out = render_blocks([
        blk("bulleted_list_item", [rt("a")]),
        blk("bulleted_list_item", [rt("b")], children=[blk("numbered_list_item", [rt("b1")])]),
        blk("numbered_list_item", [rt("one")]),
        blk("numbered_list_item", [rt("two")]),
        blk("paragraph", [rt("after")]),
    ])
    assert out.html == ("<ul><li>a</li><li>b<ol><li>b1</li></ol></li></ul>"
                        "<ol><li>one</li><li>two</li></ol><p>after</p>")


def test_quote_callout_divider_toggle_table_columns():
    table = {"type": "table", "has_children": True,
             "table": {"has_column_header": True, "table_width": 2},
             "children": [
                 {"type": "table_row", "table_row": {"cells": [[rt("Plan")], [rt("Limit")]]}},
                 {"type": "table_row", "table_row": {"cells": [[rt("Free")], [rt("<5>")]]}},
             ]}
    out = render_blocks([
        blk("quote", [rt("q")]),
        blk("callout", [rt("Note")], icon={"type": "emoji", "emoji": "💡"}),
        {"type": "divider", "divider": {}, "has_children": False},
        blk("toggle", [rt("More")], children=[blk("paragraph", [rt("inside")])]),
        table,
        {"type": "column_list", "column_list": {}, "has_children": True, "children": [
            {"type": "column", "column": {}, "has_children": True,
             "children": [blk("paragraph", [rt("col")])]}]},
    ])
    assert out.html == (
        "<blockquote>q</blockquote><blockquote>Note</blockquote><hr>"
        "<p><strong>More</strong></p><p>inside</p>"
        "<table><tbody><tr><th>Plan</th><th>Limit</th></tr>"
        "<tr><td>Free</td><td>&lt;5&gt;</td></tr></tbody></table><p>col</p>")


def test_images_are_reported_not_embedded():
    img = {"type": "image", "has_children": False, "image": {
        "type": "file", "file": {"url": "https://prod-files-secure.s3/x.png?X-Amz-Expires=3600"},
        "caption": [rt("The points screen")]}}
    out = render_blocks([blk("paragraph", [rt("see")]), img])
    assert out.html == "<p>see</p>"
    assert out.images == ["The points screen"]
    ext = {"type": "image", "has_children": False,
           "image": {"type": "external", "external": {"url": "https://x/y.png"}, "caption": []}}
    assert render_blocks([ext]).images == ["(image without caption)"]


def test_unsupported_blocks_are_reported():
    out = render_blocks([
        {"type": "embed", "embed": {"url": "https://x"}, "has_children": False},
        {"type": "child_database", "child_database": {"title": "x"}, "has_children": False},
        {"type": "synced_block", "synced_block": {}, "has_children": False},
        {"type": "table_of_contents", "table_of_contents": {}, "has_children": False},
        {"type": "mystery", "mystery": {}, "has_children": False},
    ])
    assert sorted(out.unsupported) == ["child_database", "embed", "mystery", "synced_block"]
    assert out.html == ""  # table_of_contents is navigation and is skipped


def test_bookmark_becomes_link_only_when_safe():
    out = render_blocks([
        {"type": "bookmark", "bookmark": {"url": "https://docs.example.com", "caption": []},
         "has_children": False},
        {"type": "bookmark", "bookmark": {"url": "javascript:x", "caption": []},
         "has_children": False},
    ])
    assert out.html == '<p><a href="https://docs.example.com">https://docs.example.com</a></p>'
    assert out.unsupported == ["bookmark"]


def test_newlines_become_breaks():
    out = render_blocks([blk("paragraph", [rt("line1\nline2")])])
    assert out.html == "<p>line1<br>line2</p>"
