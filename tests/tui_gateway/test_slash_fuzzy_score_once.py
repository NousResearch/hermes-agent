"""Fuzzy scoring: score each item once per keystroke; tokenization is memoized.

``fuzzy_rank_slash_items`` scored every catalog item twice (once to filter the
merge, once to build the scores dict) and re-tokenized the same name/meta strings
on every keystroke — ~2,600 scorings and ~5,300 tokenizations per keystroke over
an 852-skill catalog, for identical inputs.
"""

from tui_gateway.slash_fuzzy import fuzzy_rank_slash_items


def _catalog(n: int) -> list[dict]:
    return [{"text": f"/cmd{i:03d}", "meta": f"Description {i} about widgets"} for i in range(n)]


def _items(n: int) -> list[dict]:
    return [{"text": f"/prefix{i}", "meta": "prefix item"} for i in range(n)]


def test_each_item_scored_once_per_rank():
    import tui_gateway.slash_fuzzy as fz

    calls = []
    real = fz.score_slash_completion_item

    def counting(item, query):
        calls.append(1)
        return real(item, query)

    catalog, items = _catalog(100), _items(10)
    with __import__("unittest.mock", fromlist=["patch"]).patch.object(
        fz, "score_slash_completion_item", side_effect=counting
    ):
        fuzzy_rank_slash_items(items, catalog, "wid")

    assert len(calls) <= len(catalog) + len(items), (
        f"one rank scored {len(calls)} times for {len(catalog) + len(items)} items; "
        "each item must be scored exactly once"
    )


def test_tokenization_is_memoized_across_keystrokes():
    import tui_gateway.slash_fuzzy as fz

    catalog, items = _catalog(100), _items(10)
    fz._TOKEN_CACHE.clear()
    fz.fuzzy_rank_slash_items(list(items), catalog, "wid")
    size_first = len(fz._TOKEN_CACHE)
    fz.fuzzy_rank_slash_items(list(items), catalog, "wi")

    assert size_first > 0, "baseline sanity: a cold rank populates the token cache"
    assert len(fz._TOKEN_CACHE) == size_first, (
        "a second keystroke over the same items grew the token cache; "
        "identical inputs must hit the memo"
    )
    # And the memo answers with equal token lists for the same input.
    assert fz.tokenize_search_text("Description 7 About Widgets") == fz.tokenize_search_text("description 7 about widgets")
