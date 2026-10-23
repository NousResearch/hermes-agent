"""Non-ASCII query reachability in the deferred-tool catalog.

The catalog is indexed from English names/descriptions. Before the Unicode
tokenizer fix, ``_TOKEN_RE = [A-Za-z0-9]+`` shredded "tạo ảnh" into
``['t','o','nh']``: every diacritic split a word, so a Vietnamese query scored
0 against every document and the rarest-token gate rejected it outright. The
tool was unreachable, not merely low-ranked — the model could never discover
``image_generate`` from a Vietnamese prompt.

Two mechanisms are covered here:
  1. the tokenizer keeps non-ASCII word characters intact, and
  2. ``tools.tool_search.aliases`` injects the user's own vocabulary into the
     indexed document so those tokens have something to match.
"""

import json

import pytest

from tools.tool_search_catalog import (
    CatalogEntry,
    _entry_search_text,
    _search_aliases,
    _tokenize,
    build_catalog,
    search_catalog,
)


@pytest.fixture(autouse=True)
def _clear_alias_cache():
    """``_search_aliases`` is lru_cached; each test supplies its own env."""
    _search_aliases.cache_clear()
    yield
    _search_aliases.cache_clear()


def _fn(name, description, params=("prompt",)):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {p: {"type": "string"} for p in params},
            },
        },
    }


# A corpus small enough to reason about but large enough for BM25 IDF to behave.
CORPUS = [
    _fn("image_generate", "Generate images from text prompts. The active model's "
                          "edit/reference capabilities are rendered at serving time."),
    _fn("web_search", "Search the web for information.", ("query",)),
    _fn("web_extract", "Extract content from web page URLs.", ("urls",)),
    _fn("read_file", "Read a text file with line numbers and pagination.", ("path",)),
    _fn("write_file", "Write content to a file, completely replacing existing content.",
        ("path", "content")),
    _fn("search_files", "Search file contents or find files by name.", ("pattern", "path")),
    _fn("terminal", "Execute shell commands.", ("command",)),
    _fn("vision_analyze", "Load an image into the conversation so you can see it.",
        ("image_url",)),
    _fn("browser_exec", "Drive a real web browser.", ("code",)),
    _fn("session_search", "Recall past conversations and search session history.", ("query",)),
    _fn("text_to_speech", "Convert text to speech audio.", ("text",)),
]

VI_ALIASES = ("tạo ảnh sinh ảnh tạo hình ảnh vẽ tranh hình minh hoạ chỉnh sửa ảnh "
              "sửa ảnh ghép ảnh xoá vật thể xóa vật thể tách nền đổi nền phục chế ảnh")
COMMON_VI_ALIASES = {
    "web_search": "tìm kiếm web tìm trên mạng tra cứu internet tra cứu trực tuyến",
    "web_extract": "đọc trang web trích xuất trang web lấy nội dung đường link đọc liên kết",
    "read_file": "đọc file đọc tệp xem file xem tệp mở file văn bản",
    "write_file": "ghi file ghi tệp tạo file tạo tệp lưu nội dung vào file",
    "search_files": "tìm file tìm tệp tìm trong file tìm nội dung trong tệp tìm theo tên file",
    "terminal": "chạy lệnh shell chạy lệnh terminal thực thi lệnh dòng lệnh",
    "vision_analyze": "xem ảnh phân tích ảnh đọc ảnh nhận diện ảnh mô tả hình ảnh",
    "browser_exec": "mở trình duyệt duyệt web tự động thao tác trang web bấm trên trang web",
    "session_search": "tìm lịch sử trò chuyện tìm phiên cũ tìm cuộc hội thoại trước nhớ lại phiên chat",
    "text_to_speech": "đọc thành tiếng chuyển văn bản thành giọng nói tạo file âm thanh giọng đọc",
}


class TestUnicodeTokenizer:
    def test_vietnamese_words_survive_tokenization(self):
        """Diacritics must not split a word into single letters."""
        assert _tokenize("tạo ảnh") == ["tạo", "ảnh"]
        assert _tokenize("xoá vật thể") == ["xoá", "vật", "thể"]

    def test_english_stemming_still_applies(self):
        """The Unicode class must not cost us Snowball stemming."""
        assert _tokenize("generating images") == ["generat", "imag"]
        assert _tokenize("issues") == _tokenize("issue")

    def test_underscores_still_split(self):
        """``[^\\W_]+`` must keep treating ``_`` as a separator, not a word char."""
        assert _tokenize("image_generate") == ["imag", "generat"]

    def test_cjk_and_other_scripts_survive(self):
        """The fix is not Vietnamese-specific."""
        assert _tokenize("生成图片") == ["生成图片"]
        assert _tokenize("изображение") == ["изображение"]


class TestSearchAliases:
    def test_alias_text_lands_in_indexed_document(self, monkeypatch):
        monkeypatch.setenv("HERMES_TOOL_SEARCH_ALIASES",
                           json.dumps({"image_generate": VI_ALIASES}))
        _search_aliases.cache_clear()
        text = _entry_search_text(CORPUS[0]["function"] and CORPUS[0])
        assert "tạo ảnh" in text
        assert "xoá vật thể" in text

    def test_alias_accepts_list_form(self, monkeypatch):
        monkeypatch.setenv("HERMES_TOOL_SEARCH_ALIASES",
                           json.dumps({"image_generate": ["tạo ảnh", "ghép ảnh"]}))
        _search_aliases.cache_clear()
        assert _search_aliases()["image_generate"] == "tạo ảnh ghép ảnh"

    def test_malformed_alias_config_is_ignored(self, monkeypatch):
        """A bad alias blob must never break tool search."""
        monkeypatch.setenv("HERMES_TOOL_SEARCH_ALIASES", "{not json")
        _search_aliases.cache_clear()
        assert isinstance(_search_aliases(), dict)

    def test_no_aliases_leaves_document_unchanged(self, monkeypatch):
        monkeypatch.setenv("HERMES_TOOL_SEARCH_ALIASES", "{}")
        _search_aliases.cache_clear()
        assert "tạo ảnh" not in _entry_search_text(CORPUS[0])


class TestVietnameseQueriesReachImageGenerate:
    @pytest.fixture
    def catalog(self, monkeypatch):
        monkeypatch.setenv("HERMES_TOOL_SEARCH_ALIASES",
                           json.dumps({"image_generate": VI_ALIASES}))
        _search_aliases.cache_clear()
        return build_catalog(CORPUS)

    @pytest.mark.parametrize("query", [
        "tạo ảnh",
        "sinh ảnh",
        "tạo hình ảnh",
        "chỉnh sửa ảnh",
        "sửa ảnh",
        "xoá vật thể",
        "xóa vật thể",
        "ghép ảnh",
        "vẽ tranh",
        "tách nền",
        "phục chế ảnh",
    ])
    def test_vietnamese_image_query_finds_tool(self, catalog, query):
        hits = [h.name for h in search_catalog(catalog, query, limit=4)]
        assert "image_generate" in hits, f"{query!r} did not reach image_generate: {hits}"

    @pytest.mark.parametrize("query", ["generate image", "edit image", "image generation"])
    def test_english_queries_still_work(self, catalog, query):
        hits = [h.name for h in search_catalog(catalog, query, limit=4)]
        assert "image_generate" in hits

    @pytest.mark.parametrize("query", [
        "tìm kiếm web", "đọc file", "chạy lệnh",
        "web search", "read file", "run shell command",
    ])
    def test_unrelated_queries_do_not_hit_image_generate(self, catalog, query):
        """Aliases must not turn image_generate into a catch-all."""
        hits = [h.name for h in search_catalog(catalog, query, limit=4)]
        assert "image_generate" not in hits, f"{query!r} wrongly matched: {hits}"

    def test_unrelated_english_queries_still_route_correctly(self, catalog):
        assert "web_search" in [h.name for h in search_catalog(catalog, "web search", limit=3)]
        assert "read_file" in [h.name for h in search_catalog(catalog, "read file", limit=3)]


class TestCommonVietnameseAliases:
    def test_vietnamese_aliases_route_common_tools(self, monkeypatch):
        monkeypatch.setenv(
            "HERMES_TOOL_SEARCH_ALIASES",
            json.dumps(COMMON_VI_ALIASES, ensure_ascii=False),
        )
        _search_aliases.cache_clear()
        catalog = build_catalog(CORPUS)
        cases = {
            "tra cứu trên mạng": "web_search",
            "đọc nội dung đường link": "web_extract",
            "xem tệp văn bản": "read_file",
            "ghi nội dung vào file": "write_file",
            "tìm nội dung trong tệp": "search_files",
            "thực thi lệnh shell": "terminal",
            "phân tích ảnh": "vision_analyze",
            "mở trình duyệt": "browser_exec",
            "tìm cuộc hội thoại trước": "session_search",
            "chuyển văn bản thành giọng nói": "text_to_speech",
        }
        for query, expected in cases.items():
            assert search_catalog(catalog, query, limit=1)[0].name == expected


class TestRegressionWithoutAliases:
    def test_vietnamese_misses_without_aliases(self, monkeypatch):
        """Documents the mechanism: the tokenizer alone is not sufficient — the
        indexed document must actually contain the user's vocabulary."""
        monkeypatch.setenv("HERMES_TOOL_SEARCH_ALIASES", "{}")
        _search_aliases.cache_clear()
        catalog = build_catalog(CORPUS)
        hits = [h.name for h in search_catalog(catalog, "tạo ảnh", limit=4)]
        assert "image_generate" not in hits
