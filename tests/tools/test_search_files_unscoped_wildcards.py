"""#76628: reject pure-splat search_files only at broad/default path scope."""

from tools.file_tools import _is_broad_search_path, _is_unscoped_search_pattern, search_tool


def test_broad_path_detection():
    assert _is_broad_search_path(None) is True
    assert _is_broad_search_path("") is True
    assert _is_broad_search_path(".") is True
    assert _is_broad_search_path("./") is True
    assert _is_broad_search_path("./src") is False
    assert _is_broad_search_path("small-dir") is False
    assert _is_broad_search_path("/tmp/job") is False


def test_broad_path_detection_canonicalizes_cwd_and_parent_spellings(tmp_path, monkeypatch):
    """Raw spelling is not enough: ./child/.. and .. must classify as broad."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "child").mkdir()
    (tmp_path / "child" / "nested").mkdir()

    assert _is_broad_search_path("./child/..") is True
    assert _is_broad_search_path("child/../.") is True
    assert _is_broad_search_path("..") is True
    # Explicit subdir remains narrow even when spelled relatively.
    assert _is_broad_search_path("./child") is False
    assert _is_broad_search_path("child/nested") is False


def test_files_target_pure_star_blocked_only_at_broad_path():
    assert _is_unscoped_search_pattern("*", "files", path=".") is True
    assert _is_unscoped_search_pattern("**", "files", path=".") is True
    assert _is_unscoped_search_pattern("*", "files", path="./small-dir") is False
    assert _is_unscoped_search_pattern("*", "files", path="/tmp/job") is False
    assert _is_unscoped_search_pattern("*.py", "files", path=".") is False
    assert _is_unscoped_search_pattern("*config*", "files", path=".") is False


def test_files_target_pure_star_blocks_cwd_equivalent_and_parent_paths(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "child").mkdir()

    assert _is_unscoped_search_pattern("*", "files", path="./child/..") is True
    assert _is_unscoped_search_pattern("*", "files", path="..") is True
    assert _is_unscoped_search_pattern("*", "files", path="./child") is False


def test_content_target_match_all_regex_blocked_only_at_broad_path():
    assert _is_unscoped_search_pattern(".*", "content", path=".") is True
    assert _is_unscoped_search_pattern(".+", "content", path=".") is True
    assert _is_unscoped_search_pattern(".*", "content", path="./pkg") is False
    assert _is_unscoped_search_pattern("def ", "content", path=".") is False
    assert _is_unscoped_search_pattern("foo.*bar", "content", path=".") is False


def test_search_tool_blocks_pure_star_at_default_path():
    result = search_tool(pattern="*", target="files", path=".")
    assert "BLOCKED" in result
    assert "broad/default" in result or "unscoped wildcard" in result


def test_search_tool_blocks_cwd_equivalent_path_spelling(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "child").mkdir()
    result = search_tool(pattern="*", target="files", path="./child/..")
    assert "BLOCKED" in result


def test_search_tool_blocks_parent_path_spelling(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = search_tool(pattern="*", target="files", path="..")
    assert "BLOCKED" in result


def test_search_tool_allows_pure_star_in_narrow_directory(tmp_path):
    (tmp_path / "a.txt").write_text("x\n", encoding="utf-8")
    (tmp_path / "b.txt").write_text("y\n", encoding="utf-8")
    result = search_tool(pattern="*", target="files", path=str(tmp_path))
    assert "BLOCKED" not in result


def test_search_tool_allows_narrow_glob_at_default_path(tmp_path, monkeypatch):
    # Use an explicit narrow path so we do not depend on cwd contents.
    (tmp_path / "app.py").write_text("x\n", encoding="utf-8")
    result = search_tool(pattern="*.py", target="files", path=str(tmp_path))
    assert "BLOCKED" not in result
