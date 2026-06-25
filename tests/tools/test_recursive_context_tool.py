import json

from hermes_constants import reset_hermes_home_override, set_hermes_home_override


def _call(**kwargs):
    from tools.recursive_context_tool import recursive_context

    return json.loads(recursive_context(**kwargs))


def test_create_search_and_read_keeps_large_text_external(tmp_path):
    token = set_hermes_home_override(tmp_path)
    try:
        text = "\n".join(
            [
                "Alpha introduction about unrelated setup.",
                "Beta section: the launch plan depends on pricing evidence.",
                "Gamma middle filler that should not be returned by search.",
                "Delta conclusion: pricing evidence must cite source rows.",
            ]
        )

        created = _call(action="create", name="pricing memo", text=text, chunk_lines=2)

        assert created["success"] is True
        assert created["corpus"]["name"] == "pricing memo"
        assert created["corpus"]["line_count"] == 4
        assert created["corpus"]["chunk_count"] == 2
        assert "text" not in created["corpus"]

        found = _call(action="search", corpus_id=created["corpus"]["corpus_id"], query="pricing evidence", limit=5)

        assert found["success"] is True
        assert [m["line"] for m in found["matches"]] == [2, 4]
        assert all("pricing evidence" in m["snippet"].lower() for m in found["matches"])
        assert all("Gamma middle filler" not in m["snippet"] for m in found["matches"])

        window = _call(action="read", corpus_id=created["corpus"]["corpus_id"], start_line=2, line_count=2)

        assert window["success"] is True
        assert window["range"] == {"start_line": 2, "end_line": 3}
        assert window["lines"] == [
            {"line": 2, "text": "Beta section: the launch plan depends on pricing evidence."},
            {"line": 3, "text": "Gamma middle filler that should not be returned by search."},
        ]
    finally:
        reset_hermes_home_override(token)


def test_map_returns_delegation_tasks_over_bounded_chunks(tmp_path):
    token = set_hermes_home_override(tmp_path)
    try:
        text = "\n".join(f"Line {i}: evidence block {i}" for i in range(1, 8))
        created = _call(action="create", name="research corpus", text=text, chunk_lines=3)

        mapped = _call(
            action="map",
            corpus_id=created["corpus"]["corpus_id"],
            task="Extract claims and source-backed caveats",
            max_chunks=2,
        )

        assert mapped["success"] is True
        assert mapped["corpus_id"] == created["corpus"]["corpus_id"]
        assert len(mapped["tasks"]) == 2
        assert mapped["tasks"][0]["range"] == {"start_line": 1, "end_line": 3}
        assert mapped["tasks"][1]["range"] == {"start_line": 4, "end_line": 6}
        assert "Extract claims" in mapped["tasks"][0]["prompt"]
        assert "recursive_context" in mapped["tasks"][0]["prompt"]
    finally:
        reset_hermes_home_override(token)


def test_path_ingest_preserves_source_citations_in_read_and_search(tmp_path):
    token = set_hermes_home_override(tmp_path / "home")
    try:
        source_a = tmp_path / "alpha.md"
        source_b = tmp_path / "beta.md"
        source_a.write_text("A1 intro\nA2 fusion evidence\nA3 outro\n", encoding="utf-8")
        source_b.write_text("B1 intro\nB2 fusion evidence with caveat\n", encoding="utf-8")

        created = _call(action="create", name="multi-source", paths=[str(source_a), str(source_b)], chunk_lines=2)
        corpus_id = created["corpus"]["corpus_id"]

        window = _call(action="read", corpus_id=corpus_id, start_line=2, line_count=4)

        assert window["success"] is True
        assert window["lines"] == [
            {"line": 2, "text": "A2 fusion evidence", "source": str(source_a), "source_line": 2},
            {"line": 3, "text": "A3 outro", "source": str(source_a), "source_line": 3},
            {"line": 4, "text": "B1 intro", "source": str(source_b), "source_line": 1},
            {"line": 5, "text": "B2 fusion evidence with caveat", "source": str(source_b), "source_line": 2},
        ]

        found = _call(action="search", corpus_id=corpus_id, query="fusion evidence", limit=5)

        assert [m["line"] for m in found["matches"]] == [2, 5]
        assert found["matches"][0]["source"] == str(source_a)
        assert found["matches"][0]["source_line"] == 2
        assert found["matches"][1]["source"] == str(source_b)
        assert found["matches"][1]["source_line"] == 2
    finally:
        reset_hermes_home_override(token)


def test_search_requires_all_query_terms_by_default_to_reduce_noise(tmp_path):
    token = set_hermes_home_override(tmp_path)
    try:
        text = "\n".join([
            "fusion evidence is strong",
            "fusion only appears here",
            "evidence only appears here",
            "FUSION with scattered EVIDENCE still counts",
        ])
        created = _call(action="create", name="search quality", text=text)

        found = _call(action="search", corpus_id=created["corpus"]["corpus_id"], query="fusion evidence", limit=10)

        assert [m["line"] for m in found["matches"]] == [1, 4]
        assert found["total_matches"] == 2
    finally:
        reset_hermes_home_override(token)


def test_map_prompt_contains_machine_usable_read_call_and_citation_contract(tmp_path):
    token = set_hermes_home_override(tmp_path)
    try:
        text = "\n".join(f"Line {i}" for i in range(1, 5))
        created = _call(action="create", name="mapping", text=text, chunk_lines=2)

        mapped = _call(action="map", corpus_id=created["corpus"]["corpus_id"], task="Find risks", max_chunks=1)

        prompt = mapped["tasks"][0]["prompt"]
        assert 'recursive_context(action="read"' in prompt
        assert "start_line=1" in prompt
        assert "line_count=2" in prompt
        assert "source_line" in prompt
        assert "corpus line" in prompt.lower()
    finally:
        reset_hermes_home_override(token)


def test_path_corpus_id_includes_source_identity_to_prevent_citation_overwrite(tmp_path):
    token = set_hermes_home_override(tmp_path / "home")
    try:
        source_a = tmp_path / "same-a.txt"
        source_b = tmp_path / "same-b.txt"
        source_a.write_text("same text\n", encoding="utf-8")
        source_b.write_text("same text\n", encoding="utf-8")

        first = _call(action="create", name="same", paths=[str(source_a)])
        second = _call(action="create", name="same", paths=[str(source_b)])

        assert first["corpus"]["corpus_id"] != second["corpus"]["corpus_id"]
        first_read = _call(action="read", corpus_id=first["corpus"]["corpus_id"], start_line=1, line_count=1)
        second_read = _call(action="read", corpus_id=second["corpus"]["corpus_id"], start_line=1, line_count=1)
        assert first_read["lines"][0]["source"] == str(source_a)
        assert second_read["lines"][0]["source"] == str(source_b)
    finally:
        reset_hermes_home_override(token)


def test_read_beyond_end_returns_empty_non_inverted_range(tmp_path):
    token = set_hermes_home_override(tmp_path)
    try:
        created = _call(action="create", name="short", text="one\ntwo")

        window = _call(action="read", corpus_id=created["corpus"]["corpus_id"], start_line=99, line_count=5)

        assert window["success"] is True
        assert window["range"] == {"start_line": 99, "end_line": 99}
        assert window["lines"] == []
    finally:
        reset_hermes_home_override(token)


def test_recursive_context_is_registered_as_file_toolset():
    from tools.recursive_context_tool import registry

    entry = registry.get_entry("recursive_context")
    assert entry is not None
    assert entry.toolset == "file"


def test_create_rejects_malformed_paths_without_iterating_characters(tmp_path):
    token = set_hermes_home_override(tmp_path / "home")
    try:
        source = tmp_path / "source.txt"
        source.write_text("hello\n", encoding="utf-8")

        string_paths = _call(action="create", name="bad paths", paths=str(source))
        assert string_paths["success"] is False
        assert "paths" in string_paths["error"]
        assert "list" in string_paths["error"]
        assert "not found" not in string_paths["error"].lower()

        non_string_entry = _call(action="create", name="bad paths", paths=[str(source), 42])
        assert non_string_entry["success"] is False
        assert "paths" in non_string_entry["error"]
        assert "strings" in non_string_entry["error"]
    finally:
        reset_hermes_home_override(token)


def test_numeric_parameters_return_clear_parameter_errors(tmp_path):
    token = set_hermes_home_override(tmp_path)
    try:
        bad_chunk = _call(action="create", name="numbers", text="one", chunk_lines="nope")
        assert bad_chunk["success"] is False
        assert "chunk_lines" in bad_chunk["error"]
        assert "invalid literal" not in bad_chunk["error"]

        created = _call(action="create", name="numbers", text="one\ntwo")
        corpus_id = created["corpus"]["corpus_id"]

        bad_start = _call(action="read", corpus_id=corpus_id, start_line="nope")
        assert bad_start["success"] is False
        assert "start_line" in bad_start["error"]
        assert "invalid literal" not in bad_start["error"]

        bad_limit = _call(action="search", corpus_id=corpus_id, query="one", limit="nope")
        assert bad_limit["success"] is False
        assert "limit" in bad_limit["error"]
        assert "invalid literal" not in bad_limit["error"]
    finally:
        reset_hermes_home_override(token)


def test_search_matches_include_text_and_structured_context_with_source_citations(tmp_path):
    token = set_hermes_home_override(tmp_path / "home")
    try:
        source_a = tmp_path / "a.txt"
        source_b = tmp_path / "b.txt"
        source_a.write_text("A1 setup\nA2 target evidence\n", encoding="utf-8")
        source_b.write_text("B1 adjacent context\n", encoding="utf-8")
        created = _call(action="create", name="context citations", paths=[str(source_a), str(source_b)])

        found = _call(
            action="search",
            corpus_id=created["corpus"]["corpus_id"],
            query="target evidence",
            context_lines=1,
        )

        match = found["matches"][0]
        assert match["text"] == "A2 target evidence"
        assert match["context"] == [
            {"line": 1, "text": "A1 setup", "source": str(source_a.resolve()), "source_line": 1},
            {"line": 2, "text": "A2 target evidence", "source": str(source_a.resolve()), "source_line": 2},
            {"line": 3, "text": "B1 adjacent context", "source": str(source_b.resolve()), "source_line": 1},
        ]
    finally:
        reset_hermes_home_override(token)


def test_path_ingest_uses_resolved_absolute_source_paths(tmp_path):
    token = set_hermes_home_override(tmp_path / "home")
    try:
        nested = tmp_path / "nested"
        nested.mkdir()
        source = nested / "source.txt"
        source.write_text("target line\n", encoding="utf-8")

        created = _call(action="create", name="resolved paths", paths=[str(source)])
        corpus_id = created["corpus"]["corpus_id"]
        assert created["corpus"]["sources"] == [str(source.resolve())]

        window = _call(action="read", corpus_id=corpus_id, start_line=1, line_count=1)
        assert window["lines"][0]["source"] == str(source.resolve())

        found = _call(action="search", corpus_id=corpus_id, query="target line")
        assert found["matches"][0]["source"] == str(source.resolve())
    finally:
        reset_hermes_home_override(token)


def test_read_falls_back_to_corpus_text_when_records_jsonl_is_corrupt(tmp_path):
    token = set_hermes_home_override(tmp_path)
    try:
        created = _call(action="create", name="recoverable", text="one\ntwo")
        corpus_id = created["corpus"]["corpus_id"]
        records_path = tmp_path / "recursive_context" / corpus_id / "records.jsonl"
        records_path.write_text("{not json}\n", encoding="utf-8")

        window = _call(action="read", corpus_id=corpus_id, start_line=1, line_count=2)

        assert window["success"] is True
        assert window["lines"] == [{"line": 1, "text": "one"}, {"line": 2, "text": "two"}]
    finally:
        reset_hermes_home_override(token)


def test_path_ingest_uses_read_file_safety_guards_for_sensitive_and_binary_paths(tmp_path):
    token = set_hermes_home_override(tmp_path / "home")
    try:
        sensitive = tmp_path / "home" / "auth.json"
        sensitive.parent.mkdir(parents=True, exist_ok=True)
        sensitive.write_text('{"placeholder":"not-secret"}', encoding="utf-8")
        blocked = _call(action="create", name="sensitive", paths=[str(sensitive)])
        assert blocked["success"] is False
        assert "access denied" in blocked["error"].lower() or "blocked" in blocked["error"].lower() or "refusing" in blocked["error"].lower()

        image = tmp_path / "image.png"
        image.write_bytes(b"\x89PNG\r\n\x1a\n")
        binary = _call(action="create", name="binary", paths=[str(image)])
        assert binary["success"] is False
        assert "binary" in binary["error"].lower()
    finally:
        reset_hermes_home_override(token)


def test_path_ingest_redacts_sensitive_content_before_storage_and_search(tmp_path):
    token = set_hermes_home_override(tmp_path / "home")
    try:
        source = tmp_path / "allowed.txt"
        raw_secret = "sk-" + "a" * 26 + "123456"
        source.write_text(f"safe prefix {raw_secret} safe suffix\n", encoding="utf-8")

        created = _call(action="create", name="redacted", paths=[str(source)])
        corpus_id = created["corpus"]["corpus_id"]
        window = _call(action="read", corpus_id=corpus_id, start_line=1, line_count=1)
        found = _call(action="search", corpus_id=corpus_id, query="safe prefix")

        assert raw_secret not in window["lines"][0]["text"]
        assert raw_secret not in found["matches"][0]["text"]
        assert "sk-" in window["lines"][0]["text"]
        assert "..." in window["lines"][0]["text"]
    finally:
        reset_hermes_home_override(token)


def test_raw_text_create_redacts_sensitive_content_before_storage_and_search(tmp_path):
    token = set_hermes_home_override(tmp_path)
    try:
        raw_secret = "sk-" + "b" * 26 + "654321"
        created = _call(action="create", name="raw redacted", text=f"safe prefix {raw_secret} safe suffix")
        corpus_id = created["corpus"]["corpus_id"]

        window = _call(action="read", corpus_id=corpus_id, start_line=1, line_count=1)
        found = _call(action="search", corpus_id=corpus_id, query="safe prefix")

        assert raw_secret not in window["lines"][0]["text"]
        assert raw_secret not in found["matches"][0]["text"]
        assert "..." in window["lines"][0]["text"]
    finally:
        reset_hermes_home_override(token)


def test_path_ingest_redacts_env_and_json_style_secrets_before_storage(tmp_path):
    token = set_hermes_home_override(tmp_path / "home")
    try:
        source = tmp_path / "config.txt"
        env_secret = "env" + "x" * 24
        json_secret = "json" + "y" * 24
        source.write_text(f"SERVICE_" + f"API_KEY={env_secret}\n{{\"password\": \"{json_secret}\"}}\n", encoding="utf-8")

        created = _call(action="create", name="strict redaction", paths=[str(source)])
        corpus_id = created["corpus"]["corpus_id"]
        window = _call(action="read", corpus_id=corpus_id, start_line=1, line_count=2)
        text = "\n".join(line["text"] for line in window["lines"])

        assert env_secret not in text
        assert json_secret not in text
        assert "SERVICE_" + "API_KEY=" in text
        assert "password" in text
    finally:
        reset_hermes_home_override(token)


def test_path_ingest_rejects_extensionless_binary_content(tmp_path):
    token = set_hermes_home_override(tmp_path / "home")
    try:
        binary = tmp_path / "maybe-text"
        binary.write_bytes(b"hello\x00world\x00")

        blocked = _call(action="create", name="binary content", paths=[str(binary)])

        assert blocked["success"] is False
        assert "binary" in blocked["error"].lower()
    finally:
        reset_hermes_home_override(token)


def test_path_ingest_rejects_non_regular_files(tmp_path):
    import os
    import stat

    token = set_hermes_home_override(tmp_path / "home")
    try:
        fifo = tmp_path / "pipe"
        os.mkfifo(fifo)

        blocked = _call(action="create", name="fifo", paths=[str(fifo)])

        assert blocked["success"] is False
        assert "regular file" in blocked["error"].lower()
        assert stat.S_ISFIFO(fifo.stat().st_mode)
    finally:
        reset_hermes_home_override(token)


def test_legacy_file_tools_alias_exposes_recursive_context():
    import tools.recursive_context_tool  # noqa: F401
    from model_tools import get_tool_definitions

    definitions = get_tool_definitions(enabled_toolsets=["file_tools"], quiet_mode=True)
    assert "recursive_context" in [definition["function"]["name"] for definition in definitions]


def test_delete_removes_corpus_directory_even_when_metadata_is_corrupt(tmp_path):
    token = set_hermes_home_override(tmp_path)
    try:
        created = _call(action="create", name="corrupt", text="one")
        corpus_id = created["corpus"]["corpus_id"]
        corpus_dir = tmp_path / "recursive_context" / corpus_id
        (corpus_dir / "metadata.json").write_text("{not json}\n", encoding="utf-8")

        deleted = _call(action="delete", corpus_id=corpus_id)

        assert deleted == {"success": True, "deleted": corpus_id}
        assert not corpus_dir.exists()
    finally:
        reset_hermes_home_override(token)


def test_recursive_context_is_exposed_through_file_toolset_definitions():
    import tools.recursive_context_tool  # noqa: F401
    from model_tools import get_tool_definitions
    from toolsets import resolve_toolset

    assert "recursive_context" in resolve_toolset("file")
    definitions = get_tool_definitions(enabled_toolsets=["file"], quiet_mode=True)
    assert "recursive_context" in [definition["function"]["name"] for definition in definitions]


def test_corpus_id_includes_chunk_lines_to_prevent_metadata_overwrite(tmp_path):
    token = set_hermes_home_override(tmp_path)
    try:
        first = _call(action="create", name="same text", text="one\ntwo\nthree", chunk_lines=1)
        second = _call(action="create", name="same text", text="one\ntwo\nthree", chunk_lines=3)

        assert first["corpus"]["corpus_id"] != second["corpus"]["corpus_id"]
        first_map = _call(action="map", corpus_id=first["corpus"]["corpus_id"], task="Summarize")
        second_map = _call(action="map", corpus_id=second["corpus"]["corpus_id"], task="Summarize")
        assert first_map["tasks"][0]["range"] == {"start_line": 1, "end_line": 1}
        assert second_map["tasks"][0]["range"] == {"start_line": 1, "end_line": 3}
    finally:
        reset_hermes_home_override(token)


def test_delete_removes_corpus(tmp_path):
    token = set_hermes_home_override(tmp_path)
    try:
        created = _call(action="create", name="throwaway", text="one\ntwo")
        corpus_id = created["corpus"]["corpus_id"]

        deleted = _call(action="delete", corpus_id=corpus_id)
        assert deleted == {"success": True, "deleted": corpus_id}

        missing = _call(action="read", corpus_id=corpus_id)
        assert missing["success"] is False
        assert "not found" in missing["error"].lower()
    finally:
        reset_hermes_home_override(token)
