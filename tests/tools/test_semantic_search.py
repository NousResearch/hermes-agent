"""Tests for semantic search tool."""

import json
import os
import sqlite3
import tempfile

import pytest


class TestGetIndexDb:
    def test_hash_based_index_path(self, tmp_path):
        """Index path should use canonical-root hash, not basename."""
        from tools.semantic_search import _get_index_db

        db1 = _get_index_db("/work/a/myapp")
        db2 = _get_index_db("/work/b/myapp")
        # Different roots -> different DB files (no basename collision)
        assert db1 != db2

    def test_same_root_same_index(self, tmp_path):
        """Same root path should produce the same index."""
        from tools.semantic_search import _get_index_db

        db1 = _get_index_db("/work/a/myapp")
        db2 = _get_index_db("/work/a/myapp")
        assert db1 == db2

    def test_symlink_resolves(self, tmp_path):
        """Symlinked root should resolve to canonical path."""
        from tools.semantic_search import _get_index_db

        real = tmp_path / "real"
        real.mkdir()
        link = tmp_path / "link"
        link.symlink_to(real)

        db_real = _get_index_db(str(real))
        db_link = _get_index_db(str(link))
        assert db_real == db_link


class TestSensitiveFileExclusion:
    def test_env_files_excluded(self):
        from tools.semantic_search import _is_sensitive_file
        assert _is_sensitive_file(".env")
        assert _is_sensitive_file(".env.local")
        assert _is_sensitive_file(".env.production")

    def test_normal_files_allowed(self):
        from tools.semantic_search import _is_sensitive_file
        assert not _is_sensitive_file("main.py")
        assert not _is_sensitive_file("config.yaml")
        assert not _is_sensitive_file("README.md")


class TestChunkFile:
    def test_basic_chunking(self):
        from tools.semantic_search import _chunk_file
        content = "\n".join(f"line {i}" for i in range(100))
        chunks = _chunk_file(content, max_chars=200)
        assert len(chunks) > 1
        # All original lines present
        full = "\n".join(chunks)
        for i in range(100):
            assert f"line {i}" in full

    def test_small_content_single_chunk(self):
        from tools.semantic_search import _chunk_file
        chunks = _chunk_file("hello world")
        assert len(chunks) == 1
        assert chunks[0] == "hello world"


class TestClassifyQuery:
    def test_symbol_query(self):
        from tools.semantic_search import _classify_query
        result = _classify_query("find class FooBar")
        assert result["type"] == "symbol"
        assert result["target"] == "FooBar"

    def test_error_query(self):
        from tools.semantic_search import _classify_query
        result = _classify_query("error in auth flow")
        assert result["type"] == "error"

    def test_semantic_query(self):
        from tools.semantic_search import _classify_query
        result = _classify_query("how does the caching layer work")
        assert result["type"] == "semantic"


class TestSemanticSearchIndex:
    def test_index_mode(self, tmp_path):
        """Index mode should create DB and index files."""
        from tools.semantic_search import semantic_search

        # Create a test project
        project = tmp_path / "testproject"
        project.mkdir()
        (project / "main.py").write_text("def hello():\n    return 'world'")
        (project / ".env").write_text("SECRET=abc123")  # should be excluded

        result = json.loads(semantic_search(
            query="hello",
            project_root=str(project),
            mode="index",
        ))
        assert result["success"] is True
        assert result["stats"]["files_processed"] >= 1
        # .env file should NOT be indexed
        db_path = result["project"]

    def test_index_excludes_sensitive_files(self, tmp_path):
        """Sensitive files should not be indexed."""
        from tools.semantic_search import _get_index_db, semantic_search

        project = tmp_path / "secrets_project"
        project.mkdir()
        (project / "app.py").write_text("import os")
        (project / ".env").write_text("SECRET_KEY=xyz")

        result = json.loads(semantic_search(
            query="secret",
            project_root=str(project),
            mode="index",
        ))

        # Check that .env content is NOT in the index
        db = _get_index_db(str(project))
        conn = sqlite3.connect(db)
        cursor = conn.execute("SELECT text FROM chunks")
        all_text = " ".join(row[0] for row in cursor.fetchall())
        conn.close()
        assert "SECRET_KEY" not in all_text


class TestSemanticSearchHybrid:
    def test_hybrid_mode_returns_results(self, tmp_path):
        """Hybrid mode should return results without crashing."""
        from tools.semantic_search import semantic_search

        project = tmp_path / "hybrid_project"
        project.mkdir()
        (project / "auth.py").write_text(
            "def authenticate_user(username, password):\n"
            "    if not username:\n"
            "        raise ValueError('missing username')\n"
            "    return {'token': 'abc'}"
        )

        # Index first
        semantic_search(query="auth", project_root=str(project), mode="index")

        # Search
        result = json.loads(semantic_search(
            query="user login authentication",
            project_root=str(project),
            mode="hybrid",
        ))
        assert result["success"] is True
        assert result["mode"] == "hybrid"
        assert len(result["results"]) > 0


class TestRerankWithLlm:
    def test_rerank_survives_bad_json(self):
        """Reranker should not crash on unparseable LLM response."""
        from unittest.mock import patch, MagicMock
        from tools.semantic_search import _rerank_with_llm

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "not valid json"

        with patch("agent.auxiliary_client.call_llm", return_value=mock_response):
            candidates = [{"file": "a.py", "snippet": "code", "keyword_score": 0.5}]
            result = _rerank_with_llm("test query", candidates)
            # Should return candidates unchanged (no crash)
            assert len(result) == 1

    def test_rerank_extracts_content(self):
        """Reranker should extract .choices[0].message.content."""
        from unittest.mock import patch, MagicMock
        from tools.semantic_search import _rerank_with_llm

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps([
            {"index": 0, "score": 9, "reason": "perfect match"}
        ])

        with patch("agent.auxiliary_client.call_llm", return_value=mock_response):
            candidates = [{"file": "a.py", "snippet": "code", "keyword_score": 0.5}]
            result = _rerank_with_llm("test query", candidates)
            assert result[0]["relevance"] == 9
            assert result[0]["score"] == 0.9
