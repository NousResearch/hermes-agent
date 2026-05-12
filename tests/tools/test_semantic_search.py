"""Tests for semantic_search tool."""

import json
import os
import pytest
import tempfile
from unittest.mock import patch


class TestSemanticSearch:
    @pytest.fixture
    def temp_project(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            src = os.path.join(tmpdir, "src")
            os.makedirs(src)
            files = {
                "auth.py": "def login(username, password):\n    return True\n",
                "api.py": "def get_user(user_id):\n    return {'id': user_id}\n",
                "utils.py": "def format_date(date):\n    return str(date)\n",
            }
            for name, content in files.items():
                with open(os.path.join(src, name), "w") as f:
                    f.write(content)
            yield tmpdir

    def test_check_requirements(self):
        from tools.semantic_search import check_semantic_search_requirements
        assert check_semantic_search_requirements() is True

    def test_build_index(self, temp_project):
        from tools.semantic_search import semantic_search
        output = semantic_search("test", temp_project, mode="index")
        data = json.loads(output)
        assert data["success"] is True
        assert data["operation"] == "index"
        assert data["stats"]["files_processed"] >= 1
        assert data["stats"]["chunks_indexed"] >= 1

    @patch("agent.auxiliary_client.call_llm")
    def test_semantic_search(self, mock_llm, temp_project):
        mock_llm.return_value = '[{"index": 0, "score": 9, "reason": "relevant"}]'
        from tools.semantic_search import semantic_search
        semantic_search("test", temp_project, mode="index")
        output = semantic_search("login function", temp_project, mode="semantic")
        data = json.loads(output)
        assert data["success"] is True
        assert data["total"] >= 1

    @patch("agent.auxiliary_client.call_llm")
    def test_hybrid_search(self, mock_llm, temp_project):
        mock_llm.return_value = '[{"index": 0, "score": 8, "reason": "relevant"}]'
        from tools.semantic_search import semantic_search
        semantic_search("test", temp_project, mode="index")
        output = semantic_search("get user data", temp_project, mode="hybrid")
        data = json.loads(output)
        assert data["success"] is True
        assert data["total"] >= 1

    def test_keyword_search(self, temp_project):
        from tools.semantic_search import semantic_search
        output = semantic_search("login", temp_project, mode="keyword")
        data = json.loads(output)
        assert data["success"] is True
        assert data["mode"] == "keyword"
        assert data["total"] >= 1

    def test_query_classification_symbol(self):
        from tools.semantic_search import _classify_query
        result = _classify_query("class User")
        assert result["type"] == "symbol"
        assert result["confidence"] > 0.8

    def test_query_classification_error(self):
        from tools.semantic_search import _classify_query
        result = _classify_query("error in authentication")
        assert result["type"] == "error"
        assert result["confidence"] > 0.6

    def test_no_index_returns_error(self, temp_project):
        from tools.semantic_search import semantic_search
        output = semantic_search("test", temp_project, mode="hybrid")
        data = json.loads(output)
        assert data["success"] is False
        assert "No index found" in data["error"]

    def test_project_root_not_found(self):
        from tools.semantic_search import semantic_search
        output = semantic_search("test", "/nonexistent")
        data = json.loads(output)
        assert data["success"] is False

    @patch("agent.auxiliary_client.call_llm")
    def test_file_pattern_filter(self, mock_llm, temp_project):
        mock_llm.return_value = '[{"index": 0, "score": 8, "reason": ""}]'
        from tools.semantic_search import semantic_search
        semantic_search("test", temp_project, mode="index")
        output = semantic_search("test", temp_project, file_pattern="*.py")
        data = json.loads(output)
        assert data["success"] is True

    @patch("agent.auxiliary_client.call_llm")
    def test_max_results(self, mock_llm, temp_project):
        mock_llm.return_value = '[{"index": 0, "score": 8, "reason": ""}]'
        from tools.semantic_search import semantic_search
        semantic_search("test", temp_project, mode="index")
        output = semantic_search("test", temp_project, max_results=2)
        data = json.loads(output)
        assert data["success"] is True
        assert len(data["results"]) <= 2

    def test_intent_in_response(self, temp_project):
        from tools.semantic_search import semantic_search
        output = semantic_search("authentication error", temp_project, mode="keyword")
        data = json.loads(output)
        assert "intent" in data
        assert data["intent"]["type"] in ("symbol", "search", "semantic", "error")


class TestSemanticSearchSchema:
    def test_schema_has_required_fields(self):
        from tools.semantic_search import SEMANTIC_SEARCH_SCHEMA
        assert SEMANTIC_SEARCH_SCHEMA["name"] == "semantic_search"
        props = SEMANTIC_SEARCH_SCHEMA["parameters"]["properties"]
        assert "query" in props
        assert "project_root" in props
        assert "mode" in props
        assert props["mode"]["enum"] == ["hybrid", "semantic", "keyword", "index"]