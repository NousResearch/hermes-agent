"""Tests for session_search tool — semantic and hybrid search parameters."""

import json
import pytest
from unittest.mock import MagicMock, patch


class TestSessionSearchSemanticParam:
    """Tests for the semantic=True parameter on session_search."""

    def test_semantic_param_accepted(self):
        """session_search accepts semantic=False (default, backward compat)."""
        from tools.session_search_tool import session_search

        with patch("hermes_state.SessionDB") as mock_db_cls:
            mock_db = MagicMock()
            mock_db_cls.return_value = mock_db
            mock_db.search_messages.return_value = []
            mock_db.list_sessions_rich.return_value = []
            mock_db.resolve_session_by_title.return_value = None

            result = json.loads(session_search(query="test", semantic=False))
            assert result["success"] is True

    def test_semantic_true_routes_to_hybrid(self):
        """semantic=True calls search_hybrid instead of search_messages."""
        from tools.session_search_tool import session_search

        with patch("hermes_state.SessionDB") as mock_db_cls, \
             patch("tools.session_search_tool._resolve_embedding_provider") as mock_resolve:
            mock_db = MagicMock()
            mock_db_cls.return_value = mock_db
            mock_db.search_hybrid.return_value = [
                {"id": 1, "session_id": "s1", "role": "user",
                 "snippet": "test", "timestamp": 1.0, "tool_name": None,
                 "source": "cli", "model": None, "session_started": 1.0,
                 "context": [], "hybrid_score": 0.1}
            ]
            mock_db.list_sessions_rich.return_value = []
            mock_db.get_anchored_view.return_value = {
                "window": [], "bookend_start": [], "bookend_end": [],
                "messages_before": 0, "messages_after": 0,
            }
            mock_db.get_session.return_value = {}
            mock_db.resolve_session_by_title.return_value = None
            mock_resolve.return_value = MagicMock()

            result = json.loads(session_search(query="test", semantic=True))
            assert result["success"] is True
            mock_db.search_hybrid.assert_called_once()
            mock_db.search_messages.assert_not_called()

    def test_semantic_true_passes_hybrid_weight(self):
        """hybrid_weight parameter is forwarded to search_hybrid."""
        from tools.session_search_tool import session_search

        with patch("hermes_state.SessionDB") as mock_db_cls, \
             patch("tools.session_search_tool._resolve_embedding_provider") as mock_resolve:
            mock_db = MagicMock()
            mock_db_cls.return_value = mock_db
            mock_db.search_hybrid.return_value = []
            mock_db.list_sessions_rich.return_value = []
            mock_db.resolve_session_by_title.return_value = None
            mock_resolve.return_value = MagicMock()

            session_search(query="test", semantic=True, hybrid_weight=0.7)
            call_kwargs = mock_db.search_hybrid.call_args.kwargs
            assert call_kwargs.get("hybrid_weight") == 0.7

    def test_semantic_false_uses_fts5(self):
        """semantic=False (default) uses existing FTS5 search_messages path."""
        from tools.session_search_tool import session_search

        with patch("hermes_state.SessionDB") as mock_db_cls:
            mock_db = MagicMock()
            mock_db_cls.return_value = mock_db
            mock_db.search_messages.return_value = [
                {"id": 1, "session_id": "s1", "role": "user",
                 "snippet": "test", "timestamp": 1.0, "tool_name": None,
                 "source": "cli", "model": None, "session_started": 1.0,
                 "context": []}
            ]
            mock_db.list_sessions_rich.return_value = []
            mock_db.get_anchored_view.return_value = {
                "window": [], "bookend_start": [], "bookend_end": [],
                "messages_before": 0, "messages_after": 0,
            }
            mock_db.get_session.return_value = {}
            mock_db.resolve_session_by_title.return_value = None

            result = json.loads(session_search(query="test", semantic=False))
            assert result["success"] is True
            mock_db.search_messages.assert_called_once()
            mock_db.search_hybrid.assert_not_called()

    def test_semantic_true_with_role_filter(self):
        """role_filter is passed through to search_hybrid."""
        from tools.session_search_tool import session_search

        with patch("hermes_state.SessionDB") as mock_db_cls, \
             patch("tools.session_search_tool._resolve_embedding_provider") as mock_resolve:
            mock_db = MagicMock()
            mock_db_cls.return_value = mock_db
            mock_db.search_hybrid.return_value = []
            mock_db.list_sessions_rich.return_value = []
            mock_db.resolve_session_by_title.return_value = None
            mock_resolve.return_value = MagicMock()

            session_search(query="test", semantic=True, role_filter="user,assistant")
            call_kwargs = mock_db.search_hybrid.call_args.kwargs
            assert "role_filter" in call_kwargs
            assert call_kwargs["role_filter"] == ["user", "assistant"]

    def test_semantic_true_exclude_sources_passed(self):
        """_HIDDEN_SESSION_SOURCES are passed as exclude_sources to search_hybrid."""
        from tools.session_search_tool import session_search

        with patch("hermes_state.SessionDB") as mock_db_cls, \
             patch("tools.session_search_tool._resolve_embedding_provider") as mock_resolve:
            mock_db = MagicMock()
            mock_db_cls.return_value = mock_db
            mock_db.search_hybrid.return_value = []
            mock_db.list_sessions_rich.return_value = []
            mock_db.resolve_session_by_title.return_value = None
            mock_resolve.return_value = MagicMock()

            session_search(query="test", semantic=True)
            call_kwargs = mock_db.search_hybrid.call_args.kwargs
            assert "exclude_sources" in call_kwargs


class TestResolveEmbeddingProviderCache:
    """Tests for the module-level provider cache in session_search_tool."""

    @pytest.fixture(autouse=True)
    def _clear_cache(self):
        """Reset the module-level cache before each test."""
        import tools.session_search_tool as mod
        mod._cached_embedding_provider = None
        mod._cached_embedding_config_hash = None

    def test_cache_hit_returns_same_object(self):
        """Second call returns the cached provider, not a new instance."""
        from tools.session_search_tool import _resolve_embedding_provider

        with patch("hermes_cli.config.load_config") as mock_load_config, \
             patch("agent.embedding_provider.resolve_embedding_provider") as mock_resolve:
            mock_load_config.return_value = {
                "auxiliary": {
                    "session_search": {
                        "embedding": {
                            "provider": "ollama",
                            "model": "nomic-embed-text",
                            "base_url": "http://localhost:11434/v1",
                            "api_key": "",
                            "dimensions": 768,
                        }
                    }
                }
            }
            mock_provider = MagicMock()
            mock_resolve.return_value = mock_provider

            first = _resolve_embedding_provider()
            second = _resolve_embedding_provider()

            assert first is second, "Cache should return the same object"
            # resolve_embedding_provider should only be called once
            mock_resolve.assert_called_once()

    def test_cache_busts_on_config_change(self):
        """Different config produces a new provider."""
        from tools.session_search_tool import _resolve_embedding_provider

        with patch("hermes_cli.config.load_config") as mock_load_config, \
             patch("agent.embedding_provider.resolve_embedding_provider") as mock_resolve:
            mock_load_config.return_value = {
                "auxiliary": {
                    "session_search": {
                        "embedding": {
                            "provider": "ollama",
                            "model": "nomic-embed-text",
                            "base_url": "http://localhost:11434/v1",
                            "api_key": "",
                            "dimensions": 768,
                        }
                    }
                }
            }
            mock_provider_a = MagicMock()
            mock_provider_b = MagicMock()
            mock_resolve.side_effect = [mock_provider_a, mock_provider_b]

            first = _resolve_embedding_provider()
            assert first is mock_provider_a

            # Change config
            mock_load_config.return_value = {
                "auxiliary": {
                    "session_search": {
                        "embedding": {
                            "provider": "openai",
                            "model": "text-embedding-3-small",
                            "base_url": "",
                            "api_key": "sk-test",
                            "dimensions": 1536,
                        }
                    }
                }
            }

            second = _resolve_embedding_provider()
            assert second is mock_provider_b
            assert second is not first, "Different config should produce different provider"
            assert mock_resolve.call_count == 2
