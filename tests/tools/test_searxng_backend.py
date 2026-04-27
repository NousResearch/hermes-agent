import pytest
from unittest.mock import patch, MagicMock

class TestSearxngBackendIntegration:
    """Test suite for SearXNG backend integration logic."""

    def setup_method(self):
        import tools.web_tools
        self.web_tools = tools.web_tools

    def test_get_backend_accepts_searxng(self):
        """_get_backend() returns 'searxng' when configured."""
        with patch.object(self.web_tools, '_load_web_config', return_value={'backend': 'searxng'}):
            assert self.web_tools._get_backend() == 'searxng'

    def test_is_backend_available_with_base_url(self):
        """_is_backend_available('searxng') returns True with base_url."""
        with patch.object(self.web_tools, '_load_web_config', return_value={'backend': 'searxng', 'base_url': 'http://searxng:8080'}):
            assert self.web_tools._is_backend_available('searxng') is True

    def test_is_backend_available_without_base_url(self):
        """_is_backend_available('searxng') returns False without base_url."""
        with patch.object(self.web_tools, '_load_web_config', return_value={'backend': 'searxng'}):
            assert self.web_tools._is_backend_available('searxng') is False

    def test_check_web_api_key_includes_searxng(self):
        """check_web_api_key() works with searxng configured (available)."""
        with patch.object(self.web_tools, '_is_backend_available', return_value=True):
            assert self.web_tools.check_web_api_key() is True

    def test_web_search_tool_dispatches_searxng(self):
        """mock _searxng_search to verify web_search_tool dispatches to it."""
        import json
        with patch.object(self.web_tools, '_get_backend', return_value='searxng'), \
             patch.object(self.web_tools, '_searxng_search', return_value={'success': True, 'data': {'web': []}}) as mock_search:
            result = self.web_tools.web_search_tool('test query')
            mock_search.assert_called_once_with('test query', 5)
            # The tool might serialize dicts to json depending on implementation
            if isinstance(result, str):
                result = json.loads(result)
            assert result == {'success': True, 'data': {'web': []}}

    def test_searxng_search_basic(self):
        """basic _searxng_search call returns success with results (mock requests.get)."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "results": [
                {"title": "Test Title", "url": "https://test.com", "content": "Test Content"}
            ]
        }
        mock_resp.raise_for_status = MagicMock()

        with patch.object(self.web_tools, '_load_web_config', return_value={'backend': 'searxng', 'base_url': 'http://searxng:8080'}), \
             patch('requests.get', return_value=mock_resp) as mock_get:
            result = self.web_tools._searxng_search('test query')
            
            mock_get.assert_called_once()
            assert result['success'] is True
            assert len(result['data']['web']) == 1
            assert result['data']['web'][0]['title'] == 'Test Title'
