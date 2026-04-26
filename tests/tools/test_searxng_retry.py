"""Tests for SearXNG search retry logic.

Coverage:
  _searxng_search() — normal results, empty results retry, exception retry,
  both-attempts-failure, interrupted mid-retry, base_url missing.
"""

import pytest
from unittest.mock import patch, MagicMock


class TestSearxngSearchRetry:
    """Test suite for _searxng_search retry behavior."""

    def setup_method(self):
        import tools.web_tools
        self._config_patcher = patch.object(
            tools.web_tools, '_load_web_config',
            return_value={'base_url': 'http://test-searxng:8881', 'backend': 'searxng'}
        )
        self.mock_config = self._config_patcher.start()
        # Ensure interrupt check returns False by default
        self._interrupt_patcher = patch('tools.interrupt.is_interrupted', return_value=False)
        self._interrupt_patcher.start()

    def teardown_method(self):
        self._config_patcher.stop()
        self._interrupt_patcher.stop()

    def _mock_response(self, results=None, status_code=200, json_data=None):
        """Create a mock requests.Response with SearXNG JSON structure."""
        mock = MagicMock()
        mock.status_code = status_code
        if json_data is not None:
            mock.json.return_value = json_data
        elif results is not None:
            mock.json.return_value = {
                "results": results,
                "number_of_results": len(results),
                "query": "test query"
            }
        else:
            mock.json.return_value = {"results": [], "number_of_results": 0, "query": "test"}
        mock.raise_for_status = MagicMock()
        return mock

    def test_returns_results_on_first_attempt(self):
        """Normal case: results returned on first try, no retry needed."""
        import tools.web_tools
        mock_resp = self._mock_response(results=[
            {"title": "Result 1", "url": "https://ex.com/1", "content": "Desc 1"},
            {"title": "Result 2", "url": "https://ex.com/2", "content": "Desc 2"},
            {"title": "Result 3", "url": "https://ex.com/3", "content": "Desc 3"},
        ])

        with patch('requests.get', return_value=mock_resp) as mock_get:
            result = tools.web_tools._searxng_search("test query", limit=5)

        assert result['success'] is True
        assert len(result['data']['web']) == 3
        assert result['data']['web'][0]['title'] == 'Result 1'
        assert result['data']['web'][0]['position'] == 1
        mock_get.assert_called_once()

    def test_retries_on_empty_results(self):
        """Empty results on first attempt → retry → succeed on second."""
        import tools.web_tools
        empty_resp = self._mock_response(results=[])
        success_resp = self._mock_response(results=[
            {"title": "Delayed Result", "url": "https://ex.com/d", "content": "C"},
        ])

        with patch('requests.get',
                   side_effect=[empty_resp, success_resp]) as mock_get:
            result = tools.web_tools._searxng_search("test", limit=5)

        assert result['success'] is True
        assert len(result['data']['web']) == 1
        assert result['data']['web'][0]['title'] == 'Delayed Result'
        assert mock_get.call_count == 2
        # Second call should have timeout=45
        assert mock_get.call_args_list[1][1]['timeout'] == 45

    def test_both_attempts_empty_returns_error(self):
        """Both attempts return empty → error with descriptive message."""
        import tools.web_tools
        empty_resp = self._mock_response(results=[])

        with patch('requests.get', return_value=empty_resp) as mock_get:
            result = tools.web_tools._searxng_search("nonsense", limit=5)

        assert result['success'] is False
        assert '0 results after 2 attempts' in result['error']
        assert mock_get.call_count == 2

    def test_retries_on_exception(self):
        """First attempt raises exception → retry → succeed on second."""
        import tools.web_tools
        success_resp = self._mock_response(results=[
            {"title": "Recovered", "url": "https://ex.com/r", "content": "C"},
        ])

        with patch('requests.get',
                   side_effect=[ConnectionError("timeout"), success_resp]) as mock_get:
            result = tools.web_tools._searxng_search("test", limit=5)

        assert result['success'] is True
        assert len(result['data']['web']) == 1
        assert mock_get.call_count == 2

    def test_both_attempts_exception_returns_error(self):
        """Both attempts raise exception → error with last exception message."""
        import tools.web_tools

        with patch('requests.get',
                   side_effect=ConnectionError("refused")) as mock_get:
            result = tools.web_tools._searxng_search("test", limit=5)

        assert result['success'] is False
        assert 'refused' in result['error']
        assert mock_get.call_count == 2

    def test_interrupt_on_retry(self):
        """Interrupt fires between attempts → aborts and returns error."""
        import tools.web_tools
        empty_resp = self._mock_response(results=[])

        with patch('requests.get', return_value=empty_resp):
            with patch('tools.interrupt.is_interrupted',
                       side_effect=[False, True]) as mock_int:
                result = tools.web_tools._searxng_search("test", limit=5)

        assert result['success'] is False
        assert result['error'] == 'Interrupted'

    def test_missing_base_url(self):
        """No base_url configured → immediate error without retry."""
        import tools.web_tools
        self._config_patcher.stop()
        config_patcher = patch.object(
            tools.web_tools, '_load_web_config',
            return_value={'backend': 'searxng'}
        )
        config_patcher.start()

        try:
            result = tools.web_tools._searxng_search("test", limit=5)
            assert result['success'] is False
            assert 'base_url' in result['error'].lower()
        finally:
            config_patcher.stop()
            self._config_patcher.start()

    def test_result_limit_respected(self):
        """The limit parameter caps returned results."""
        import tools.web_tools
        many_results = [
            {"title": f"R{i}", "url": f"https://ex.com/{i}", "content": f"C{i}"}
            for i in range(15)
        ]
        mock_resp = self._mock_response(results=many_results)

        with patch('requests.get', return_value=mock_resp):
            result = tools.web_tools._searxng_search("test", limit=3)

        assert len(result['data']['web']) == 3
        assert result['data']['web'][0]['position'] == 1
        assert result['data']['web'][2]['position'] == 3

    def test_first_timeout_second_normal(self):
        """Regression: first attempt uses 30s, second uses 45s client timeout."""
        import tools.web_tools
        empty_resp = self._mock_response(results=[])
        success_resp = self._mock_response(results=[
            {"title": "T", "url": "https://ex.com", "content": "C"},
        ])

        with patch('requests.get',
                   side_effect=[empty_resp, success_resp]) as mock_get:
            tools.web_tools._searxng_search("test", limit=5)

        call1_kwargs = mock_get.call_args_list[0][1]
        call2_kwargs = mock_get.call_args_list[1][1]
        assert call1_kwargs['timeout'] == 30
        assert call2_kwargs['timeout'] == 45
