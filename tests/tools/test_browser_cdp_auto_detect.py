"""Tests for Chrome CDP auto-detection in browser_tool.py."""

from unittest.mock import Mock, patch
import socket

import pytest


class TestCheckCdpPort:
    """Tests for _check_cdp_port function."""

    def test_returns_true_when_port_is_open(self):
        """Should return True when Chrome is listening on the port."""
        from tools.browser_tool import _check_cdp_port

        with patch('socket.socket') as mock_socket:
            mock_instance = Mock()
            mock_socket.return_value = mock_instance
            mock_instance.connect.return_value = None
            
            result = _check_cdp_port(port=9222)
            
            assert result is True
            mock_instance.connect.assert_called_once_with(("127.0.0.1", 9222))
            mock_instance.close.assert_called_once()

    def test_returns_false_when_port_is_closed(self):
        """Should return False when nothing is listening on the port."""
        from tools.browser_tool import _check_cdp_port

        with patch('socket.socket') as mock_socket:
            mock_instance = Mock()
            mock_socket.return_value = mock_instance
            mock_instance.connect.side_effect = ConnectionRefusedError("Connection refused")
            
            result = _check_cdp_port(port=9222)
            
            assert result is False

    def test_returns_false_on_timeout(self):
        """Should return False when connection times out."""
        from tools.browser_tool import _check_cdp_port

        with patch('socket.socket') as mock_socket:
            mock_instance = Mock()
            mock_socket.return_value = mock_instance
            mock_instance.connect.side_effect = socket.timeout("Timeout")
            
            result = _check_cdp_port(port=9222, timeout=1.0)
            
            assert result is False

    def test_returns_false_on_os_error(self):
        """Should return False on any OS error."""
        from tools.browser_tool import _check_cdp_port

        with patch('socket.socket') as mock_socket:
            mock_instance = Mock()
            mock_socket.return_value = mock_instance
            mock_instance.connect.side_effect = OSError("Network unreachable")
            
            result = _check_cdp_port(port=9222)
            
            assert result is False

    def test_custom_host_and_port(self):
        """Should use custom host and port when provided."""
        from tools.browser_tool import _check_cdp_port

        with patch('socket.socket') as mock_socket:
            mock_instance = Mock()
            mock_socket.return_value = mock_instance
            
            result = _check_cdp_port(host="192.168.1.100", port=9223)
            
            assert result is True
            mock_instance.connect.assert_called_once_with(("192.168.1.100", 9223))


class TestTryAutoDetectCdp:
    """Tests for _try_auto_detect_cdp function."""

    def test_detects_chrome_on_default_port(self):
        """Should detect Chrome running on default port 9222."""
        from tools.browser_tool import _try_auto_detect_cdp

        with patch('tools.browser_tool._check_cdp_port') as mock_check:
            mock_check.side_effect = lambda port, **kw: port == 9222
            
            result = _try_auto_detect_cdp()
            
            assert result == "ws://127.0.0.1:9222"
            mock_check.assert_any_call(port=9222)

    def test_detects_chrome_on_alternative_port(self):
        """Should detect Chrome running on alternative port 9223."""
        from tools.browser_tool import _try_auto_detect_cdp

        with patch('tools.browser_tool._check_cdp_port') as mock_check:
            mock_check.side_effect = lambda port, **kw: port == 9223
            
            result = _try_auto_detect_cdp()
            
            assert result == "ws://127.0.0.1:9223"

    def test_checks_multiple_alternative_ports(self):
        """Should check ports 9223-9225 for SSH tunnel scenarios."""
        from tools.browser_tool import _try_auto_detect_cdp

        with patch('tools.browser_tool._check_cdp_port') as mock_check:
            mock_check.side_effect = lambda port, **kw: port == 9224
            
            result = _try_auto_detect_cdp()
            
            assert result == "ws://127.0.0.1:9224"
            # Should have checked 9222, 9223, then 9224
            assert mock_check.call_count >= 3

    def test_returns_empty_when_no_chrome_detected(self):
        """Should return empty string when no Chrome instance is found."""
        from tools.browser_tool import _try_auto_detect_cdp

        with patch('tools.browser_tool._check_cdp_port') as mock_check:
            mock_check.return_value = False
            
            result = _try_auto_detect_cdp()
            
            assert result == ""
            # Should have checked default port and all alternative ports
            assert mock_check.call_count == 4  # 9222, 9223, 9224, 9225

    def test_respects_auto_detect_env_var(self):
        """Should skip detection when BROWSER_CDP_AUTO_DETECT=false."""
        from tools.browser_tool import _try_auto_detect_cdp

        with patch.dict('os.environ', {'BROWSER_CDP_AUTO_DETECT': 'false'}):
            with patch('tools.browser_tool._check_cdp_port') as mock_check:
                result = _try_auto_detect_cdp()
                
                assert result == ""
                mock_check.assert_not_called()

    def test_auto_detect_enabled_by_default(self):
        """Should enable auto-detection by default."""
        from tools.browser_tool import _try_auto_detect_cdp

        with patch.dict('os.environ', {}, clear=False):
            # Ensure env var is not set
            import os
            original = os.environ.pop('BROWSER_CDP_AUTO_DETECT', None)
            try:
                with patch('tools.browser_tool._check_cdp_port') as mock_check:
                    mock_check.return_value = False
                    result = _try_auto_detect_cdp()
                    
                    assert result == ""
                    assert mock_check.call_count == 4  # Should still check ports
            finally:
                if original:
                    os.environ['BROWSER_CDP_AUTO_DETECT'] = original


class TestGetSessionInfoWithAutoDetect:
    """Tests for _get_session_info integration with auto-detection."""

    def test_uses_auto_detected_cdp_when_available(self):
        """Should use auto-detected CDP when Chrome is running."""
        from tools.browser_tool import _get_session_info, _active_sessions

        # Clear any existing sessions
        _active_sessions.clear()

        with patch('tools.browser_tool._get_cdp_override', return_value=""):
            with patch('tools.browser_tool._try_auto_detect_cdp', return_value="ws://127.0.0.1:9222"):
                with patch('tools.browser_tool._get_cloud_provider', return_value=None):
                    result = _get_session_info("test_task")
                    
                    assert result["cdp_url"] == "ws://127.0.0.1:9222"
                    assert result["features"] == {"cdp_override": True}

    def test_falls_back_to_cloud_when_no_cdp(self):
        """Should fall back to cloud provider when no CDP detected."""
        from tools.browser_tool import _get_session_info, _active_sessions

        _active_sessions.clear()

        mock_provider = Mock()
        mock_provider.create_session.return_value = {
            "session_name": "test_session",
            "bb_session_id": "bb_123",
            "cdp_url": "wss://browserbase.com/ws/123",
            "features": {"browserbase": True},
        }

        with patch('tools.browser_tool._get_cdp_override', return_value=""):
            with patch('tools.browser_tool._try_auto_detect_cdp', return_value=""):
                with patch('tools.browser_tool._get_cloud_provider', return_value=mock_provider):
                    result = _get_session_info("test_task")
                    
                    assert result["bb_session_id"] == "bb_123"
                    assert "browserbase" in result["features"]

    def test_falls_back_to_local_when_nothing_else(self):
        """Should fall back to local mode when no CDP or cloud."""
        from tools.browser_tool import _get_session_info, _active_sessions

        _active_sessions.clear()

        with patch('tools.browser_tool._get_cdp_override', return_value=""):
            with patch('tools.browser_tool._try_auto_detect_cdp', return_value=""):
                with patch('tools.browser_tool._get_cloud_provider', return_value=None):
                    result = _get_session_info("test_task")
                    
                    assert result["session_name"].startswith("h_")
                    assert result["features"] == {"local": True}

    def test_explicit_cdp_takes_priority_over_auto_detect(self):
        """Should use explicit CDP_URL even if auto-detect finds Chrome."""
        from tools.browser_tool import _get_session_info, _active_sessions

        _active_sessions.clear()

        with patch('tools.browser_tool._get_cdp_override', return_value="ws://custom:9999"):
            with patch('tools.browser_tool._try_auto_detect_cdp', return_value="ws://127.0.0.1:9222"):
                result = _get_session_info("test_task")
                
                assert result["cdp_url"] == "ws://custom:9999"
