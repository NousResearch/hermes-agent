import os
from unittest.mock import patch
import pytest

from tools.browser_providers.browserless import BrowserlessProvider

def test_is_configured_true():
    provider = BrowserlessProvider()
    with patch.dict(os.environ, {"BROWSERLESS_API_KEY": "test_key"}):
        assert provider.is_configured() is True

def test_is_configured_false():
    provider = BrowserlessProvider()
    with patch.dict(os.environ, clear=True):
        assert provider.is_configured() is False

def test_create_session():
    provider = BrowserlessProvider()
    with patch.dict(os.environ, {"BROWSERLESS_API_KEY": "test_key", "BROWSERLESS_API_URL": "http://localhost:3000"}):
        with patch("uuid.uuid4") as mock_uuid:
            mock_uuid.return_value.hex = "1234567890abcdef"
            session = provider.create_session("task123")
            
            assert session["session_name"] == "hermes_task123_12345678"
            assert session["bb_session_id"] == "hermes_task123_12345678"
            assert session["cdp_url"] == "ws://localhost:3000?token=test_key&timeout=300000&trackingId=hermes_task123_12345678"
            assert session["features"] == {"browserless": True}

def test_create_session_default_url():
    provider = BrowserlessProvider()
    with patch.dict(os.environ, {"BROWSERLESS_API_KEY": "test_key"}):
        with patch("uuid.uuid4") as mock_uuid:
            mock_uuid.return_value.hex = "1234567890abcdef"
            session = provider.create_session("task123")
            
            assert session["cdp_url"] == "wss://chrome.browserless.io?token=test_key&timeout=300000&trackingId=hermes_task123_12345678"

def test_create_session_no_key():
    provider = BrowserlessProvider()
    with patch.dict(os.environ, clear=True):
        with pytest.raises(ValueError, match="BROWSERLESS_API_KEY is required."):
            provider.create_session("task123")

def test_close_session():
    provider = BrowserlessProvider()
    assert provider.close_session("some_session") is True

def test_provider_name():
    provider = BrowserlessProvider()
    assert provider.provider_name() == "Browserless"
