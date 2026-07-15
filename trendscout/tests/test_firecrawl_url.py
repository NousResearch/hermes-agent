"""
Tests for firecrawl_client URL resolution at call time.

Verifies the fix for the import-order bug: FIRECRAWL_URL was read at module
import (before config.load_config() ran), so it was always None in real runs.

Now scrape() resolves the URL at call time with precedence:
1. Explicit api_url argument
2. FIRECRAWL_API_URL environment variable
3. Default fallback: http://localhost:3002
"""
import os
from unittest.mock import patch

from trendscout import firecrawl_client


def test_scrape_uses_explicit_api_url():
    """Verify scrape() uses the explicit api_url argument when provided."""
    # Ensure env var is NOT set — simulates the import-order bug scenario
    old_val = os.environ.pop('FIRECRAWL_API_URL', None)
    
    try:
        # Import happens at module level (before this test runs)
        # The bug was that FIRECRAWL_URL was None here and never re-read
        
        # Call with explicit api_url — should use it, not None/default
        custom_url = 'http://custom-firecrawl:3002'
        
        # Mock urlopen to capture the URL called
        with patch('urllib.request.urlopen') as mock_urlopen:
            mock_urlopen.return_value.__enter__ = lambda s: s
            mock_urlopen.return_value.__exit__ = lambda s, *args: None
            mock_urlopen.return_value.read.return_value = b'{"success": true, "data": {"markdown": "test"}}'
            
            firecrawl_client.scrape('http://example.com', api_url=custom_url)
            
            # Verify the request was made to the custom URL
            call_args = mock_urlopen.call_args
            req = call_args[0][0]  # First positional arg is the Request object
            assert req.full_url == f'{custom_url}/v1/scrape', f"Expected {custom_url}/v1/scrape, got {req.full_url}"
        
        print(f"✓ Explicit api_url used: {custom_url}")
    finally:
        # Restore
        if old_val:
            os.environ['FIRECRAWL_API_URL'] = old_val


def test_scrape_uses_env_var():
    """Verify scrape() uses FIRECRAWL_API_URL env var when no explicit api_url."""
    old_val = os.environ.pop('FIRECRAWL_API_URL', None)
    
    try:
        env_url = 'http://env-firecrawl:3002'
        os.environ['FIRECRAWL_API_URL'] = env_url
        
        with patch('urllib.request.urlopen') as mock_urlopen:
            mock_urlopen.return_value.__enter__ = lambda s: s
            mock_urlopen.return_value.__exit__ = lambda s, *args: None
            mock_urlopen.return_value.read.return_value = b'{"success": true, "data": {"markdown": "test"}}'
            
            firecrawl_client.scrape('http://example.com')
            
            call_args = mock_urlopen.call_args
            req = call_args[0][0]
            assert req.full_url == f'{env_url}/v1/scrape', f"Expected {env_url}/v1/scrape, got {req.full_url}"
        
        print(f"✓ FIRECRAWL_API_URL env var used: {env_url}")
    finally:
        if old_val:
            os.environ['FIRECRAWL_API_URL'] = old_val
        else:
            os.environ.pop('FIRECRAWL_API_URL', None)


def test_scrape_uses_default_fallback():
    """Verify scrape() falls back to localhost:3002 when no api_url or env var."""
    old_val = os.environ.pop('FIRECRAWL_API_URL', None)
    
    try:
        with patch('urllib.request.urlopen') as mock_urlopen:
            mock_urlopen.return_value.__enter__ = lambda s: s
            mock_urlopen.return_value.__exit__ = lambda s, *args: None
            mock_urlopen.return_value.read.return_value = b'{"success": true, "data": {"markdown": "test"}}'
            
            firecrawl_client.scrape('http://example.com')
            
            call_args = mock_urlopen.call_args
            req = call_args[0][0]
            expected = f'{firecrawl_client.DEFAULT_FIRECRAWL_URL}/v1/scrape'
            assert req.full_url == expected, f"Expected {expected}, got {req.full_url}"
        
        print(f"✓ Default fallback used: {firecrawl_client.DEFAULT_FIRECRAWL_URL}")
    finally:
        if old_val:
            os.environ['FIRECRAWL_API_URL'] = old_val


def test_scrape_api_key_resolution():
    """Verify api_key argument and env var resolution for authentication."""
    old_key = os.environ.pop('FIRECRAWL_API_KEY', None)
    
    try:
        custom_key = 'test-secret-key'
        
        with patch('urllib.request.urlopen') as mock_urlopen:
            mock_urlopen.return_value.__enter__ = lambda s: s
            mock_urlopen.return_value.__exit__ = lambda s, *args: None
            mock_urlopen.return_value.read.return_value = b'{"success": true, "data": {"markdown": "test"}}'
            
            firecrawl_client.scrape('http://example.com', api_key=custom_key)
            
            call_args = mock_urlopen.call_args
            req = call_args[0][0]
            auth_header = req.get_header('Authorization')
            assert auth_header == f'Bearer {custom_key}', f"Expected 'Bearer {custom_key}', got {auth_header}"
        
        print(f"✓ Explicit api_key used")
    finally:
        if old_key:
            os.environ['FIRECRAWL_API_KEY'] = old_key


if __name__ == '__main__':
    test_scrape_uses_explicit_api_url()
    test_scrape_uses_env_var()
    test_scrape_uses_default_fallback()
    test_scrape_api_key_resolution()
    print("\n✅ All firecrawl URL resolution tests passed")
