"""
Tests for trendscout config path resolution and Firecrawl URL threading.
"""
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import yaml

from hermes_constants import get_hermes_home, set_hermes_home_override, reset_hermes_home_override
from trendscout import config as cfg


def test_paths_default_to_hermes_home():
    """Verify db and chroma paths default to profile-aware Hermes home."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a minimal config without explicit paths
        config_path = Path(tmpdir) / 'config.yaml'
        config_path.write_text(yaml.dump({
            'reddit': {'enabled': True},
            'firecrawl': {'enabled': False},
        }))
        
        # Load config
        loaded = cfg.load_config(config_path)
        
        # Paths should default to Hermes home
        hermes_home = get_hermes_home()
        assert loaded['paths']['db'] == str(hermes_home / 'trendscout' / 'trendscout.db')
        assert loaded['paths']['chroma'] == str(hermes_home / 'trendscout' / 'chroma')
        print("✓ Default paths point to Hermes home")


def test_firecrawl_url_not_mutated_in_config():
    """Verify config.load_config() does NOT mutate FIRECRAWL_API_URL env var.
    
    The old implementation set os.environ.setdefault() during config load,
    which caused import-order bugs. Now the URL is passed explicitly at call
    time to firecrawl_client.scrape().
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / 'config.yaml'
        custom_url = 'http://custom-firecrawl:3002'
        config_path.write_text(yaml.dump({
            'firecrawl': {'enabled': True, 'api_url': custom_url},
        }))
        
        # Clear env var before loading config
        old_val = os.environ.pop('FIRECRAWL_API_URL', None)
        
        try:
            loaded = cfg.load_config(config_path)
            # Env var should NOT be set — URL is passed at call time instead
            assert os.environ.get('FIRECRAWL_API_URL') is None
            # But the config should still have the value for explicit passing
            assert loaded['firecrawl']['api_url'] == custom_url
            print(f"✓ Config does not mutate env var (URL passed at call time)")
        finally:
            if old_val:
                os.environ['FIRECRAWL_API_URL'] = old_val


def test_profile_aware_path_resolution():
    """Verify paths respect HERMES_HOME override (profile switching)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        profile_home = Path(tmpdir) / 'profile-test'
        profile_home.mkdir()
        
        # Set override
        token = set_hermes_home_override(profile_home)
        
        try:
            config_path = profile_home / 'config.yaml'
            config_path.write_text(yaml.dump({
                'reddit': {'enabled': True},
                'firecrawl': {'enabled': False},
            }))
            
            loaded = cfg.load_config(config_path)
            
            # Should use the override profile home, not default
            assert loaded['paths']['db'] == str(profile_home / 'trendscout' / 'trendscout.db')
            assert loaded['paths']['chroma'] == str(profile_home / 'trendscout' / 'chroma')
            print(f"✓ Profile-aware path resolution: {profile_home}")
        finally:
            reset_hermes_home_override(token)


def test_source_files_relative_to_project_root():
    """Verify subreddits_file and urls_file resolve relative to project root."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / 'config.yaml'
        config_path.write_text(yaml.dump({
            'paths': {
                'subreddits_file': 'config/subreddits.txt',
                'urls_file': 'config/urls.txt',
            },
            'reddit': {'enabled': True},
            'firecrawl': {'enabled': False},
        }))
        
        # Create dummy source files
        (Path(tmpdir) / 'config').mkdir()
        (Path(tmpdir) / 'config' / 'subreddits.txt').write_text('test\n')
        (Path(tmpdir) / 'config' / 'urls.txt').write_text('https://example.com\n')
        
        loaded = cfg.load_config(config_path)
        
        # Should be absolute paths under project root
        assert Path(loaded['paths']['subreddits_file']).is_absolute()
        assert Path(loaded['paths']['urls_file']).is_absolute()
        print("✓ Source file paths resolved relative to project root")


if __name__ == '__main__':
    test_paths_default_to_hermes_home()
    test_firecrawl_url_not_mutated_in_config()
    test_profile_aware_path_resolution()
    test_source_files_relative_to_project_root()
    print("\n✅ All config tests passed")
