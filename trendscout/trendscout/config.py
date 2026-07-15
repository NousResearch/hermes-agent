#!/usr/bin/env python3
"""Config loading for trendscout — config.yaml + plain-text source lists."""

from pathlib import Path

import yaml

# Import the canonical Hermes home resolver — context-var-aware, profile-aware
from hermes_constants import get_hermes_home


def load_config(path: str | Path = None) -> dict:
    path = Path(path) if path else PROJECT_ROOT / 'config' / 'config.yaml'
    with open(path) as f:
        config = yaml.safe_load(f)

    # Resolve paths relative to Hermes home (profile-aware)
    hermes_home = get_hermes_home()
    paths = config.setdefault('paths', {})
    
    # Default paths if not specified
    if 'db' not in paths:
        paths['db'] = str(hermes_home / 'trendscout' / 'trendscout.db')
    if 'chroma' not in paths:
        paths['chroma'] = str(hermes_home / 'trendscout' / 'chroma')
    
    # Resolve any explicit paths (expanduser for backward compat)
    for key, value in list(paths.items()):
        if key.endswith('_file'):
            # Source list files relative to project root
            paths[key] = str(PROJECT_ROOT / value)
        else:
            # DB/chroma paths: expanduser only (not profile-aware overrides)
            paths[key] = str(Path(value).expanduser())
    
    # Thread Firecrawl API URL from config into environment for firecrawl_client
    # This ensures localhost:3002 (or custom) is used consistently
    firecrawl_cfg = config.get('firecrawl', {})
    if firecrawl_cfg.get('enabled'):
        api_url = firecrawl_cfg.get('api_url', 'http://localhost:3002')
        import os
        os.environ.setdefault('FIRECRAWL_API_URL', api_url)

    return config


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _read_lines(path: str | Path) -> list[str]:
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        return []
    lines = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith('#'):
            lines.append(line)
    return lines


def load_subreddits(config: dict) -> list[str]:
    return _read_lines(config['paths']['subreddits_file'])


def load_urls(config: dict) -> list[str]:
    return _read_lines(config['paths']['urls_file'])
