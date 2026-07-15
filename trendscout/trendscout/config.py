#!/usr/bin/env python3
"""Config loading for trendscout — config.yaml + plain-text source lists."""

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | Path = None) -> dict:
    path = Path(path) if path else PROJECT_ROOT / 'config' / 'config.yaml'
    with open(path) as f:
        config = yaml.safe_load(f)

    paths = config.setdefault('paths', {})
    for key, value in paths.items():
        paths[key] = str(Path(value).expanduser())

    return config


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
