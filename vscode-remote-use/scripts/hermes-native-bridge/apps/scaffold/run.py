from __future__ import annotations
import json
import sys
import time
from pathlib import Path


def run(payload: dict) -> dict:
    prompt = (payload.get('prompt') or '').strip() or 'Hello from Hermes Native local app.'
    return {
        'app': 'scaffold',
        'ok': True,
        'input': prompt,
        'output': prompt.upper(),
        'meta': {'echo': True, 'timestamp': time.time()}
    }


def main() -> int:
    payload = json.loads(sys.stdin.read() or '{}')
    result = run(payload)
    print(json.dumps(result, ensure_ascii=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
