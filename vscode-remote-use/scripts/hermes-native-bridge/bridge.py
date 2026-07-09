from __future__ import annotations
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from lib_gate import Progress, default_gate_check, run_when_passed


class HermesNativeBridge:
    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root else Path(__file__).resolve().parent
        self.apps_dir = self.root / 'apps'
        self.bridge_log = Path(tempfile.gettempdir()) / '.hermes-native-bridge.log'

    def available_apps(self) -> list[dict[str, object]]:
        apps: list[dict[str, object]] = []
        if not self.apps_dir.exists():
            return apps
        for candidate in sorted(self.apps_dir.iterdir()):
            if not candidate.is_dir():
                continue
            manifest = candidate / 'app.json'
            if not manifest.exists():
                continue
            try:
                data = json.loads(manifest.read_text(encoding='utf-8'))
            except Exception:
                continue
            data.setdefault('name', candidate.name)
            data.setdefault('path', str(candidate))
            apps.append(data)
        return apps

    def run_app(self, app: str, payload: dict | None = None, timeout: int = 20) -> dict:
        target = self.apps_dir / app
        manifest = target / 'app.json'
        if not target.exists() or not target.is_dir() or not manifest.exists():
            return {
                'ok': False,
                'error': f'unknown_app:{app}',
                'available': [x.get('name', '') for x in self.available_apps()],
            }

        data = json.loads(manifest.read_text(encoding='utf-8'))
        entry = data.get('entry')
        if not entry:
            return {'ok': False, 'error': 'missing_entry', 'app': app}

        payload = payload or {}
        progress_path = str(Path(tempfile.gettempdir()) / f'.hermes-native-{app}.progress.json')

        @run_when_passed(progress_path, model=app)
        def _run() -> dict:
            script = target / entry
            if not script.exists():
                return {'ok': False, 'error': f'missing_script:{entry}'}
            try:
                completed = subprocess.run(
                    [sys.executable, str(script)],
                    input=json.dumps(payload, ensure_ascii=True),
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=timeout,
                )
            except subprocess.TimeoutExpired:
                return {'ok': False, 'error': 'timeout', 'timeout': timeout}
            except Exception as e:  # pragma: no cover
                return {'ok': False, 'error': f'spawn_failed:{e}'}

            out = (completed.stdout or '').strip()
            err = (completed.stderr or '').strip()
            try:
                return json.loads(out) if out else {'ok': False, 'error': 'empty_output', 'stderr': err}
            except json.JSONDecodeError:
                return {'ok': completed.returncode == 0, 'raw': out, 'stderr': err}

        return _run()


def main() -> int:
    bridge = HermesNativeBridge()
    if len(sys.argv) > 1 and sys.argv[1] == '--self-test':
        passed, detail = default_gate_check()
        print(json.dumps({
            'ok': True,
            'command': 'self_test',
            'gate': {'passed': passed, 'detail': detail},
            'apps': bridge.available_apps(),
        }, ensure_ascii=True))
        return 0

    message = json.loads(sys.stdin.read() or '{}') if len(sys.argv) == 1 else {'command': 'list_apps'}
    if message.get('command') == 'list_apps':
        result = {'ok': True, 'command': 'list_apps', 'apps': bridge.available_apps()}
    elif message.get('command') == 'run_app':
        result = bridge.run_app(message.get('app') or '', message.get('payload') or {})
    else:
        result = {'ok': False, 'error': f"unknown_command:{message.get('command')}", 'supported': ['list_apps', 'run_app']}
    print(json.dumps(result, ensure_ascii=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
