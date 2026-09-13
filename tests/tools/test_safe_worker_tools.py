"""Fresh-interpreter proof of the safe worker's reviewed tool import closure."""
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest


def run_worker(tmp_path, body, mode="safe"):
    home = tmp_path / mode
    home.mkdir(exist_ok=True)
    root = Path(__file__).resolve().parents[2]
    env = dict(os.environ, HERMES_HOME=str(home), HOME=str(home), PROBE_MODE=mode,
               TERMINAL_ENV="local", TERMINAL_CWD=str(home))
    for key in ("HERMES_SAFE_MODE", "HERMES_IGNORE_USER_CONFIG", "HERMES_IGNORE_RULES"):
        env.pop(key, None)
    script = home / "probe.py"
    script.write_text(f"import sys; sys.path.insert(0, {str(root)!r})\n" + textwrap.dedent(body), encoding="utf-8")
    result = subprocess.run([sys.executable, str(script)], cwd=root, env=env,
                            stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("mode", ["safe", "config", "ordinary"])
def test_actual_agent_tool_round_never_loads_plugins_in_safe_worker(tmp_path, mode):
    result = run_worker(tmp_path, r'''
        import json, os, sys, threading
        from pathlib import Path
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        home = Path(os.environ['HERMES_HOME'])
        mode = os.environ['PROBE_MODE']
        plugin = home / 'plugins' / 'sentinel'
        plugin.mkdir(parents=True)
        (plugin / 'plugin.yaml').write_text('name: sentinel\nversion: 1.0.0\nkind: standalone\n')
        (plugin / '__init__.py').write_text("from pathlib import Path\nimport os\nPath(os.environ['HERMES_HOME'], 'plugin-executed').touch()\ndef register(ctx): pass\n")
        config = {'plugins': {'enabled': ['sentinel']}, 'agent': {'environment_probe': False},
                  'auxiliary': {'title_generation': {'enabled': False}},
                  'terminal': {'backend': 'local', 'cwd': str(home)}}
        (home / 'config.yaml').write_text(json.dumps(config))
        if mode != 'ordinary':
            from agent.safe_worker_policy import _bind_safe_worker_policy
            _bind_safe_worker_policy(safe_mode=mode == 'safe', ignore_user_config=True, config=config)
        reads, executions = [], []
        def audit(event, args):
            if event == 'open' and '/plugins/' in str(args[0]) and args[1] != 'w':
                frame = sys._getframe(); stack = []
                while frame:
                    if 'importlib' not in frame.f_code.co_filename:
                        stack.append((frame.f_code.co_filename, frame.f_lineno))
                    frame = frame.f_back
                reads.append((str(args[0]), stack))
            if event == 'exec' and '/plugins/' in str(getattr(args[0], 'co_filename', '')):
                executions.append(args[0].co_filename)
        sys.addaudithook(audit)
        from run_agent import AIAgent
        requests = []
        class Peer(BaseHTTPRequestHandler):
            def log_message(self, *args): pass
            def do_POST(self):
                request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                messages = request.get('messages', [])
                requests.append(request)
                results = [m for m in messages if m['role'] == 'tool']
                message = {'role': 'assistant', 'content': 'CORE_ROUND_DONE'}
                if not results:
                    message = {'role': 'assistant', 'content': None, 'tool_calls': [{
                        'id': 'terminal-fixture', 'type': 'function', 'function': {
                        'name': 'terminal', 'arguments': json.dumps({'command': 'printf CORE_EXECUTED',
                        'workdir': str(home)})}}]}
                tool = 'tool_calls' in message
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream' if request.get('stream') else 'application/json')
                self.end_headers()
                if request.get('stream'):
                    if tool: message['tool_calls'][0]['index'] = 0
                    for delta, finish in [(message, None), ({}, 'tool_calls' if tool else 'stop')]:
                        frame = {'id': 'local', 'object': 'chat.completion.chunk', 'created': 1,
                            'model': 'fixture', 'choices': [{'index': 0, 'delta': delta, 'finish_reason': finish}]}
                        self.wfile.write(('data: ' + json.dumps(frame) + '\n\n').encode())
                    self.wfile.write(b'data: [DONE]\n\n')
                else:
                    self.wfile.write(json.dumps({'id': 'local', 'object': 'chat.completion', 'created': 1,
                        'model': 'fixture', 'choices': [{'index': 0, 'message': message,
                        'finish_reason': 'tool_calls' if tool else 'stop'}]}).encode())
                self.wfile.flush()
        server = ThreadingHTTPServer(('127.0.0.1', 0), Peer)
        thread = threading.Thread(target=server.serve_forever); thread.start()
        try:
            agent = AIAgent(model='fixture', provider='custom',
                base_url=f'http://127.0.0.1:{server.server_port}/v1', api_key='fixture',
                enabled_toolsets=['terminal', 'file'], quiet_mode=True, save_trajectories=False,
                max_iterations=3)
            result = agent.run_conversation('Run the core terminal fixture.')
            assert result['final_response'] == 'CORE_ROUND_DONE', result
            tool_results = [m for m in result['messages'] if m['role'] == 'tool']
            assert tool_results and 'CORE_EXECUTED' in tool_results[0]['content'], tool_results
            assert json.loads(tool_results[0]['content'])['exit_code'] == 0, tool_results
            names = {t['function']['name'] for r in requests if r.get('messages') for t in r.get('tools', [])}
            assert {'terminal', 'read_file', 'write_file', 'patch', 'search_files'} <= names, names
            from tools.registry import discover_builtin_tools
            # No ContextVar propagation: the process-bound policy must still apply.
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(1) as pool:
                imported = pool.submit(discover_builtin_tools).result()
            if mode == 'safe':
                assert not reads, reads
                assert not executions, executions
                assert not (home / 'plugin-executed').exists()
            else:
                assert (home / 'plugin-executed').exists()
                assert reads and executions
            print(json.dumps({'mode': mode, 'plugin_reads': len(reads),
                'plugin_execs': len(executions), 'tool_result': tool_results[0]['content'],
                'helper_modules': imported}))
        finally:
            server.shutdown(); server.server_close(); thread.join()
    ''', mode)
    assert 'CORE_EXECUTED' in result['tool_result']
    if mode == 'safe':
        assert result['plugin_reads'] == result['plugin_execs'] == 0
