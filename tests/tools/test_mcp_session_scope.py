"""Real stdio registrations must not become sibling-session capabilities."""
import concurrent.futures
import sys


def test_session_mcp_registry_isolation_and_scoped_release(tmp_path):
    from tools.registry import registry, session_tool_scope
    from tools.mcp_tool_discovery import register_mcp_servers
    from tools.mcp_tool_lifecycle import shutdown_mcp_servers

    script = tmp_path / 'peer.py'
    script.write_text('''import json, os, sys
for line in sys.stdin:
 r=json.loads(line); method=r.get('method'); ident=r.get('id')
 if ident is None: continue
 if method=='initialize': result={'protocolVersion':r['params']['protocolVersion'],'capabilities':{'tools':{}},'serverInfo':{'name':'owned-peer','version':'1'}}
 elif method=='tools/list': result={'tools':[{'name':'echo','description':'owned echo','inputSchema':{'type':'object','properties':{}}}]}
 elif method=='tools/call': result={'content':[{'type':'text','text':os.environ['OWNED_MARKER']}]}
 else: result={}
 print(json.dumps({'jsonrpc':'2.0','id':ident,'result':result}),flush=True)
''')
    def connect(scope, name, marker):
        with session_tool_scope(scope):
            register_mcp_servers({name: {'command': sys.executable, 'args': [str(script)],
                'env': {'OWNED_MARKER': marker}, 'connect_timeout': 5}})
            names = [n for n in registry.get_all_tool_names() if n.startswith('mcp__')]
            assert len(names) == 1, names
            return names[0], registry.dispatch(names[0], {})
    try:
        with concurrent.futures.ThreadPoolExecutor(2) as pool:
            a = pool.submit(connect, 'editor-session:a', 'owned_a', 'ONLY_A')
            b = pool.submit(connect, 'editor-session:b', 'owned_b', 'ONLY_B')
            an, av = a.result(timeout=30)
            bn, bv = b.result(timeout=30)
        assert 'ONLY_A' in av and 'ONLY_B' not in av
        assert 'ONLY_B' in bv and 'ONLY_A' not in bv
        assert registry.get_entry(an) is None and registry.get_entry(bn) is None
        shutdown_mcp_servers(scope='editor-session:a')
        with session_tool_scope('editor-session:b'):
            assert registry.get_entry(an) is None
            assert 'ONLY_B' in registry.dispatch(bn, {})
    finally:
        shutdown_mcp_servers(scope='editor-session:a')
        shutdown_mcp_servers(scope='editor-session:b')
