"""Submit -> explicit registered tool -> published bytes -> real recipient staging."""
import base64
import json
import time
from types import SimpleNamespace

import pytest

from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from tools import hosted_room_artifact
from tools.registry import registry
from tui_gateway import classic_exports, server
from tui_gateway.transport import bind_transport, reset_transport
from tests.tui_gateway.test_prompt_accept_logging import _session, turn_env


@pytest.mark.parametrize("interrupted", [False, True])
def test_real_submit_publication_and_recipient_bytes(turn_env, tmp_path, monkeypatch, interrupted):
    home = tmp_path / "writer"
    home.mkdir()
    token = set_hermes_home_override(home)
    transport = SimpleNamespace(write=lambda frame: True)
    transport_token = bind_transport(transport)
    data = b"# welcome\n\x00\xff\xf0\x9f\x91\x8b"
    output = home / "welcome.bin"
    output.write_bytes(data)
    calls = []

    def inference(*args, **kwargs):
        calls.append(args)
        result = json.loads(registry.dispatch("share_group_file", {"path": str(output)}))
        assert result["ok"], result
        return {"final_response": "Shared welcome.bin", "interrupted": interrupted}

    agent = SimpleNamespace(session_id="writer-session", tools=[], valid_tool_names=set(), platform="desktop",
                            clear_interrupt=lambda: None, run_conversation=inference)
    session = _session(agent, profile_home=str(home), room_plumbing=True, transport=transport, cwd=str(home))
    monkeypatch.setitem(server._sessions, "writer-runtime", session)
    monkeypatch.setattr(server, "_start_agent_build", lambda *args: None)
    monkeypatch.setattr(server, "_session_cwd", lambda s: s["cwd"])
    classic_exports.install_schema(session)
    request = {"request_id": "natural-request", "issued_at": time.time(), "group_id": "workshop", "thread_id": "thread",
               "recipients": [{"installation": "reviewer-install", "profile": "reviewer"}]}
    try:
        response = server._methods["prompt.submit"](1, {"session_id": "writer-runtime", "text": "create and share", "classic_export": request})
        assert "error" not in response, response
        assert len(calls) == 1
        export_id = response["result"]["classic_export"]["export_id"]
        store = classic_exports.store_for(session)
        status = store.status(export_id)
        assert status["state"] == ("retired" if interrupted else "published"), status
        duplicate = server._methods["prompt.submit"](2, {"session_id": "writer-runtime", "text": "create and share", "classic_export": request})
        assert "error" not in duplicate
        assert len(calls) == 1
        assert "_classic_export_admission" not in session
        if not interrupted:
            item = status["items"][0]
            _, copied = store.read(export_id, item["artifact_id"])
            receiver_home = tmp_path / "other-installation" / "reviewer"
            receiver_home.mkdir(parents=True)
            receiver = _session(SimpleNamespace(), profile_home=str(receiver_home), transport=transport, cwd=str(receiver_home))
            monkeypatch.setitem(server._sessions, "reviewer-runtime", receiver)
            attached = server._methods["file.attach"](3, {"session_id": "reviewer-runtime", "name": item["name"],
                "data_url": f"data:{item['mime']};base64,{base64.b64encode(copied).decode()}"})
            assert "error" not in attached, attached
            from pathlib import Path
            staged = Path(attached["result"]["path"])
            assert staged.read_bytes() == data
            assert staged != output
    finally:
        reset_transport(transport_token)
        reset_hermes_home_override(token)


@pytest.mark.parametrize('failure', ['persistence', 'build_start', 'build_ready'])
def test_pre_run_failure_retires_admission_without_model_or_replay(turn_env, tmp_path, monkeypatch, failure):
    home = tmp_path / 'failed-writer'
    home.mkdir()
    token = set_hermes_home_override(home)
    transport = SimpleNamespace(write=lambda frame: True)
    transport_token = bind_transport(transport)
    calls = []
    agent = SimpleNamespace(session_id='failed-session', tools=[], valid_tool_names=set(), platform='desktop',
                            clear_interrupt=lambda: None, run_conversation=lambda *a, **kw: calls.append(a))
    session = _session(agent, profile_home=str(home), room_plumbing=True, transport=transport, cwd=str(home))
    monkeypatch.setitem(server._sessions, 'failed-runtime', session)
    monkeypatch.setattr(server, '_session_cwd', lambda value: value['cwd'])
    classic_exports.install_schema(session)
    if failure == 'persistence':
        monkeypatch.setattr(server, '_ensure_session_db_row', lambda _: False)
    elif failure == 'build_start':
        monkeypatch.setattr(server, '_restart_completed_failed_agent_build', lambda *a: False)
        def fail_build(*args):
            raise RuntimeError('forced builder startup failure')
        monkeypatch.setattr(server, '_start_agent_build', fail_build)
    else:
        monkeypatch.setattr(server, '_start_agent_build', lambda *a: None)
        monkeypatch.setattr(server, '_wait_agent_for_prompt', lambda *a: server._err(1, 5032, 'forced build failure'))
    request = {'request_id': f'failed-{failure}', 'issued_at': time.time(), 'group_id': 'workshop', 'thread_id': 'thread',
               'recipients': [{'installation': 'reviewer-install', 'profile': 'reviewer'}]}
    params = {'session_id': 'failed-runtime', 'text': 'create and share', 'classic_export': request}
    try:
        response = server._methods['prompt.submit'](1, params)
        assert 'error' in response or response['result']['status'] == 'streaming'
        store = classic_exports.store_for(session)
        row = store.prior(session['session_key'], request['request_id'])
        assert row['state'] == 'retired'
        assert not session['running']
        assert '_classic_export_admission' not in session
        repeated = server._methods['prompt.submit'](2, params)
        assert repeated['result']['classic_export']['state'] == 'retired'
        assert calls == []
    finally:
        reset_transport(transport_token)
        reset_hermes_home_override(token)
