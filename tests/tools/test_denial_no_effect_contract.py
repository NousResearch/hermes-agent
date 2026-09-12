"""Literal unknown-selector no-effects contract, including stale UI bookkeeping.

No waiting consumer exists in this boundary case; failure is not an approval bypass.
"""
import asyncio
from types import SimpleNamespace as NS
import pytest
from gateway.slash_commands import GatewaySlashCommandsMixin
from gateway.session import SessionSource
from gateway.config import Platform
from tools import approval as ap

@pytest.mark.parametrize('args',['0'*32,'unknown-id'],ids=['unknown-exact-id','malformed-id'])
def test_unrecognized_deny_selector_preserves_stale_ui_state(tmp_path,monkeypatch,args):
    monkeypatch.setenv('HERMES_HOME',str(tmp_path))
    ap._gateway_queues.clear()
    class Slash(GatewaySlashCommandsMixin):
        def _session_key_for_source(self,source):return 'fixture-stale-only'
        async def _deliver_approval_confirmation(self,*args):raise AssertionError('no confirmation expected')
    slash=Slash();record={'command':'inert obsolete UI prompt'}
    slash._pending_approvals={'fixture-stale-only':record}
    event=NS(source=SessionSource(platform=Platform.TELEGRAM,chat_id='fixture',user_id='fixture'),get_command_args=lambda:args)
    asyncio.run(slash._handle_deny_command(event))
    assert not ap._gateway_queues
    assert slash._pending_approvals=={'fixture-stale-only':record}, 'unknown selector removed a different stale UI record'
