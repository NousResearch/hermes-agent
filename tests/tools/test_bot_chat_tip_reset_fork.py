"""Discovery pins the receipt lineage's child after a local reset.

A local reset keeps the receipt on the creation id and advances its ``lineage`` to a
physical child, but ``get_compression_tip`` deliberately does not follow reset forks
(#114271). ``_bot_chat_tip`` therefore read the owner's receipt lineage first and falls
back to the compression walk when no receipt answers.
"""
import uuid
from contextlib import closing

from hermes_state import SessionDB
import hermes_state_runtime as rt
from tests.hermes_state.test_target_advance_fence import _local_session, _mutate
from tools.bot_live_delivery import _bot_chat_tip


def _titled_bot_chat(db, sid):
    db.set_session_title(sid, 'Bot Chat')


def test_reset_forked_bot_chat_pins_the_lineage_child(tmp_path):
    """After a reset the pin follows the receipt lineage's child, not the creation id."""
    with closing(SessionDB(tmp_path / 'state.db')) as db:
        epoch = rt.begin_runtime_epoch(db, instance_id='owner')
        sid = _local_session(db, epoch, tmp_path)
        _titled_bot_chat(db, sid)
        # get_compression_tip cannot see reset forks: it still names the creation id.
        assert db.get_compression_tip(sid) == sid
        child = _mutate(db, epoch, sid, 'reset')['target_session_id']
        assert _bot_chat_tip(tmp_path) == child


def test_no_receipt_falls_back_to_compression_tip(tmp_path):
    """Without a receipt the pre-patch compression walk is the answer."""
    with closing(SessionDB(tmp_path / 'state.db')) as db:
        sid = 'local-' + uuid.uuid4().hex
        db.create_session(session_id=sid, source='local')
        _titled_bot_chat(db, sid)
        assert _bot_chat_tip(tmp_path) == sid
