"""Compact model references resolve in the native desktop/TUI resume path."""
from types import SimpleNamespace
from hermes_state import SessionDB
from tui_gateway import server


def test_resume_resolves_compact_reference_in_owning_store(tmp_path):
    stores = [SessionDB(db_path=tmp_path / name / 'state.db') for name in ('a', 'b')]
    try:
        for index, db in enumerate(stores):
            db.create_session(f'profile-{index}-123456789abc', source='cli')
        for index in (0, 1, 0):
            ctx = SimpleNamespace(db=stores[index], target='~123456789abc', rid=1)
            assert server._resume_locate(ctx) is None
            assert ctx.target == f'profile-{index}-123456789abc'
            assert ctx.found['id'] == ctx.target
        stores[0].create_session('collision-123456789abc', source='cli')
        ctx = SimpleNamespace(db=stores[0], target='~123456789abc', rid=2)
        assert 'ambiguous' in server._resume_locate(ctx)['error']['message']
    finally:
        for db in stores:
            db.close()
