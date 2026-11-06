"""Bounded legacy stored listing: raw windows and eligible logical ranks are distinct."""
from pathlib import Path


def list_bounds(params):
    offset, limit = params.get('offset', 0), params.get('limit')
    limit = 200 if limit is None else limit
    if type(offset) is not int or not 0 <= offset <= 9007199254740991:
        raise ValueError('offset must be a safe nonnegative integer')
    if type(limit) is not int or not 1 <= limit <= 200:
        raise ValueError('limit must be an integer from 1 to 200')
    if params.get('title') and offset:
        raise ValueError('title lookup requires offset zero')
    return offset, limit


def stored_page(db, *, offset, limit, include_hidden, listing):
    from hermes_cli.session_listing import show_subagent_sessions
    from hermes_state_sessions import INTERNAL_LISTING_SOURCES
    db_path = getattr(db, 'db_path', None)
    include_subagents = bool(db_path) and show_subagent_sessions(Path(db_path).parent)
    page, seen, skipped, db_offset = [], set(), 0, 0
    while True:
        rows = listing(db, 200, offset=db_offset, filter_sources=False,
                       include_hidden=include_hidden, include_subagents=include_subagents)
        db_offset += len(rows)
        for row in rows:
            if (row.get('source') or '').strip().lower() in INTERNAL_LISTING_SOURCES:
                continue
            if row.get('source') == 'cron' and not row.get('message_count'):
                continue
            identity = row.get('_lineage_root_id') or row['id']
            if identity in seen:
                continue
            seen.add(identity)
            if skipped < offset:
                skipped += 1
            elif len(page) < limit:
                page.append(row)
            else:
                return page, True
        if len(rows) < 200:
            return page, False
