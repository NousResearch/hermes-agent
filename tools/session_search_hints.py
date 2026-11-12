"""Taught session_search discovery form, shared by the tool schema and compaction footers.

Leaf module with no imports: ``agent.context_compressor`` needs these strings, and importing
``tools.session_search_tool`` from there closes an import cycle through ``hermes_state_common``.
Current-session context is implicit; passing session_id enters READ and ignores query.
"""

SESSION_SEARCH_DISCOVERY_CALL = (
    "session_search(query='<keywords>', role_filter='user,assistant,tool')"
)
SESSION_SEARCH_DISCOVERY_HINT = (
    "discovery form; current-session context is implicit. Do not pass "
    "session_id because that enters READ and ignores query."
)
