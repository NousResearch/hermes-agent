"""MCP client: connects the configured MCP servers and registers their tools.

This file imports nothing: callers import the module they need, so loading one module does not
load the rest. Modules:

- ``client``: the origin module (shared connection state, ``MCPServerTask``, SDK loading); the
  others read it back at call time through ``common._OriginProxy``.
- ``config``: server config loading, ``${VAR}`` interpolation, the stdio launch environment.
- ``discovery``: connect and discover (``register_mcp_servers``, ``discover_mcp_tools``), lazy start.
- ``registration``: a server's tools into the tool registry, with trust metadata.
- ``handlers``: the registry-facing call handlers and the per-call recovery ladder.
- ``schema`` / ``schema_cache``: schema normalisation and tool naming; the per-server schema cache.
- ``content``: result content blocks rendered as model-facing text.
- ``transport``: stdio spawn (after ``osv_check``), Streamable HTTP and SSE bring-up.
- ``server_run``: the long-lived ``MCPServerTask.run`` state machine.
- ``health`` / ``liveness``: keepalive, ``list_changed`` refresh; liveness parsing and descriptions.
- ``lifecycle`` / ``loop``: stdio child tracking and shutdown; the background event loop.
- ``errors``: connection and transport error classification, URL validation.
- ``sampling`` / ``elicitation_form``: server-initiated sampling and elicitation requests.
- ``events``: the MCP Events client and webhook delivery.
- ``costs``: declared per-tool costs and the per-chat spend tally.
- ``scope``: connection-ledger keys under a profile multiplexer.
- ``node_abi``: stdio servers whose native addon was built for another Node.js.
- ``oauth`` / ``oauth_manager`` / ``oauth_provider`` / ``oauth_device`` / ``dashboard_oauth``: the
  OAuth 2.1 client, per-server OAuth state, SDK provider customisations, device login, and the
  dashboard callback bridge.
- ``osv_check``: the OSV malware check on a package before a stdio spawn.
- ``common``: small helpers shared by the modules above.
"""
