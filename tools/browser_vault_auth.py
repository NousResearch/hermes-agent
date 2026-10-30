"""Native-auth fallback for browser_vault; never an alternate credential API."""


def _native_authorizer(task_id):
    from tools.browser_vault_tool import _bot_desktop_browser_session
    from hermes_constants import hermes_home_key
    home = hermes_home_key()
    desktop = _bot_desktop_browser_session(task_id)
    if desktop:
        from tools.bot_desktop import lease
        epoch = lease.assert_agent_may_act().epoch

    def authorize():
        if hermes_home_key() != home:
            raise RuntimeError("native auth profile changed")
        if desktop and lease.assert_agent_may_act().epoch != epoch:
            raise RuntimeError("browser control changed during native auth")

    return authorize


def fill_native_basic(task_id, meta, backend, handle, allowed):
    from agent.redact import register_vault_redaction_value
    from tools.browser_vault_tool import _ensure_supervisor, _bot_desktop_browser_session

    supervisor = _ensure_supervisor(task_id)
    if supervisor is None:
        return {"success": False, "error_type": "supervisor_required",
                "error": "Native HTTP authentication requires the supervised browser session."}

    authorize = _native_authorizer(task_id)

    def resolve(origin, transmit):
        from tools.browser_vault_context import snapshot_refusal
        # Keep manager snapshot/lock, revalidation, resolution and CDP write
        # on the same worker thread, inside the existing backend context.
        with backend.fill_context():
            refusal = snapshot_refusal(backend, handle, meta, origin)
            if refusal is not None:
                raise RuntimeError("vault item binding changed")
            if _bot_desktop_browser_session(task_id):
                from tools.bot_desktop import lease
                lease.assert_agent_may_act()
            password = backend.resolve_password(handle)
            register_vault_redaction_value(password)
            import base64
            encoded = base64.b64encode(f"{meta.identifier}:{password}".encode()).decode()
            register_vault_redaction_value(encoded)
            if _bot_desktop_browser_session(task_id):
                lease.assert_agent_may_act()
            transmit(meta.identifier, password)

    return supervisor.authenticate_vault_basic(allowed, resolve, authorize)
