"""Guarded extension of native vault fill; no model-facing schema or resolver."""

import json


def fill_masked_password(task_id, backend, handle, origin):
    from agent.redact import register_vault_redaction_value
    from agent.vault_backends import UnlockRequired
    from tools import browser_vault_tool

    def refused(reason):
        return json.dumps(
            {
                "success": False,
                "error_type": reason,
                "error": "JavaScript-masked password fill refused; do not retry or submit an uncertain fill.",
            }
        )

    # A plugin/preview page is a separate target; never switch transports.
    if browser_vault_tool._page_backend(task_id) is not None:
        return refused("unsupported_password_widget")
    supervisor = browser_vault_tool._ensure_supervisor(task_id)
    if supervisor is None:
        return refused("supervisor_required")
    prepared = supervisor.prepare_masked_password(origin)
    if not prepared.get("ok"):
        return refused(prepared.get("error_type", "unsupported_password_widget"))
    proof = prepared["proof"]
    try:
        checked = supervisor.check_masked_password(proof)
        if not checked.get("ok") or checked.get("result", {}).get("ready") is not True:
            return refused("widget_context_changed")
        try:
            password = backend.resolve_password(handle)
        except UnlockRequired:
            return refused("unlock_required")
        if not password:
            return refused("empty_password")
        register_vault_redaction_value(password)
        if browser_vault_tool._bot_desktop_browser_session(task_id):
            from tools.bot_desktop import lease

            try:
                lease.assert_agent_may_act()
            except lease.HumanHasControl:
                return refused("human_has_control")
        result = supervisor.fill_masked_password(proof, password)
        if not result.get("ok"):
            return refused(result.get("error_type", "widget_delivery_uncertain"))
        delivered = result.get("result", {})
        if delivered.get("filled") != 1:
            return refused(delivered.get("refused", "widget_delivery_uncertain"))
        return json.dumps(
            {
                "success": True,
                "filled_fields": 1,
                "backend": backend.name,
                "kind": "login",
                "origin": origin,
                "next": "Submit normally only if the page still shows the intended sign-in.",
            }
        )
    except Exception:
        # Neither provider failures nor site exceptions may echo a value.
        return refused("widget_fill_failed")
    finally:
        supervisor.dispose_masked_password(proof)
