"""One-operation human consent for plugin policies, without persistent grants."""


def grant_human_choice(choice, *, spec, mandatory, smart_denied, session_key,
                       pattern_key, pattern_keys, description):
    from tools import approval as a

    if mandatory and choice != "once":
        return a._denied("BLOCKED: this action requires one-operation human consent",
                         pattern_key=pattern_key, description=description, outcome="denied", noun=spec.noun)
    if not smart_denied and not mandatory:
        a._persist_choice(session_key, choice, pattern_keys)
    if spec.user_approved:
        return a._user_approved(session_key, description)
    return a._approved()


def request_mandatory_tool_approval(tool_name, description, rule_key, effective_args, approval_callback):
    import json
    from agent.redact import redact_sensitive_text
    from tools import approval as a

    callback, is_cli, is_gateway, is_ask = a._presence(approval_callback)
    if not (is_cli or is_gateway or is_ask):
        return a._blocked("BLOCKED: mandatory approval requires a human channel",
                          pattern_key=rule_key, description=description)
    return a._human_decision(
        a._ACTION_GATE, command=redact_sensitive_text(json.dumps(
            {"tool": tool_name, "args": effective_args or {}}, ensure_ascii=False, allow_nan=False), force=True),
        description=redact_sensitive_text(description, force=True), pattern_key=f"plugin_rule:{rule_key}",
        pattern_keys=[f"plugin_rule:{rule_key}"], session_key=a.get_current_session_key(),
        approval_callback=callback, is_cli=is_cli, is_gateway=is_gateway, is_ask=is_ask, mandatory=True,
    )
