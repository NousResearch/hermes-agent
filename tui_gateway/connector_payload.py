"""Connector metadata egress, preserving authorization links and future fields."""

from agent.redact import _key_has_secret_keyword, redact_sensitive_text


def connector_ui_payload(value):
    if isinstance(value, dict):
        return {key: ("[REDACTED]" if (_key_has_secret_keyword(key)
                      or key.lower() in {"authorization", "proxy-authorization", "cookie", "set-cookie", "credentials"})
                      else item if key == "connect_url" and isinstance(item, str)
                      else connector_ui_payload(item)) for key, item in value.items()}
    if isinstance(value, list):
        return [connector_ui_payload(item) for item in value]
    return redact_sensitive_text(value, force=True, redact_url_credentials=True) if isinstance(value, str) else value
