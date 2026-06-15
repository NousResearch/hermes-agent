# Utils Module — Stateless utility functions
from utils.text_processor import (
    strip_think_blocks,
    extract_first_line,
    normalize_code_blocks,
    strip_ansi_escape,
    build_system_prompt,
    join_messages,
    sanitize_api_response_text,
    strip_whitespace_edges,
    collapse_empty_lines,
    estimate_tokens_rough,
    classify_truncated_response,
    extract_error_reason,
)

# Import functions from parent utils.py module
import importlib.util
_spec = importlib.util.spec_from_file_location("utils._utils", f"{__path__[0]}/../utils.py")
_utils = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_utils)

atomic_json_write = _utils.atomic_json_write
atomic_replace = _utils.atomic_replace
atomic_yaml_write = _utils.atomic_yaml_write
safe_json_loads = _utils.safe_json_loads
is_truthy_value = _utils.is_truthy_value
env_var_enabled = _utils.env_var_enabled
env_int = _utils.env_int
env_bool = _utils.env_bool
base_url_hostname = _utils.base_url_hostname
base_url_host_matches = _utils.base_url_host_matches
normalize_proxy_url = _utils.normalize_proxy_url
normalize_proxy_env_vars = _utils.normalize_proxy_env_vars

__all__ = [
    "strip_think_blocks",
    "extract_first_line",
    "normalize_code_blocks",
    "strip_ansi_escape",
    "build_system_prompt",
    "join_messages",
    "sanitize_api_response_text",
    "strip_whitespace_edges",
    "collapse_empty_lines",
    "estimate_tokens_rough",
    "classify_truncated_response",
    "extract_error_reason",
    "atomic_json_write",
    "atomic_replace",
    "atomic_yaml_write",
    "safe_json_loads",
    "is_truthy_value",
    "env_var_enabled",
    "env_int",
    "env_bool",
    "base_url_hostname",
    "base_url_host_matches",
    "normalize_proxy_url",
    "normalize_proxy_env_vars",
]