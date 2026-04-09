"""
Kimi-K2 JSON Sanitizer

Handles malformed JSON from moonshotai/kimi-k2-instruct and similar models
that produce truncated/invalid JSON in tool call arguments.

Issue: https://github.com/NousResearch/hermes-agent/issues/XXX
"""

import json
import re
import logging

logger = logging.getLogger(__name__)

# Models known to produce malformed JSON
KIMI_MODELS = {
    "moonshotai/kimi-k2-instruct",
    "moonshotai/kimi-k1.5-instruct",
    "kimi-k2-instruct",
    "kimi-k1.5-instruct",
}


def is_kimi_model(model: str) -> bool:
    """Check if the model is a Kimi variant that needs JSON repair."""
    if not model:
        return False
    model_lower = model.lower()
    return any(k.lower() in model_lower for k in KIMI_MODELS)


def sanitize_kimi_json(raw_args: str) -> tuple[dict | None, str | None]:
    """
    Attempt to repair malformed JSON from Kimi models.
    
    Args:
        raw_args: The raw JSON string from the model
        
    Returns:
        (repaired_dict, None) on success
        (None, error_message) on failure
    """
    if not raw_args:
        return {}, None
        
    original = raw_args
    
    # Pattern 1: Unterminated string at start (char 1 error)
    # Often the opening quote is there but closing quote is missing
    if raw_args.startswith('{"') and raw_args.count('"') % 2 != 0:
        # Try adding closing quote
        raw_args = raw_args + '"'
    
    # Pattern 2: Missing closing braces
    open_braces = raw_args.count('{') - raw_args.count('}')
    open_brackets = raw_args.count('[') - raw_args.count(']')
    
    if open_braces > 0:
        raw_args = raw_args + '}' * open_braces
    if open_brackets > 0:
        raw_args = raw_args + ']' * open_brackets
    
    # Pattern 3: Truncated mid-value (e.g., "command": "ls -la", "time)
    # Try to complete truncated keys
    truncated_key_pattern = r',\s*"[^"]*":\s*[^,}]*$'
    if re.search(truncated_key_pattern, raw_args):
        # Remove the truncated key-value pair
        raw_args = re.sub(truncated_key_pattern, '', raw_args)
        # Re-add closing brace if needed
        if not raw_args.endswith('}'):
            raw_args = raw_args + '}'
    
    # Pattern 4: Empty tool name field (becomes "")
    # This is handled at the tool dispatch level, not here
    
    try:
        parsed = json.loads(raw_args)
        if parsed != original:
            logger.info(f"Kimi JSON sanitizer repaired: {original[:50]}... -> valid JSON")
        return parsed, None
    except json.JSONDecodeError as e:
        # Final fallback: try to extract key-value pairs with regex
        return _regex_extract_params(raw_args, str(e))


def _regex_extract_params(broken_json: str, error_msg: str) -> tuple[dict | None, str | None]:
    """
    Last resort: extract parameters using regex when JSON parsing fails.
    """
    result = {}
    
    # Match "key": "value" or "key": value patterns
    string_pattern = r'"([^"]+)":\s*"([^"]*)"'
    number_pattern = r'"([^"]+)":\s*(\d+(?:\.\d+)?)'
    bool_pattern = r'"([^"]+)":\s*(true|false)'
    
    for match in re.finditer(string_pattern, broken_json):
        result[match.group(1)] = match.group(2)
    
    for match in re.finditer(number_pattern, broken_json):
        val = match.group(2)
        result[match.group(1)] = float(val) if '.' in val else int(val)
    
    for match in re.finditer(bool_pattern, broken_json):
        result[match.group(1)] = match.group(2) == 'true'
    
    if result:
        logger.warning(f"Kimi JSON sanitizer used regex extraction: {error_msg}")
        return result, None
    
    return None, error_msg
