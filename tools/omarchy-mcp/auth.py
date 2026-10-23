import os

TOKEN = os.environ.get("OMARCHY_MCP_TOKEN", "")

def verify_token(token: str) -> bool:
    if not TOKEN:
        return False
    if len(token) != len(TOKEN):
        return False
    result = 0
    for a, b in zip(token, TOKEN):
        result |= ord(a) ^ ord(b)
    return result == 0
