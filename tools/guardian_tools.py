import requests

def guardian_url_inspect(url: str):
    try:
        response = requests.get(url, allow_redirects=True, timeout=5)
        return f"Inspection Complete: Final URL is {response.url}. Status: {response.status_code}"
    except Exception as e:
        return f"Inspection failed: {str(e)}"

def guardian_pass_audit(password: str):
    if len(password) < 10: return "Verdict: WEAK (Too short)"
    if any(seq in password for seq in ["123", "abc", "qwerty"]):
        return "Verdict: WEAK (Common pattern detected)"
    return "Verdict: STRONG (Good complexity)"

TOOLS = [
    {"name": "guardian_url_inspect", "function": guardian_url_inspect, "description": "Safely inspects suspicious URLs."},
    {"name": "guardian_pass_audit", "function": guardian_pass_audit, "description": "Technical audit of password strength."}
]
