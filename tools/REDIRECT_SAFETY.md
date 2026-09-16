# Safe Redirect Helpers

## Overview

The `redirect_safety` module provides helpers to prevent **open redirect vulnerabilities** — a class of security issues where attackers can trick your application into redirecting users to malicious sites.

## Why This Matters

Open redirect vulnerabilities enable:
- **Phishing attacks**: Redirect to fake login pages that look legitimate
- **Malware distribution**: Redirect to sites that download malware
- **Session hijacking**: Redirect to attacker-controlled OAuth callbacks
- **Reputation damage**: Your domain appears in phishing URLs

## Quick Start

### Basic Usage (Same-Origin Only)

```python
from tools.redirect_safety import safe_redirect_url

# In a web endpoint handler
redirect_to = request.args.get('next', '/')
safe_url = safe_redirect_url(redirect_to, base_url="https://app.example.com")

# Always safe - returns either a valid same-origin URL or fallback
return redirect(safe_url)
```

### With Allowlist (Trusted External Domains)

```python
# Allow redirects to specific trusted domains
safe_url = safe_redirect_url(
    redirect_to,
    base_url="https://app.example.com",
    allowed_origins=[
        "https://docs.example.com",
        "https://auth.example.com",
    ],
)
```

### Wildcard Subdomains

```python
# Allow redirects to any subdomain
safe_url = safe_redirect_url(
    redirect_to,
    base_url="https://app.example.com",
    allowed_origins=["https://*.example.com"],
)

# This allows:
# - https://api.example.com
# - https://tenant1.example.com
# - https://v1.api.example.com
# But NOT:
# - https://example.com (root domain not matched by *.example.com)
# - https://evil.com
```

## Function Reference

### `safe_redirect_url(redirect_url, base_url, ...)`

Validates and returns a safe redirect URL.

**Parameters:**

- `redirect_url` (str): URL to redirect to (from user input)
- `base_url` (str): Your application's base URL (e.g., "https://app.example.com")
- `fallback_url` (str): URL to use if validation fails (default: "/")
- `allowed_origins` (list[str] | None): Additional allowed origins/domains (default: None)
- `allowed_schemes` (frozenset[str]): Allowed URL schemes (default: {"http", "https"})

**Returns:** A validated safe URL string

**Behavior:**

1. **Relative URLs**: Always allowed and resolved to absolute
   - `/dashboard` → `https://app.example.com/dashboard` ✓
   - `?param=value` → `https://app.example.com?param=value` ✓

2. **Same-origin absolute URLs**: Always allowed
   - `https://app.example.com/page` ✓

3. **Cross-origin URLs**: Only allowed if in `allowed_origins`
   - `https://evil.com` → fallback (blocked) ✗
   - `https://docs.example.com` → allowed if in allowlist ✓

4. **Dangerous schemes**: Always blocked
   - `javascript:alert(1)` → fallback ✗
   - `data:text/html,...` → fallback ✗
   - `file:///etc/passwd` → fallback ✗

5. **Malformed/empty URLs**: Return fallback

## Real-World Examples

### OAuth Callback

```python
@app.route('/oauth/callback')
def oauth_callback():
    # User came from another service, redirect back safely
    return_url = request.args.get('return_to', '/dashboard')
    
    safe_url = safe_redirect_url(
        return_url,
        base_url=request.host_url.rstrip('/'),
        allowed_origins=[
            "https://auth.example.com",
            "https://partner.example.com",
        ],
        fallback_url='/dashboard',
    )
    
    return redirect(safe_url)
```

### Logout Redirect

```python
@app.route('/logout')
def logout():
    logout_user()
    
    # Allow redirect to public pages or external marketing site
    next_url = request.args.get('next', '/')
    safe_url = safe_redirect_url(
        next_url,
        base_url=request.host_url.rstrip('/'),
        allowed_origins=["https://www.example.com"],  # marketing site
        fallback_url='/',
    )
    
    return redirect(safe_url)
```

### Multi-Tenant SaaS

```python
@app.route('/switch-tenant')
def switch_tenant():
    tenant_url = request.args.get('tenant_url')
    
    # Allow redirects to any tenant subdomain
    safe_url = safe_redirect_url(
        tenant_url,
        base_url=request.host_url.rstrip('/'),
        allowed_origins=["https://*.tenants.example.com"],
        fallback_url='/tenant-selector',
    )
    
    return redirect(safe_url)
```

## Attack Scenarios Prevented

### 1. Basic Open Redirect

```
Attacker sends: https://app.example.com/login?next=https://evil.com/phishing
                                                  ^^^^^^^^^^^^^^^^^^^^^^^^
Without protection: User is redirected to evil.com
With safe_redirect_url: User is redirected to fallback (/)
```

### 2. Subdomain Takeover Exploit

```
Attacker registers: old-subdomain.example.com (abandoned by company)
Sends: https://app.example.com/auth?return=https://old-subdomain.example.com/steal-token

Without allowlist: Blocked (different origin)
With *.example.com allowlist: Blocked (you must explicitly add domains)
```

### 3. JavaScript Scheme XSS

```
Attacker sends: https://app.example.com/goto?url=javascript:alert(document.cookie)

Without protection: JavaScript executes in victim's browser
With safe_redirect_url: Blocked (javascript not in allowed_schemes)
```

## Security Considerations

### What This DOES Protect Against

- ✅ Open redirect to attacker-controlled domains
- ✅ JavaScript/data/file URI schemes
- ✅ Accidental redirects to different subdomains
- ✅ Protocol downgrade (https → http)
- ✅ Port mismatch attacks

### What This DOES NOT Protect Against

- ❌ **XSS** (Cross-Site Scripting) — use output encoding
- ❌ **SSRF** (Server-Side Request Forgery) — use `tools/url_safety.py`
- ❌ **Path traversal** — validate file paths separately
- ❌ **DNS rebinding** — requires connection-level validation

### Best Practices

1. **Always specify a safe fallback URL**
   ```python
   # Good
   safe_redirect_url(user_input, base_url, fallback_url='/dashboard')
   
   # Risky (fallback is "/" which might be unauthenticated)
   safe_redirect_url(user_input, base_url)
   ```

2. **Use the smallest allowlist possible**
   ```python
   # Good - explicit list
   allowed_origins=["https://docs.example.com", "https://api.example.com"]
   
   # Risky - overly broad
   allowed_origins=["https://*.example.com"]  # allows ALL subdomains
   ```

3. **Don't trust URL parameters blindly**
   ```python
   # NEVER do this
   return redirect(request.args.get('next'))
   
   # ALWAYS do this
   safe_url = safe_redirect_url(
       request.args.get('next', '/'),
       base_url=request.host_url.rstrip('/'),
   )
   return redirect(safe_url)
   ```

4. **Validate on the server side**
   - Client-side validation can be bypassed
   - Always perform redirect validation server-side

5. **Log blocked redirects**
   - The module logs warnings for blocked redirects
   - Monitor these logs to detect attack attempts

## Testing

Run the test suite:

```bash
pytest tests/tools/test_redirect_safety.py -v
```

The test suite includes:
- 58 test cases covering all validation scenarios
- Edge cases (empty URLs, malformed URLs, unicode)
- Real-world integration scenarios
- Attack scenario verification

## Related Security Modules

- `tools/url_safety.py` — SSRF protection for outbound HTTP requests
- `tools/approval.py` — Dangerous command detection
- `gateway/platforms/*/` — SSRF redirect guards for media downloads

## References

- [OWASP: Unvalidated Redirects and Forwards](https://cheatsheetseries.owasp.org/cheatsheets/Unvalidated_Redirects_and_Forwards_Cheat_Sheet.html)
- [CWE-601: URL Redirection to Untrusted Site](https://cwe.mitre.org/data/definitions/601.html)
- [PortSwigger: Open Redirection](https://portswigger.net/kb/issues/00500100_open-redirection-reflected)
