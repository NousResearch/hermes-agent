"""Safe redirect helpers — prevents open redirect vulnerabilities.

Validates redirect URLs to ensure they point to safe, trusted destinations.
Prevents attackers from using the application to redirect users to malicious
sites (phishing, malware distribution, credential theft).

Two validation modes:

1. Same-origin validation (strictest, recommended):
   - URL must be on the same origin (scheme + domain + port)
   - Relative URLs are allowed and safe
   - Example: /dashboard, /auth/callback, https://app.example.com/profile

2. Allowlist validation (for trusted external domains):
   - URL must match an allowed origin or domain pattern
   - Supports wildcards for subdomains
   - Example allowlist: ["https://docs.example.com", "*.trusted-partner.com"]

Usage:

    # Same-origin validation (default)
    from tools.redirect_safety import safe_redirect_url
    
    redirect_to = request.args.get('next', '/')
    safe_url = safe_redirect_url(redirect_to, base_url="https://app.example.com")
    return redirect(safe_url)
    
    # Allowlist validation
    safe_url = safe_redirect_url(
        redirect_to,
        base_url="https://app.example.com",
        allowed_origins=["https://docs.example.com", "https://*.example.com"],
    )
"""

import logging
from urllib.parse import urlparse, urljoin

logger = logging.getLogger(__name__)


def _normalize_origin(url: str) -> str:
    """Extract normalized origin (scheme://hostname:port) from a URL."""
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        return ""
    
    # Port is already included in netloc if non-standard
    # (http://example.com:8080 -> netloc='example.com:8080')
    return f"{parsed.scheme}://{parsed.netloc}"


def _matches_wildcard_domain(hostname: str, pattern: str) -> bool:
    """Check if hostname matches a wildcard domain pattern.
    
    Examples:
        _matches_wildcard_domain("api.example.com", "*.example.com") -> True
        _matches_wildcard_domain("example.com", "*.example.com") -> False
        _matches_wildcard_domain("api.example.com", "example.com") -> False
    """
    if not pattern.startswith("*."):
        return hostname == pattern
    
    # Pattern is *.example.com -> match api.example.com but not example.com
    suffix = pattern[1:]  # Remove leading *
    if hostname.endswith(suffix):
        # Ensure there's at least one subdomain component
        # (prevents "example.com" from matching "*.example.com")
        prefix = hostname[:-len(suffix)]
        return "." in prefix or prefix != ""
    return False


def _is_allowed_origin(url: str, allowed_origins: list[str]) -> bool:
    """Check if a URL's origin matches any entry in the allowlist.
    
    Args:
        url: Full URL to check
        allowed_origins: List of allowed origins or domain patterns.
                        Exact matches: "https://example.com"
                        Wildcard subdomains: "https://*.example.com"
    
    Returns:
        True if the URL's origin is allowed, False otherwise
    """
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        return False
    
    url_origin = _normalize_origin(url)
    hostname = parsed.hostname or ""
    
    for allowed in allowed_origins:
        # Exact origin match
        if url_origin == allowed:
            return True
        
        # Check for wildcard domain pattern
        allowed_parsed = urlparse(allowed)
        if allowed_parsed.scheme == parsed.scheme:
            allowed_hostname = allowed_parsed.hostname or ""
            if _matches_wildcard_domain(hostname, allowed_hostname):
                # If the allowed pattern has a port, it must match
                if allowed_parsed.port is None or allowed_parsed.port == parsed.port:
                    return True
    
    return False


def safe_redirect_url(
    redirect_url: str,
    base_url: str,
    fallback_url: str = "/",
    allowed_origins: list[str] | None = None,
    allowed_schemes: frozenset[str] = frozenset({"http", "https"}),
) -> str:
    """Validate and return a safe redirect URL.
    
    Args:
        redirect_url: The URL to redirect to (from user input, query param, etc.)
        base_url: The application's base URL (e.g., "https://app.example.com")
        fallback_url: URL to use if validation fails (default: "/")
        allowed_origins: Optional list of additional allowed origins/domains.
                        If None (default), only same-origin redirects are allowed.
                        Supports wildcard subdomains: "https://*.example.com"
        allowed_schemes: Set of allowed URL schemes (default: http, https)
    
    Returns:
        A validated safe redirect URL. Returns fallback_url if validation fails.
    
    Examples:
        # Same-origin validation
        >>> safe_redirect_url("/dashboard", "https://app.example.com")
        'https://app.example.com/dashboard'
        
        >>> safe_redirect_url("https://evil.com", "https://app.example.com")
        '/'
        
        # Allowlist validation
        >>> safe_redirect_url(
        ...     "https://docs.example.com/guide",
        ...     "https://app.example.com",
        ...     allowed_origins=["https://docs.example.com"]
        ... )
        'https://docs.example.com/guide'
    """
    if not redirect_url or not isinstance(redirect_url, str):
        logger.warning("Invalid redirect URL (empty or not a string), using fallback")
        return fallback_url
    
    redirect_url = redirect_url.strip()
    if not redirect_url:
        return fallback_url
    
    try:
        # Parse the base URL to get the application origin
        base_parsed = urlparse(base_url)
        base_origin = _normalize_origin(base_url)
        
        if not base_origin:
            logger.error(
                "Invalid base_url provided to safe_redirect_url: %s. Using fallback.",
                base_url,
            )
            return fallback_url
        
        # Handle relative URLs (these are always safe)
        parsed = urlparse(redirect_url)
        if not parsed.scheme and not parsed.netloc:
            # Relative URL - resolve against base and return
            absolute_url = urljoin(base_url, redirect_url)
            logger.debug("Relative redirect URL resolved: %s -> %s", redirect_url, absolute_url)
            return absolute_url
        
        # Absolute URL - validate scheme first
        if parsed.scheme not in allowed_schemes:
            logger.warning(
                "Blocked redirect to URL with disallowed scheme '%s': %s",
                parsed.scheme,
                redirect_url,
            )
            return fallback_url
        
        # Validate origin
        redirect_origin = _normalize_origin(redirect_url)
        if not redirect_origin:
            logger.warning("Blocked redirect to malformed URL: %s", redirect_url)
            return fallback_url
        
        # Check same-origin
        if redirect_origin == base_origin:
            logger.debug("Same-origin redirect allowed: %s", redirect_url)
            return redirect_url
        
        # Check allowlist if provided
        if allowed_origins:
            if _is_allowed_origin(redirect_url, allowed_origins):
                logger.debug(
                    "Cross-origin redirect allowed (allowlist match): %s",
                    redirect_url,
                )
                return redirect_url
        
        # If we get here, the redirect is not allowed
        logger.warning(
            "Blocked cross-origin redirect (not in allowlist): %s -> %s",
            base_origin,
            redirect_origin,
        )
        return fallback_url
    
    except Exception as exc:
        # Fail closed on unexpected errors
        logger.warning(
            "Error validating redirect URL %s: %s. Using fallback.",
            redirect_url,
            exc,
            exc_info=True,
        )
        return fallback_url


# Alias for compatibility (some codebases may use camelCase naming)
getSafeRedirect = safe_redirect_url
