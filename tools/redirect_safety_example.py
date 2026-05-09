"""Example usage of the safe_redirect_url helper function."""

from tools.redirect_safety import safe_redirect_url

# Example 1: Basic same-origin validation
print("=== Example 1: Same-origin validation ===")
base = "https://app.example.com"

# Safe relative URL
result = safe_redirect_url("/dashboard", base)
print(f"Input: /dashboard -> {result}")

# Safe same-origin absolute URL
result = safe_redirect_url("https://app.example.com/profile", base)
print(f"Input: https://app.example.com/profile -> {result}")

# Blocked cross-origin URL
result = safe_redirect_url("https://evil.com/phishing", base)
print(f"Input: https://evil.com/phishing -> {result} (blocked!)")

print()

# Example 2: With allowlist
print("=== Example 2: With allowlist ===")
allowed = ["https://docs.example.com", "https://*.trusted.com"]

result = safe_redirect_url(
    "https://docs.example.com/guide",
    base,
    allowed_origins=allowed,
)
print(f"Input: https://docs.example.com/guide -> {result}")

result = safe_redirect_url(
    "https://api.trusted.com/endpoint",
    base,
    allowed_origins=allowed,
)
print(f"Input: https://api.trusted.com/endpoint -> {result}")

result = safe_redirect_url(
    "https://evil.com",
    base,
    allowed_origins=allowed,
)
print(f"Input: https://evil.com -> {result} (blocked!)")

print()

# Example 3: Dangerous schemes blocked
print("=== Example 3: Dangerous schemes blocked ===")

result = safe_redirect_url("javascript:alert(1)", base)
print(f"Input: javascript:alert(1) -> {result} (blocked!)")

result = safe_redirect_url("data:text/html,<script>alert(1)</script>", base)
print(f"Input: data:text/html,... -> {result} (blocked!)")

print()
print("All examples completed successfully!")
