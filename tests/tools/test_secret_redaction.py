"""Regression tests for tools.secret_redaction.

Includes a direct regression test for the real DATABASE_URL credential
exposure that happened during this engagement (a `docker inspect` env dump
whose name-only filter missed the embedded password inside the URI value).
"""

import time

from tools.secret_redaction import redact_mapping, redact_text


class TestNamePatternRedaction:
    def test_token_suffix(self):
        assert "abc123" not in redact_text("MCP_TOKEN_SIGNING_SECRET=abc123")

    def test_password_suffix(self):
        assert "hunter2" not in redact_text("POSTGRES_PASSWORD=hunter2")

    def test_secret_suffix(self):
        assert "s3cr3t" not in redact_text("N8N_ENCRYPTION_KEY=s3cr3t")

    def test_api_key_suffix(self):
        assert "sk-live-x" not in redact_text("STRIPE_API_KEY=sk-live-x")

    def test_client_secret_suffix(self):
        assert "cs-x" not in redact_text("OAUTH_CLIENT_SECRET=cs-x")

    def test_benign_names_pass_through_unchanged(self):
        line = "HOSTNAME=0.0.0.0 GIT_SHA=09ba1a9 NODE_ENV=production MCP_HOST=0.0.0.0"
        assert redact_text(line) == line


class TestDatabaseUrlRegression:
    """Direct regression test for the real leak this engagement produced."""

    def test_database_url_embedded_credential_redacted(self):
        leaked = "DATABASE_URL=postgresql://projectos:cTo9wE9i8aJkO9LwrkEUNNxxu8HYmqL7@db:5432/projectos"
        result = redact_text(leaked)
        assert "cTo9wE9i8aJkO9LwrkEUNNxxu8HYmqL7" not in result
        # Host/scheme/username/dbname stay visible — they're the useful
        # diagnostic part, not the secret.
        assert "postgresql://" in result
        assert "projectos:" in result
        assert "@db:5432/projectos" in result

    def test_database_url_redacted_even_without_matching_var_name(self):
        # The exact failure mode from the real incident: the value pattern
        # must be caught independent of what variable name holds it.
        leaked = "SOME_UNRELATED_NAME=postgresql://user:hunter2@host:5432/db"
        assert "hunter2" not in redact_text(leaked)

    def test_generic_uri_credential_any_scheme(self):
        assert "s3cr3t" not in redact_text("mysql://root:s3cr3t@localhost/db")
        assert "s3cr3t" not in redact_text("redis://:s3cr3t@localhost:6379")


class TestNonUriNamePatternRegression:
    """Direct regression test for a second, independently-confirmed leak:
    the name-pattern matcher was `^`-anchored against the stripped line, so
    it only fired when the secret-shaped NAME was the very first token —
    exactly the one shape `DATABASE_URL=...` happens to have, and exactly
    the shape almost nothing else has. A real `docker inspect` env dump
    (JSON array of `"NAME=value"` strings, each indented and quoted) never
    matched, so every non-URI secret in it leaked in full."""

    def test_docker_inspect_env_array_shape(self):
        # The literal shape `docker inspect`'s `.Config.Env` produces: an
        # indented, quoted, comma-terminated JSON array element.
        line = '            "MCP_TOKEN_SIGNING_SECRET=abc123def456",'
        result = redact_text(line)
        assert "abc123def456" not in result
        assert "MCP_TOKEN_SIGNING_SECRET" in result

    def test_multiple_docker_inspect_env_lines(self):
        dump = "\n".join([
            '        "Env": [',
            '            "PATH=/usr/local/sbin:/usr/sbin",',
            '            "POSTGRES_PASSWORD=s3cr3t-prod-value",',
            '            "HERMES_SKILLS_BRIDGE_TOKEN=bridgetok123456",',
            '            "OPENAI_API_KEY=sk-not-a-real-key-but-shaped-like-one",',
            '        ]',
        ])
        result = redact_text(dump)
        assert "s3cr3t-prod-value" not in result
        assert "bridgetok123456" not in result
        assert "sk-not-a-real-key-but-shaped-like-one" not in result
        # Non-secret lines and the surrounding JSON structure survive.
        assert "/usr/local/sbin" in result
        assert '"Env": [' in result

    def test_bare_password_variable_no_underscore_prefix(self):
        # PGPASSWORD is the canonical Postgres env var and has no
        # underscore before the secret word — the old key regex required
        # one and would have missed this even at line start.
        assert "topsecret" not in redact_text("PGPASSWORD=topsecret")
        assert "s3cr3t" not in redact_text("    APIKEY=s3cr3t")

    def test_secret_embedded_mid_line_in_json_body(self):
        # An http_probe response body is one JSON-encoded line; the secret
        # sits inside the outer "body" field's own string, with no other
        # quote between it and that field's closing quote. The redaction is
        # bounded to the JSON string the pair is embedded in (the "body"
        # field's closing quote) — which in this case means the harmless
        # trailing text is swallowed too, since it shares that same string.
        # That is the deliberate, safe direction: a redaction that consumes
        # extra harmless text is fine; one that leaks part of a secret is
        # not (see TestDelimiterContainingSecretValues below for why a
        # fixed-delimiter cutoff was tried and rejected).
        body = '{"status": "ok", "body": "SESSION_TOKEN=abcdef123456 and more text after"}'
        result = redact_text(body)
        assert "abcdef123456" not in result
        assert "and more text after" not in result
        assert "[REDACTED]" in result

    def test_json_style_quoted_key_value(self):
        line = '  "POSTGRES_PASSWORD": "hunter2value",'
        result = redact_text(line)
        assert "hunter2value" not in result
        assert "POSTGRES_PASSWORD" in result

    def test_quoted_value_does_not_swallow_sibling_json_fields(self):
        # Unlike the bare/ambiguous case above, a value with its OWN
        # opening+closing quote pair is bounded to exactly that pair, even
        # when the key itself was also preceded by a quote — it must not
        # spill into unrelated sibling fields on the same line.
        line = '{"a": "PASSWORD=secretvalue", "b": "unrelated"}'
        result = redact_text(line)
        assert "secretvalue" not in result
        assert '"unrelated"' in result


class TestDelimiterContainingSecretValues:
    """Regression: a value pattern that stops at the first space/comma/
    semicolon/ampersand/brace/bracket truncates any secret whose VALUE
    itself contains one of those characters, leaking its tail. Real secrets
    routinely do (a generated passphrase with punctuation, a token used in
    a query string with '&amp;'). This was a real regression introduced by an
    earlier fix for the docker-inspect leak above — fixing the anchor
    without fixing the value boundary just relocated the bug."""

    def test_ampersand_in_value_fully_redacted(self):
        result = redact_text("POSTGRES_PASSWORD=aB3&xY9zQw7Lm2Pk5Rt8Nv")
        assert "aB3" not in result
        assert "xY9zQw7Lm2Pk5Rt8Nv" not in result

    def test_comma_in_value_fully_redacted(self):
        result = redact_text("ADMIN_PASSWORD=Tr0ub4dor,3xKcd-9")
        assert "Tr0ub4dor" not in result
        assert "3xKcd-9" not in result

    def test_space_in_value_fully_redacted(self):
        result = redact_text("ADMIN_PASSWORD=correct horse battery staple")
        assert "correct" not in result
        assert "horse battery staple" not in result

    def test_semicolon_and_brace_in_value_fully_redacted(self):
        assert "part1" not in redact_text("APP_SESSION_SECRET=part1;part2")
        assert "part2" not in redact_text("APP_SESSION_SECRET=part1;part2")
        assert "abc" not in redact_text("TOKEN={abc}")


class TestEscapedQuoteInValueFullyRedacted:
    """Regression: the quote-boundary fix above used a naive `str.find` for
    the closing quote, which matches an ESCAPED quote (`\\"` inside a JSON
    string) exactly as if it were the real terminator — cutting the
    redaction short and leaving everything after it, including the rest of
    the secret, in plaintext. `docker inspect` emits valid JSON, so any
    real secret containing a literal double-quote arrives in exactly this
    escaped form. Same bug class as TestDelimiterContainingSecretValues
    above, different trigger character."""

    def test_docker_inspect_shape_with_escaped_quote_in_value(self):
        line = '        "DB_PASSWORD=pa\\"ssTAILLEAK123",'
        result = redact_text(line)
        assert "TAILLEAK123" not in result
        assert "pa" not in result or "[REDACTED]" in result

    def test_json_value_with_escaped_quote(self):
        line = '{"PASSWORD": "it\\"s complicated xyz789"}'
        result = redact_text(line)
        assert "xyz789" not in result
        assert "complicated" not in result

    def test_double_escaped_backslash_before_quote_is_a_real_terminator(self):
        # Two backslashes means the backslash itself is escaped, so the
        # quote that follows it IS a real, unescaped closing quote — this
        # must still bound the value correctly, not be misread as escaped.
        line = '{"API_KEY": "value\\\\"}extra_after_close'
        result = redact_text(line)
        assert "value" not in result
        assert "extra_after_close" in result

    def test_benign_substring_not_falsely_flagged_by_boundary(self):
        # "SOMETOKENISH" contains "TOKEN" but isn't secret-shaped on its
        # own without a following separator+value — nothing here should
        # explode or over-match past the actual value.
        line = "DESCRIPTION=a component named SOMETOKENISH exists"
        assert redact_text(line) == line


class TestAuthorizationHeaderRedaction:
    def test_bearer(self):
        result = redact_text("Authorization: Bearer abcdef123456")
        assert "abcdef123456" not in result

    def test_basic(self):
        result = redact_text("Authorization: Basic dXNlcjpwYXNz")
        assert "dXNlcjpwYXNz" not in result


class TestTokenPrefixRedaction:
    def test_github_pat(self):
        assert "ghp_1234567890abcdefghij1234567890abcdef" not in redact_text(
            "token: ghp_1234567890abcdefghij1234567890abcdef"
        )

    def test_github_fine_grained_pat(self):
        assert "github_pat_11ABCDEFG0123456789_abcdefghijklmnopqrstuvwxyz" not in redact_text(
            "github_pat_11ABCDEFG0123456789_abcdefghijklmnopqrstuvwxyz"
        )

    def test_openai_style_key(self):
        assert "sk-abcdefghijklmnopqrstuvwx" not in redact_text("key=sk-abcdefghijklmnopqrstuvwx")

    def test_anthropic_style_key(self):
        assert "sk-ant-abcdefghijklmnopqrstuvwx" not in redact_text("key=sk-ant-abcdefghijklmnopqrstuvwx")

    def test_aws_access_key_id(self):
        assert "AKIAIOSFODNN7EXAMPLE" not in redact_text("AKIAIOSFODNN7EXAMPLE")

    def test_gitlab_pat(self):
        assert "glpat-1234567890abcdefghij" not in redact_text("glpat-1234567890abcdefghij")


class TestJwtRedaction:
    def test_jwt_shaped_value(self):
        jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U"
        assert jwt not in redact_text(f"token={jwt}")

    def test_short_dotted_string_not_falsely_redacted(self):
        # Guard against over-eager redaction of ordinary version-ish strings.
        assert redact_text("version=1.2.3") == "version=1.2.3"


class TestCookieRedaction:
    def test_set_cookie(self):
        result = redact_text("Set-Cookie: session=abcdefghijklmnopqrstuvwxyz; Path=/")
        assert "abcdefghijklmnopqrstuvwxyz" not in result

    def test_cookie_header(self):
        result = redact_text("Cookie: sid=abcdefghijklmnopqrstuvwxyz")
        assert "abcdefghijklmnopqrstuvwxyz" not in result


class TestMappingRedaction:
    def test_nested_dict_secret_key(self):
        data = {"env": {"DATABASE_URL": "postgresql://u:p@h/d", "HOSTNAME": "x"}}
        out = redact_mapping(data)
        assert "p@h" not in str(out) or "[REDACTED]" in out["env"]["DATABASE_URL"]
        assert out["env"]["HOSTNAME"] == "x"

    def test_secret_suffixed_key_redacted_wholesale(self):
        data = {"API_KEY": "raw-value-should-never-appear"}
        out = redact_mapping(data)
        assert out["API_KEY"] == "[REDACTED]"

    def test_bare_key_name_redacted_wholesale(self):
        # Bare key, no underscore prefix — the exact shape that slipped
        # through before the _is_secret_key fix.
        data = {"PASSWORD": "raw-value-should-never-appear"}
        out = redact_mapping(data)
        assert out["PASSWORD"] == "[REDACTED]"

    def test_list_of_strings(self):
        out = redact_mapping({"lines": ["ok=1", "PASSWORD=hunter2"]})
        assert "hunter2" not in str(out)


class TestFailClosedBehavior:
    def test_empty_string(self):
        assert redact_text("") == ""

    def test_none_safe(self):
        assert redact_text(None) is None


class TestUriPatternNotCatastrophicallySlow:
    """Regression: `_URI_CREDENTIAL_PATTERN`'s scheme group was an
    unbounded `[a-zA-Z0-9+.-]*` — on a long run of scheme-shaped characters
    with no `://` ever following, the greedy match backtracks one
    character at a time from every one of n start positions, an O(n^2)
    blowup. Live-proven through the real, registered `rob_http_probe` tool:
    ~30s of CPU on a 200,000-char adversarial line. Bounding the scheme's
    repetition (no real URI scheme is anywhere near 16 characters) removes
    the pathological case without narrowing what actually gets redacted."""

    def test_long_alnum_run_with_no_scheme_separator_is_fast(self):
        adversarial = "a" * 200_000
        started = time.monotonic()
        redact_text(adversarial)
        elapsed = time.monotonic() - started
        assert elapsed < 2.0, f"took {elapsed:.2f}s — expected well under 1s"

    def test_realistic_uri_still_redacted_after_bounding(self):
        # The bound must not break genuinely long-ish real schemes.
        assert "hunter2" not in redact_text("mongodb+srv://user:hunter2@cluster0.example.net/db")
        assert "hunter2" not in redact_text("postgresql://user:hunter2@host:5432/db")
