"""Regression tests for tools.secret_redaction.

Includes a direct regression test for the real DATABASE_URL credential
exposure that happened during this engagement (a `docker inspect` env dump
whose name-only filter missed the embedded password inside the URI value).
"""

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
        # is nowhere near the start of the line.
        body = '{"status": "ok", "body": "SESSION_TOKEN=abcdef123456 and more text after"}'
        result = redact_text(body)
        assert "abcdef123456" not in result
        assert "and more text after" in result

    def test_json_style_quoted_key_value(self):
        line = '  "POSTGRES_PASSWORD": "hunter2value",'
        result = redact_text(line)
        assert "hunter2value" not in result
        assert "POSTGRES_PASSWORD" in result

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
