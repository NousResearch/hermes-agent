# Auditable research trace

Hermes can emit a small, durable trail of web-research decisions for teams that need to explain how an answer was assembled. This supports trust, incident review, and compliance evidence without turning the run journal into a copy of browsing traffic.

## Explicit opt-in

Tracing is disabled by default. API clients must set the boolean request field `research_trace: true` on each `/v1/runs` request. The setting is per run, not global: omitted, `false`, or non-boolean values do not enable it. Existing clients therefore keep the legacy behavior and receive no `research.*` events.

```json
{
  "input": "Compare the published policies",
  "research_trace": true
}
```

The sink is context-local and attached to the run's profile/session execution. It is not shared between concurrent runs or profiles.

## Stored boundary

The trace schema is intentionally bounded and versionable. It may contain:

- a redacted, bounded query and provider name;
- bounded source metadata: redacted title/description, a normalized HTTP(S) URL, and result position;
- extraction status and a redacted error summary (never the extracted body);
- bounded decision labels/details;
- bounded counts and terminal status.

It never stores or forwards raw page content, free-form tool results, request/response bodies, headers, tokens, passwords, URL query strings, URL fragments, URL userinfo, or raw exception text. URLs keep only an HTTP(S) origin and conservative path segments; malformed, non-HTTP(S), credential-bearing, or unsafe path data is reduced or omitted. Canonical Hermes secret redaction runs before size limits, with a fail-closed fallback.

This is an audit aid, not a full browser recording or legal/compliance certification. Export, download, retention policy, and analytics are separate capabilities and are not implied by this trace.

## Verification

The focused suite covers default-off behavior, explicit boolean opt-in, nested session isolation, query/title/description/error/decision redaction, URL normalization, bounded source lists, and the guarantee that extracted content is not traced:

```bash
python -m pytest -q tests/tools/test_research_trace.py
python -m py_compile tools/research_trace.py tools/web_tools.py gateway/platforms/api_server_runs.py
```
