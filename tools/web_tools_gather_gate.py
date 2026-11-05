"""Anti-fabrication gather gate for web_search / web_extract.

Closes the data-gathering loop gap (community-reported: a model returned fake
placeholder data instead of reporting that web search failed / returned nothing).
When a gather step produced NO real data (zero results, provider failure, or the
extract backend returned nothing), stamp an explicit, model-readable marker on the
tool result so the model must report the gap rather than invent data to fill it.

This is a TOOL-LEVEL gate, not a prompt rule: prompts degrade under long context
compression, but a structured marker riding inside the tool result the model just
observed survives every compression. It deliberately does not add a new core tool,
env var, or schema field — it edits the shape of a result that already flows to the
model.
"""

from __future__ import annotations

from typing import Optional

# The exact text a model reads when a gather step returned no real data. Kept
# imperative and unambiguous; "no real data" + "report the gap" must not be soft.
_GATHER_GATE_INSTRUCTION = (
    "GATHER GATE: no real data was retrieved for this step. "
    "Do NOT invent, estimate, assume, or fabricate any results, values, examples, "
    "or content to fill the gap. Report the gap honestly to the user and either stop "
    "or retry the gather (different query, URL, or provider). Fabricated placeholder "
    "data is a hard failure."
)


def _marker(reason: str, retry_after=None) -> dict:
    """The structured marker appended onto a no-data result."""
    marker = {
        "gather_aborted": True,
        "gather_reason": reason,
        "gather_instruction": _GATHER_GATE_INSTRUCTION,
    }
    if retry_after is not None:
        marker["gather_retry_after"] = retry_after
    return marker


def gate_search_result(response_data: dict, provider_name: str) -> dict:
    """Stamp the gate marker on a web_search result if it carried no real results.

    Applies to BOTH cases: a successful-but-empty search (``success:true`` with an
    empty ``web`` list) and a failure (``success:false``). Empty-success is the
    sneaky case — the model sees ``success:true`` and is most tempted to invent.
    """
    web = (response_data.get("data") or {}).get("web") or []
    if web:
        return response_data  # real data present — no gate
    reason = _failure_reason(response_data) or f"No results returned by {provider_name}"
    return {**response_data, **_marker(reason)}


def gate_extract_results(results: list, provider_name: str) -> Optional[dict]:
    """Return (possibly marker-stamped) results for web_extract.

    Returns ``None`` when there is usable content, else a dict carrying the marker
    and the original results so no diagnostic detail is lost.
    """
    usable = [r for r in results if r.get("content") and not r.get("error")]
    if usable:
        return None
    reasons = sorted({r.get("error") or "no content returned" for r in results})
    reason = ("; ".join(reasons[:3]))[:500] or f"No content returned by {provider_name}"
    return {"success": False, "results": results, **_marker(reason)}


def gate_extract_failure(error_json: str) -> str:
    """Stamp the gate marker onto an existing serialized extract-failure payload.

    Covers the provider-less paths (search-only backend, strict-selection error,
    never-configured install) that return a bare ``{"success": false, "error": ...}``
    before any results exist to gate — the same fabrication invitation the
    results gate closes.
    """
    import json
    payload = json.loads(error_json)
    reason = str(payload.get("error") or "No web extract provider resolved")[:500]
    return json.dumps({**payload, **_marker(reason)}, ensure_ascii=False)


def _failure_reason(response_data: dict) -> Optional[str]:
    """Pull the error string from a failed search result, or ``None``."""
    error = response_data.get("error")
    if error:
        return str(error)[:500]
    return None