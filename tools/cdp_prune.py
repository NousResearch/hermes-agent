#!/usr/bin/env python3
"""Safe CDP tab pruning for coordinator Brave/CDP lanes.

After a Kanban task completes, worker lanes tend to accumulate throwaway
``about:blank`` and ``chrome://newtab/`` tabs. This module closes *only*
those clearly-disposable tabs on a lane's Chrome DevTools Protocol (CDP)
endpoint, and never touches a real page.

Design goals (see docs/cdp-tab-pruning.md):

* **Conservative by construction.** The default disposable set is exactly
  ``about:blank`` + new-tab pages. Every real URL — http(s), file, and
  even non-newtab ``chrome://``/``brave://`` pages — is preserved. Synthetic
  ``data:text/html,...`` scratch tabs are only closable when explicitly
  opted in (``allow_scratch``), and a broad domain-critical denylist
  (trading, marketplace, finance/IRS, health, auth/MFA, client/vendor)
  force-skips anything that even *looks* sensitive, as defense in depth.
* **Never orphan a lane.** At least one ``page`` target is preserved per
  browser process, so pruning can never terminate the lane's only usable
  context — this matches the coordinator-lane model where each Brave lane
  is one process launched with an intentional initial ``about:blank``.
* **Sanitized.** Logs carry lane host:port, per-class counts, and URL
  *classes* only — never page bodies, cookies, localStorage, tokens, full
  URLs, or titles.
* **Inert until opted in.** The runtime trigger is gated behind
  ``HERMES_CDP_PRUNE_ENABLED`` (default off) and dry-run
  (``HERMES_CDP_PRUNE_DRY_RUN`` default on), so importing/merging this
  changes no behavior until an operator activates it.

The pure ``classify_url`` / ``plan_prune`` functions carry all the safety
logic and are exercised directly by the test suite with mocked CDP target
lists; the thin :class:`HttpCdpClient` seam is the only part that touches a
real browser and is injectable so tests never open a socket.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

# URL "classes" a target can fall into. Only ``blank`` / ``newtab`` (and
# optionally ``scratch``) are ever closable; ``real`` and ``sensitive`` are
# always preserved.
CLASS_BLANK = "blank"
CLASS_NEWTAB = "newtab"
CLASS_SCRATCH = "scratch"
CLASS_REAL = "real"
CLASS_SENSITIVE = "sensitive"

# CDP target ``type`` values we treat as user-facing tabs. Everything else
# (service_worker, background_page, worker, iframe, browser, ...) is never a
# prune candidate.
PAGE_TYPE = "page"

# New-tab landing pages across Chromium forks. Compared against the URL with a
# trailing slash stripped and lowercased.
_NEWTAB_URLS = frozenset(
    {
        "chrome://newtab",
        "chrome://new-tab-page",
        "chrome://new-tab-page-third-party",
        "brave://newtab",
        "edge://newtab",
        "about:newtab",
        "about:home",
    }
)

# ``about:blank`` variants. ``about:blank#blocked`` shows up when a popup is
# blocked; still disposable.
_BLANK_URLS = frozenset({"", "about:blank", "about:blank#blocked"})

# Defense-in-depth denylist. If a URL matches any of these substrings we mark
# it ``sensitive`` and force-skip, regardless of any other rule. Over-matching
# here is *safe* (it only preserves more tabs); the goal is to make it
# essentially impossible for the pruner to ever close a human/domain-critical
# tab even if a future rule change widened the closable set. Matched against a
# lowercased ``host + path`` string.
_SENSITIVE_HINTS: tuple[str, ...] = (
    # finance / trading / IRS / taxes
    "bank", "chase", "wellsfargo", "wells-fargo", "citi", "capitalone",
    "fidelity", "schwab", "vanguard", "robinhood", "etrade", "e-trade",
    "coinbase", "kraken", "binance", "gemini", "trading", "broker",
    "irs.gov", "turbotax", "hrblock", "tax", "paypal", "venmo", "stripe",
    "wise.com", "invoice", "billing", "payroll", "quickbooks",
    # marketplace / seller / commerce
    "amazon", "seller", "sellercentral", "ebay", "etsy", "shopify",
    "marketplace", "mercari", "poshmark", "stockx", "walmart", "alibaba",
    "storefront", "checkout", "cart",
    # health
    "health", "mychart", "patient", "medical", "clinic", "hospital",
    "pharmacy", "cvs", "walgreens", "insurance", "medicare", "medicaid",
    # account / security / login / MFA
    "login", "signin", "sign-in", "sign_in", "logout", "account", "accounts",
    "auth", "oauth", "openid", "sso", "saml", "mfa", "2fa", "otp", "verify",
    "verification", "password", "credential", "security", "okta", "duo",
    "authy", "onelogin", "recovery",
    # client / vendor / work portals
    "client", "vendor", "portal", "admin", "dashboard", "console",
)


def _url_host_path(url: str) -> str:
    """Return a lowercased ``host + path`` string for hint matching.

    Best-effort: parse errors fall back to the raw lowercased URL so a weird
    URL still gets screened by the sensitive denylist.
    """
    low = (url or "").strip().lower()
    try:
        parsed = urlparse(low)
        host = parsed.netloc or ""
        path = parsed.path or ""
        combined = f"{host}{path}".strip()
        return combined or low
    except Exception:  # pragma: no cover - defensive
        return low


def is_sensitive(url: str) -> bool:
    """True when *url* matches a domain-critical hint (always preserve).

    Only meaningful for real navigable URLs — ``about:``/``data:``/newtab
    URLs never contain a matchable host and return False.
    """
    low = (url or "").strip().lower()
    if not low or low in _BLANK_URLS:
        return False
    if low.startswith(("about:", "data:", "chrome:", "brave:", "edge:")):
        # Internal pages are handled by their own class; the denylist is for
        # navigable web content only.
        return False
    hay = _url_host_path(url)
    return any(hint in hay for hint in _SENSITIVE_HINTS)


def classify_url(url: str) -> str:
    """Classify a target URL into one of the ``CLASS_*`` buckets.

    * ``about:blank`` (and empty)                 -> ``blank``
    * ``chrome://newtab/`` and fork variants      -> ``newtab``
    * navigable URL matching the sensitive list   -> ``sensitive``
    * ``data:text/html,...``                      -> ``scratch``
    * anything else (http/https/file/chrome://…)  -> ``real``
    """
    low = (url or "").strip().lower()
    if low in _BLANK_URLS:
        return CLASS_BLANK
    if low.rstrip("/") in _NEWTAB_URLS:
        return CLASS_NEWTAB
    if is_sensitive(url):
        return CLASS_SENSITIVE
    if low.startswith("data:text/html"):
        return CLASS_SCRATCH
    return CLASS_REAL


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------


@dataclass
class TargetDecision:
    """A per-target close/skip decision. ``url_class`` is safe to log."""

    target_id: str
    type: str
    url_class: str
    action: str  # "close" | "skip"
    reason: str


@dataclass
class PrunePlan:
    """Result of :func:`plan_prune` — pure, no I/O performed yet."""

    decisions: List[TargetDecision] = field(default_factory=list)
    to_close: List[str] = field(default_factory=list)
    preserved_safety_target: Optional[str] = None

    @property
    def counts(self) -> Dict[str, int]:
        """Sanitized counts safe to emit in logs (no URLs, no titles)."""
        c: Dict[str, int] = {
            "total": len(self.decisions),
            "close": len(self.to_close),
            "skip": sum(1 for d in self.decisions if d.action == "skip"),
        }
        for d in self.decisions:
            c[f"class:{d.url_class}"] = c.get(f"class:{d.url_class}", 0) + 1
        return c


def _normalize_target(raw: Dict[str, Any]) -> Dict[str, str]:
    """Normalize a CDP target dict from either HTTP ``/json`` or
    ``Target.getTargets`` shape into ``{target_id, type, url, title}``.
    """
    target_id = str(raw.get("id") or raw.get("targetId") or "")
    return {
        "target_id": target_id,
        "type": str(raw.get("type") or ""),
        "url": str(raw.get("url") or ""),
        "title": str(raw.get("title") or ""),
    }


def _pick_preserve(candidates: Sequence[TargetDecision]) -> TargetDecision:
    """Choose which disposable page to keep when all pages are disposable.

    Prefer the lane's intentional initial ``about:blank``, then a new-tab
    page, then anything else — deterministic given input order.
    """
    for cls in (CLASS_BLANK, CLASS_NEWTAB, CLASS_SCRATCH):
        for d in candidates:
            if d.url_class == cls:
                return d
    return candidates[0]


def plan_prune(
    targets: Sequence[Dict[str, Any]],
    *,
    allow_scratch: bool = False,
) -> PrunePlan:
    """Decide which targets to close, purely (no network).

    Rules:
      1. Only ``type == "page"`` targets are ever considered.
      2. Closable classes are ``blank`` + ``newtab`` (plus ``scratch`` when
         ``allow_scratch``). ``real`` and ``sensitive`` are always skipped.
      3. At least one ``page`` target survives per call — if every page would
         be closed, one disposable page is preserved.
    """
    decisions: List[TargetDecision] = []
    closable = {CLASS_BLANK, CLASS_NEWTAB}
    if allow_scratch:
        closable.add(CLASS_SCRATCH)

    for raw in targets:
        t = _normalize_target(raw)
        if not t["target_id"]:
            # Can't act on a target with no id; record and skip.
            decisions.append(
                TargetDecision("", t["type"], "unknown", "skip", "no-target-id")
            )
            continue
        if t["type"] != PAGE_TYPE:
            decisions.append(
                TargetDecision(
                    t["target_id"], t["type"], "non_page", "skip", "not-a-page-target"
                )
            )
            continue
        cls = classify_url(t["url"])
        if cls == CLASS_SENSITIVE:
            decisions.append(
                TargetDecision(t["target_id"], t["type"], cls, "skip", "sensitive")
            )
        elif cls in closable:
            decisions.append(
                TargetDecision(
                    t["target_id"], t["type"], cls, "close", f"disposable:{cls}"
                )
            )
        else:
            reason = "real" if cls == CLASS_REAL else f"preserve:{cls}"
            decisions.append(
                TargetDecision(t["target_id"], t["type"], cls, "skip", reason)
            )

    # --- One-page preservation (per browser process) -----------------------
    page_decisions = [d for d in decisions if d.type == PAGE_TYPE]
    surviving = [d for d in page_decisions if d.action == "skip"]
    closing = [d for d in page_decisions if d.action == "close"]
    preserved: Optional[str] = None
    if page_decisions and not surviving and closing:
        keep = _pick_preserve(closing)
        keep.action = "skip"
        keep.reason = "preserve-last-page"
        preserved = keep.target_id

    to_close = [d.target_id for d in decisions if d.action == "close"]
    return PrunePlan(
        decisions=decisions, to_close=to_close, preserved_safety_target=preserved
    )


# ---------------------------------------------------------------------------
# I/O seam
# ---------------------------------------------------------------------------


class CdpClient(Protocol):
    """Minimal CDP target list/close surface. Injectable for tests."""

    def list_targets(self) -> List[Dict[str, Any]]:
        ...

    def close_target(self, target_id: str) -> bool:
        ...


class HttpCdpClient:
    """Default client using the DevTools HTTP JSON endpoints.

    * ``GET {base}/json``               -> list targets
    * ``GET {base}/json/close/{id}``    -> close a target

    These endpoints are exposed by Chromium-family browsers (Brave included)
    started with ``--remote-debugging-port`` bound to a literal loopback
    address, which is exactly how the coordinator lanes launch. Uses stdlib
    ``urllib`` so the module stays import-light on the kanban lifecycle path.
    """

    def __init__(self, base_url: str, *, timeout: float = 5.0) -> None:
        self.base_url = normalize_base_url(base_url)
        self.timeout = timeout

    def _get(self, path: str) -> tuple[int, str]:
        import urllib.request

        url = f"{self.base_url}{path}"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # noqa: S310
            body = resp.read().decode("utf-8", "replace")
            return resp.status, body

    def list_targets(self) -> List[Dict[str, Any]]:
        status, body = self._get("/json")
        if status != 200:
            raise RuntimeError(f"/json returned HTTP {status}")
        data = json.loads(body)
        if not isinstance(data, list):
            raise RuntimeError("/json did not return a list")
        return data

    def close_target(self, target_id: str) -> bool:
        try:
            status, _ = self._get(f"/json/close/{target_id}")
            return status == 200
        except Exception as exc:
            logger.debug("close_target %s failed: %s", target_id[:8], exc)
            return False


def normalize_base_url(endpoint: str) -> str:
    """Coerce any accepted CDP endpoint form into an ``http://host:port`` base.

    Accepts ``http(s)://host:port[/...]``, ``ws(s)://host:port/devtools/...``,
    and bare ``host:port``. Strips any path so ``/json`` can be appended.
    """
    raw = (endpoint or "").strip()
    if not raw:
        return ""
    low = raw.lower()
    if low.startswith(("ws://", "wss://")):
        raw = ("https://" if low.startswith("wss://") else "http://") + raw.split("://", 1)[1]
    elif not low.startswith(("http://", "https://")):
        raw = "http://" + raw
    parsed = urlparse(raw)
    scheme = parsed.scheme or "http"
    netloc = parsed.netloc or parsed.path  # bare host:port lands in .path
    return f"{scheme}://{netloc}".rstrip("/")


# ---------------------------------------------------------------------------
# Endpoint pruning
# ---------------------------------------------------------------------------


def _sanitized_lane(base_url: str) -> str:
    """Return just ``host:port`` for logging (never any path/query)."""
    try:
        parsed = urlparse(base_url)
        return parsed.netloc or base_url
    except Exception:  # pragma: no cover - defensive
        return base_url


def prune_endpoint(
    endpoint: str,
    *,
    dry_run: bool = True,
    allow_scratch: bool = False,
    client: Optional[CdpClient] = None,
    timeout: float = 5.0,
) -> Dict[str, Any]:
    """List, classify, and (unless *dry_run*) close disposable tabs on one lane.

    Returns a sanitized summary dict — ``lane`` (host:port), ``dry_run``,
    per-class ``counts``, ``closed`` / ``close_failed`` counts, and
    ``preserved_last_page`` — containing no URLs, titles, or page data.
    Never raises; connection/parse failures are reported via an ``error`` key.
    """
    lane = _sanitized_lane(normalize_base_url(endpoint))
    summary: Dict[str, Any] = {
        "lane": lane,
        "dry_run": dry_run,
        "allow_scratch": allow_scratch,
        "counts": {},
        "closed": 0,
        "close_failed": 0,
        "preserved_last_page": False,
    }
    try:
        cdp = client or HttpCdpClient(endpoint, timeout=timeout)
        targets = cdp.list_targets()
    except Exception as exc:
        summary["error"] = f"{type(exc).__name__}: {exc}"
        logger.info("cdp-prune lane=%s unreachable: %s", lane, summary["error"])
        return summary

    plan = plan_prune(targets, allow_scratch=allow_scratch)
    summary["counts"] = plan.counts
    summary["preserved_last_page"] = plan.preserved_safety_target is not None

    if dry_run:
        logger.info(
            "cdp-prune DRY-RUN lane=%s would_close=%d counts=%s preserved_last_page=%s",
            lane,
            len(plan.to_close),
            _fmt_counts(plan.counts),
            summary["preserved_last_page"],
        )
        summary["would_close"] = len(plan.to_close)
        return summary

    closed = 0
    failed = 0
    for target_id in plan.to_close:
        if cdp.close_target(target_id):
            closed += 1
        else:
            failed += 1
    summary["closed"] = closed
    summary["close_failed"] = failed
    logger.info(
        "cdp-prune lane=%s closed=%d failed=%d counts=%s preserved_last_page=%s",
        lane,
        closed,
        failed,
        _fmt_counts(plan.counts),
        summary["preserved_last_page"],
    )
    return summary


def _fmt_counts(counts: Dict[str, int]) -> str:
    """Compact ``k=v`` rendering of the sanitized counts for a log line."""
    return " ".join(f"{k}={v}" for k, v in sorted(counts.items()))


# ---------------------------------------------------------------------------
# Runtime gating + resolution (kanban lifecycle entrypoint)
# ---------------------------------------------------------------------------


def _env_true(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


def is_enabled() -> bool:
    """Master switch for the runtime trigger. Off unless explicitly enabled."""
    return _env_true("HERMES_CDP_PRUNE_ENABLED", False)


def is_dry_run() -> bool:
    """Dry-run unless explicitly disabled — safe default even once enabled."""
    return _env_true("HERMES_CDP_PRUNE_DRY_RUN", True)


def allow_scratch() -> bool:
    return _env_true("HERMES_CDP_PRUNE_SCRATCH", False)


def _prune_timeout() -> float:
    try:
        return max(1.0, min(float(os.environ.get("HERMES_CDP_PRUNE_TIMEOUT", "5")), 30.0))
    except (TypeError, ValueError):
        return 5.0


def resolve_endpoints() -> List[str]:
    """Resolve which CDP endpoint(s) to prune.

    Precedence:
      1. ``HERMES_CDP_PRUNE_ENDPOINT`` — comma-separated explicit endpoints.
      2. The worker's connected browser via ``BROWSER_CDP_URL`` /
         ``browser.cdp_url`` (the lane this worker was actually driving).

    Returns a de-duplicated list of ``http://host:port`` bases; empty when
    nothing resolves (pruner then no-ops safely).
    """
    explicit = os.environ.get("HERMES_CDP_PRUNE_ENDPOINT", "").strip()
    raw_endpoints: List[str] = []
    if explicit:
        raw_endpoints.extend(part for part in explicit.split(",") if part.strip())
    else:
        try:
            from tools.browser_tool import _get_cdp_override

            override = (_get_cdp_override() or "").strip()
            if override:
                raw_endpoints.append(override)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("cdp-prune endpoint resolution failed: %s", exc)

    seen: Dict[str, None] = {}
    for ep in raw_endpoints:
        base = normalize_base_url(ep)
        if base:
            seen.setdefault(base, None)
    return list(seen.keys())


def prune_after_transition(
    *,
    event: str,
    task_id: str = "",
    board: Optional[str] = None,
    assignee: Optional[str] = None,
    run_id: Optional[int] = None,
    profile_name: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Gated, best-effort entrypoint fired from the kanban lifecycle.

    Returns a sanitized aggregate summary, or ``None`` when the feature is
    disabled or no endpoint resolves. Never raises — a misbehaving pruner
    must never affect a board state transition.

    ``event`` is ``"completed"`` or ``"blocked"``. Blocking only prunes when
    ``HERMES_CDP_PRUNE_ON_BLOCK`` is set, and then only ever the strictly
    disposable blank/newtab set (scratch is never closed on block).
    """
    try:
        if not is_enabled():
            return None
        if event == "blocked" and not _env_true("HERMES_CDP_PRUNE_ON_BLOCK", False):
            return None

        endpoints = resolve_endpoints()
        if not endpoints:
            logger.debug("cdp-prune: enabled but no endpoint resolved; skipping")
            return None

        # On block, force the most conservative rule set regardless of the
        # scratch flag; completion honors the operator's scratch preference.
        scratch_ok = allow_scratch() if event == "completed" else False
        dry = is_dry_run()
        timeout = _prune_timeout()

        results: List[Dict[str, Any]] = []
        for endpoint in endpoints:
            results.append(
                prune_endpoint(
                    endpoint,
                    dry_run=dry,
                    allow_scratch=scratch_ok,
                    timeout=timeout,
                )
            )

        agg = {
            "event": event,
            "task_id": task_id,
            "dry_run": dry,
            "lanes": len(results),
            "closed": sum(r.get("closed", 0) for r in results),
            "would_close": sum(r.get("would_close", 0) for r in results),
            "results": results,
        }
        logger.info(
            "cdp-prune event=%s task=%s lanes=%d dry_run=%s closed=%d would_close=%d",
            event,
            task_id or "?",
            agg["lanes"],
            dry,
            agg["closed"],
            agg["would_close"],
        )
        return agg
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("cdp-prune prune_after_transition swallowed error: %s", exc)
        return None


# ---------------------------------------------------------------------------
# CLI (dry-run by default)
# ---------------------------------------------------------------------------


def _build_arg_parser():
    import argparse

    p = argparse.ArgumentParser(
        prog="python -m tools.cdp_prune",
        description=(
            "Safely prune disposable (about:blank / newtab) tabs on a CDP lane. "
            "Dry-run by default: prints sanitized counts and closes nothing."
        ),
    )
    p.add_argument(
        "--endpoint",
        action="append",
        default=None,
        help=(
            "CDP endpoint (http://host:port, ws://host:port/devtools/..., or "
            "host:port). Repeatable. Defaults to resolve_endpoints() "
            "(HERMES_CDP_PRUNE_ENDPOINT or the connected browser)."
        ),
    )
    p.add_argument(
        "--no-dry-run",
        dest="dry_run",
        action="store_false",
        help="Actually close disposable tabs (default is dry-run).",
    )
    p.add_argument(
        "--scratch",
        action="store_true",
        help="Also close synthetic data:text/html scratch tabs (conservative off by default).",
    )
    p.add_argument("--timeout", type=float, default=5.0, help="Per-call HTTP timeout (s).")
    p.set_defaults(dry_run=True)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _build_arg_parser().parse_args(argv)
    endpoints = args.endpoint or resolve_endpoints()
    if not endpoints:
        print(
            json.dumps(
                {
                    "error": "no endpoint (pass --endpoint or set "
                    "HERMES_CDP_PRUNE_ENDPOINT / browser.cdp_url)"
                }
            )
        )
        return 2
    results = [
        prune_endpoint(
            ep,
            dry_run=args.dry_run,
            allow_scratch=args.scratch,
            timeout=args.timeout,
        )
        for ep in endpoints
    ]
    print(json.dumps({"dry_run": args.dry_run, "results": results}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
