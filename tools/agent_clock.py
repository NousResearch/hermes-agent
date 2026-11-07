"""One clock for every agent-sync action.

Why this exists
---------------
Two seats stamped the same instant differently, and a merge read as if it had
happened after the review that authorised it.  ``otc-console``/``oua-product``
measured ``oto-server`` +8 s off ``api.github.com``'s ``Date`` header while both
hosts were NTP-healthy, and a 4.5-minute inversion followed.  Host time is not
evidence; a checked time is.

The contract
------------
``probe()`` returns a reading that always says where its time came from:

* ``authority:http-date`` - checked against an HTTP ``Date`` header just now.
* ``authority:http-date:cached`` - checked within ``CACHE_TTL``; the offset is
  live, the check is recent, and the age travels with the value.
* ``host:fallback`` - the authority could not be reached. **``degraded`` is True.**
  A fallback that returns host time indistinguishable from a checked value is
  worse than no fallback: a stamp from it looks identical to a stamp that was
  checked.  Callers must disclose ``degraded``.

Nothing is configured per profile: this module is code in the install, so every
profile - existing or created later - inherits it without opting in.

CLI
---
    python -m tools.agent_clock            # one line, human readable
    python -m tools.agent_clock --json     # the full reading
    python -m tools.agent_clock --guard    # exit 1 if the clock cannot be trusted
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Optional

#: Where authoritative time is read from, in order. Any host that answers with a
#: ``Date`` header works; a Cloudflare ``cdn-cgi/trace`` is preferred because it
#: states the epoch outright (``ts=``) and does not rate-limit.  GitHub's API is
#: last on the list for a measured reason: unauthenticated it allows ~60 requests
#: an hour per address, and 22 seats checking it exhausted that in minutes - the
#: authority then answered 403 and every reading went DEGRADED.
AUTHORITY_CHAIN = [
    url.strip()
    for url in os.environ.get(
        "HERMES_CLOCK_AUTHORITY",
        "https://one.one.one.one/cdn-cgi/trace,https://www.cloudflare.com,https://api.github.com",
    ).split(",")
    if url.strip()
]

#: Seconds to wait for the authority before falling back.
TIMEOUT_SECONDS = float(os.environ.get("HERMES_CLOCK_TIMEOUT", "1.5"))

#: How long a checked offset stays usable before it is re-measured.
CACHE_TTL_SECONDS = int(os.environ.get("HERMES_CLOCK_CACHE_TTL", "300"))

SOURCE_AUTHORITY = "authority:http-date"
SOURCE_AUTHORITY_CACHED = "authority:http-date:cached"
SOURCE_FALLBACK = "host:fallback"


@dataclass(frozen=True)
class ClockReading:
    """A timestamp that carries its own provenance."""

    utc_iso: str
    source: str
    degraded: bool
    offset_seconds: float
    authority: Optional[str]
    checked_at_iso: Optional[str]
    check_age_seconds: Optional[float]
    host_utc_iso: str

    def stamp(self, prefix: str = "") -> str:
        """The ``clock=`` stamp format every seat copies rather than invents."""
        core = f"{self.utc_iso} clock={self.source}"
        if self.degraded:
            core += " clock_degraded=true"
        return f"{prefix}{core}" if prefix else core

    def as_dict(self) -> dict:
        return asdict(self)


def _cache_path() -> Path:
    """One measurement for the whole host, not one per profile.

    The cache is deliberately *not* under ``HERMES_HOME``: every profile has its
    own ``HERMES_HOME``, and a per-profile copy is exactly how two seats end up
    with the same nominal source and different values.  A first cut used
    ``HERMES_HOME`` and produced -0.08 s in one profile and -15.4 s in another,
    on one machine, within the same minute.  The measurement is a property of the
    host, so it is stored once.
    """
    return Path.home() / ".hermes" / "runtime" / "agent_clock.json"


def _read_cache() -> Optional[dict]:
    path = _cache_path()
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def _write_cache(offset_seconds: float, authority: str, checked_at: float) -> None:
    path = _cache_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "offset_seconds": offset_seconds,
                    "authority": authority,
                    "checked_at_epoch": checked_at,
                },
                handle,
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except OSError:
        # A cache that cannot be written must not break the caller; the reading
        # it would have produced is simply re-measured next time.
        pass


def _measure_offset() -> Optional[tuple[float, float, float, str]]:
    """Return ``(offset_seconds, checked_at_epoch, rtt_seconds, authority)``.

    The chain is tried in order and the first authority that answers wins.  Two
    samples are taken from it and the faster round-trip is kept: a single sample
    behind a caching proxy, or one that sat in a stalled queue, can carry a stale
    timestamp, and that error does *not* cancel with the midpoint correction.
    This host once read -15.4 s on one sample and -0.08 s on the next, on one
    machine against one authority, which cannot both be true of a clock.

    A sample whose round-trip exceeds ``MAX_RTT_SECONDS`` is refused rather than
    guessed at: a slow authority is a reason to say DEGRADED, not to publish a
    number that looks measured.
    """
    for authority in AUTHORITY_CHAIN:
        best: Optional[tuple[float, float, float]] = None
        for _ in range(2):
            sample = _one_sample(authority)
            if sample is None:
                continue
            if best is None or sample[2] < best[2]:
                best = sample
        if best is not None:
            return best[0], best[1], best[2], authority
    return None


MAX_RTT_SECONDS = float(os.environ.get("HERMES_CLOCK_MAX_RTT", "5"))


def _one_sample(authority: str) -> Optional[tuple[float, float, float]]:
    """One bracketed request against one authority; ``(offset, checked_at, rtt)``."""
    # A cache buster and explicit no-cache headers: a proxy that answers from a
    # cache returns the timestamp of the response it stored, not the one now.
    stamp = f"{time.time():.6f}"
    url = f"{authority}{'&' if '?' in authority else '?'}hermes_clock={stamp}"
    request = urllib.request.Request(
        url,
        method="GET",
        headers={
            "User-Agent": "hermes-agent-clock/1",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
        },
    )
    started = time.time()
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            header = response.headers.get("Date")
            body = ""
            if "cdn-cgi/trace" in authority:
                body = response.read(2048).decode("utf-8", "replace")
    except Exception:
        return None
    finished = time.time()
    rtt = finished - started
    if rtt > MAX_RTT_SECONDS:
        return None

    # A trace states the epoch outright; anything else is read from Date.
    authority_epoch: Optional[float] = None
    for line in body.splitlines():
        if line.startswith("ts="):
            try:
                authority_epoch = float(line[3:])
            except ValueError:
                authority_epoch = None
            break
    if authority_epoch is None:
        if not header:
            return None
        try:
            authority_dt = parsedate_to_datetime(header)
        except (TypeError, ValueError):
            return None
        if authority_dt.tzinfo is None:
            authority_dt = authority_dt.replace(tzinfo=timezone.utc)
        authority_epoch = authority_dt.timestamp()

    midpoint = (started + finished) / 2.0
    return authority_epoch - midpoint, finished, rtt


def probe(force: bool = False) -> ClockReading:
    """Read the clock, preferring a fresh check over a cached offset."""
    host_now = time.time()
    host_iso = datetime.fromtimestamp(host_now, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    cached = None if force else _read_cache()
    if cached:
        age = host_now - float(cached.get("checked_at_epoch", 0) or 0)
        if 0 <= age <= CACHE_TTL_SECONDS:
            offset = float(cached.get("offset_seconds", 0.0))
            checked_epoch = float(cached.get("checked_at_epoch", host_now))
            return ClockReading(
                utc_iso=_iso(host_now + offset),
                source=SOURCE_AUTHORITY_CACHED,
                degraded=False,
                offset_seconds=round(offset, 3),
                authority=cached.get("authority"),
                checked_at_iso=_iso(checked_epoch),
                check_age_seconds=round(age, 1),
                host_utc_iso=host_iso,
            )

    measured = _measure_offset()
    if measured is not None:
        offset, checked_epoch, _rtt, authority = measured
        _write_cache(offset, authority, checked_epoch)
        return ClockReading(
            utc_iso=_iso(host_now + offset),
            source=SOURCE_AUTHORITY,
            degraded=False,
            offset_seconds=round(offset, 3),
            authority=authority,
            checked_at_iso=_iso(checked_epoch),
            check_age_seconds=0.0,
            host_utc_iso=host_iso,
        )

    # Authority unreachable: use the host clock, and SAY SO.
    return ClockReading(
        utc_iso=host_iso,
        source=SOURCE_FALLBACK,
        degraded=True,
        offset_seconds=0.0,
        authority=None,
        checked_at_iso=None,
        check_age_seconds=None,
        host_utc_iso=host_iso,
    )


def _iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def now() -> ClockReading:
    """Convenience alias: the reading a caller should stamp with."""
    return probe()


def skew_report() -> dict:
    """Everything ``hermes clock`` prints, including the number that caused this."""
    reading = probe()
    return {
        "utc": reading.utc_iso,
        "source": reading.source,
        "degraded": reading.degraded,
        "offset_seconds": reading.offset_seconds,
        "host_utc": reading.host_utc_iso,
        "authority": reading.authority,
        "checked_at": reading.checked_at_iso,
        "check_age_seconds": reading.check_age_seconds,
        "stamp": reading.stamp(),
    }


def _main(argv: Optional[list] = None) -> int:
    import sys

    args = list(sys.argv[1:] if argv is None else argv)
    report = skew_report()
    if "--json" in args:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"utc        {report['utc']}")
        print(f"source     {report['source']}")
        print(f"degraded   {str(report['degraded']).lower()}")
        print(f"host utc   {report['host_utc']}")
        print(f"skew       {report['offset_seconds']:+.1f}s against {report['authority'] or 'no authority'}")
        print(f"stamp      {report['stamp']}")
    if "--guard" in args:
        if report["degraded"]:
            print(
                "clock guard: DEGRADED - the authority could not be reached, so any "
                "stamp taken now would carry host time it cannot vouch for.",
                file=sys.stderr,
            )
            return 1
        if abs(report["offset_seconds"]) >= float(os.environ.get("HERMES_CLOCK_MAX_SKEW", "2")):
            print(
                f"clock guard: SKEW - the host is {report['offset_seconds']:+.1f}s off "
                f"{report['authority']}; instants recorded here can invert an order.",
                file=sys.stderr,
            )
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
