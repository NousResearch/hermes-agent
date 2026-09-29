import { afterEach, describe, expect, it, vi } from "vitest";

// timeAgo resolves its Intl tag through the shared runtime-format locale
// (the app registers it at boot; tests register fa explicitly).
import { setRuntimeFormatLocale } from "@hermes/shared/i18n";
import { isoTimeAgo, timeAgo } from "./utils";

setRuntimeFormatLocale("fa");

// timeAgo drives the session-list relative-time chips through
// Intl.RelativeTimeFormat under the fa locale: numeric:"auto" renders
// word forms («دیروز») inside the 24–48h bucket and count forms
// («۵ روز پیش») beyond it. The yesterday branch once passed +1 instead
// of -1, rendering «فردا» (tomorrow) for a 30h-old session — the kind of
// bug the RTL snapshot suite catches only if a fixture row happens to
// sit in the bucket, so pin the buckets directly here.
describe("timeAgo", () => {
  afterEach(() => {
    vi.useRealTimers();
  });

  it("renders the 24–48h bucket as «دیروز», never «فردا»", () => {
    vi.setSystemTime(new Date("2026-09-29T12:00:00Z"));
    const now = Math.floor(Date.now() / 1000);
    expect(timeAgo(now - 30 * 3600)).toBe("دیروز");
  });

  it("keeps hour chips inside 24h and day counts beyond 48h", () => {
    vi.setSystemTime(new Date("2026-09-29T12:00:00Z"));
    const now = Math.floor(Date.now() / 1000);
    expect(timeAgo(now - 3 * 3600)).toBe("۳ ساعت پیش");
    expect(timeAgo(now - 8 * 3600)).toBe("۸ ساعت پیش");
    expect(timeAgo(now - 126 * 3600)).toBe("۵ روز پیش");
  });

  it("renders seconds-old sessions as the auto «اکنون» form", () => {
    // numeric:"auto" collapses 0 seconds to CLDR fa's "now" word (اکنون);
    // the desktop's minutesAgo uses a different RTF style and renders
    // همین حالا there — the two surfaces are not expected to match.
    vi.setSystemTime(new Date("2026-09-29T12:00:00Z"));
    const now = Math.floor(Date.now() / 1000);
    expect(timeAgo(now - 5)).toBe("اکنون");
  });
});

describe("isoTimeAgo", () => {
  afterEach(() => {
    vi.useRealTimers();
  });

  it("maps the yesterday bucket through the same auto form", () => {
    vi.setSystemTime(new Date("2026-09-29T12:00:00Z"));
    const iso = new Date(Date.now() - 30 * 3600 * 1000).toISOString();
    expect(isoTimeAgo(iso)).toBe("دیروز");
  });
});
