// @vitest-environment jsdom
import { afterEach, describe, expect, it, vi } from "vitest";

import {
  validOffer,
  DISMISSED_STORAGE_KEY,
  LAST_SHOWN_STORAGE_KEY,
  MAX_DISMISSED_IDS,
  offerIsEligible,
  pageIsIdle,
  readDismissedIds,
  readPageActivity,
  recordDismissal,
  recordShown,
  RESHOW_QUIET_MS,
  type OfferStorage,
} from "./in-app-offer";

function memoryStorage(seed: Record<string, string> = {}): OfferStorage {
  const data = { ...seed };
  return {
    getItem: (key) => data[key] ?? null,
    setItem: (key, value) => {
      data[key] = value;
    },
  };
}

const NOW = Date.UTC(2026, 10, 20, 12);

describe("offerIsEligible", () => {
  it("allows a fresh offer, never a dismissed one", () => {
    const storage = memoryStorage();
    expect(offerIsEligible(storage, "offer_1", NOW)).toBe(true);

    recordDismissal(storage, "offer_1");

    expect(offerIsEligible(storage, "offer_1", NOW + RESHOW_QUIET_MS * 10)).toBe(false);
  });

  it("holds every offer back for a day after one was shown", () => {
    const storage = memoryStorage();
    recordShown(storage, NOW);

    expect(offerIsEligible(storage, "offer_2", NOW + RESHOW_QUIET_MS - 1)).toBe(false);
    expect(offerIsEligible(storage, "offer_2", NOW + RESHOW_QUIET_MS)).toBe(true);
  });

  it("treats corrupt storage as a fresh browser", () => {
    const storage = memoryStorage({ [DISMISSED_STORAGE_KEY]: "{", [LAST_SHOWN_STORAGE_KEY]: "x" });

    expect(offerIsEligible(storage, "offer_1", NOW)).toBe(true);
  });
});

describe("recordDismissal", () => {
  it("keeps another tab's dismissals, dedupes and caps", () => {
    const storage = memoryStorage({ [DISMISSED_STORAGE_KEY]: JSON.stringify(["other"]) });
    recordDismissal(storage, "offer_1");
    expect(readDismissedIds(storage)).toEqual(["other", "offer_1"]);

    for (let i = 0; i < MAX_DISMISSED_IDS + 5; i++) recordDismissal(storage, `o${i}`);
    const ids = readDismissedIds(storage);
    expect(ids).toHaveLength(MAX_DISMISSED_IDS);
    expect(new Set(ids).size).toBe(ids.length);
  });
});

describe("pageIsIdle / readPageActivity", () => {
  afterEach(() => {
    document.body.innerHTML = "";
    vi.restoreAllMocks();
  });

  it("is idle only when visible, focused, with nothing open and no typing", () => {
    const idle = { visibleAndFocused: true, otherDialogOpen: false, typing: false };
    expect(pageIsIdle(idle)).toBe(true);
    expect(pageIsIdle({ ...idle, visibleAndFocused: false })).toBe(false);
    expect(pageIsIdle({ ...idle, otherDialogOpen: true })).toBe(false);
    expect(pageIsIdle({ ...idle, typing: true })).toBe(false);
  });

  it("sees the dashboard's own modal dialogs and a focused field", () => {
    vi.spyOn(document, "hasFocus").mockReturnValue(true);
    document.body.innerHTML = '<div role="dialog" aria-modal="true"></div><textarea id="t"></textarea>';
    document.getElementById("t")?.focus();

    const activity = readPageActivity(document);

    expect(activity.otherDialogOpen).toBe(true);
    expect(activity.typing).toBe(true);
  });

  it("ignores closed menus and focused buttons", () => {
    vi.spyOn(document, "hasFocus").mockReturnValue(true);
    document.body.innerHTML = '<div role="menu" data-state="closed"></div><button id="b">x</button>';
    document.getElementById("b")?.focus();

    const activity = readPageActivity(document);

    expect(activity.otherDialogOpen).toBe(false);
    expect(activity.typing).toBe(false);
  });
});

describe("validOffer", () => {
  const OK = {
    id: "offer_1",
    title: "T",
    body: "B",
    cta_label: "Go",
    cta_url: "https://portal.nousresearch.com/promos/x",
  };

  it("accepts a complete offer with an https CTA", () => {
    expect(validOffer(OK)).toEqual(OK);
  });

  it.each(["http://portal.nousresearch.com", "javascript:alert(1)", "/local/path", "https://"])(
    "rejects the CTA %s (window.open would run or open it)",
    (cta_url) => {
      expect(validOffer({ ...OK, cta_url })).toBeNull();
    },
  );

  it("keeps a valid claim deadline and drops a malformed one", () => {
    expect(validOffer({ ...OK, claimable_until: "2026-12-15T12:00:00.000Z" })?.claimable_until).toBe(
      "2026-12-15T12:00:00.000Z",
    );
    expect(validOffer({ ...OK, claimable_until: "soon" })).toEqual(OK);
  });

  it("rejects blank copy and non-objects", () => {
    expect(validOffer({ ...OK, title: " " })).toBeNull();
    expect(validOffer(null)).toBeNull();
    expect(validOffer("offer")).toBeNull();
  });
});
