// @vitest-environment jsdom
// The dashboard's in-app offer dialog: it reads the memo, refreshes after boot, opens only on a
// settled idle moment, and a close sticks.

import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { I18nProvider } from "@/i18n";
import * as offers from "@/lib/in-app-offer";
import { DISMISSED_STORAGE_KEY, FIRST_REFRESH_DELAY_MS, IDLE_SETTLE_MS } from "@/lib/in-app-offer";
import { InAppOfferDialog } from "./InAppOfferDialog";

vi.mock("@/contexts/useProfileScope", () => ({ useProfileScope: () => ({ profile: "default" }) }));

const OFFER = {
  id: "offer_1",
  title: "Welcome back",
  body: "Line one.\n\nLine two.",
  cta_label: "Claim the offer",
  cta_url: "https://portal.nousresearch.com/billing",
};

let container: HTMLDivElement;
let root: Root;

async function mount() {
  container = document.createElement("div");
  document.body.append(container);
  root = createRoot(container);
  await act(async () => root.render(<I18nProvider><InAppOfferDialog /></I18nProvider>));
}

const dialog = () => document.querySelector('[data-testid="in-app-offer-dialog"]');

function memoryStorage(): Storage {
  const data = new Map<string, string>();
  return {
    get length() {
      return data.size;
    },
    clear: () => data.clear(),
    getItem: (key) => data.get(key) ?? null,
    key: (index) => [...data.keys()][index] ?? null,
    removeItem: (key) => void data.delete(key),
    setItem: (key, value) => void data.set(key, String(value)),
  };
}

beforeEach(() => {
  vi.useFakeTimers();
  // jsdom runs without an origin under this node-default config, so localStorage can be
  // undefined. Stub it, as SidebarTargetSize.test.tsx and ChatPage.test.tsx do.
  vi.stubGlobal("localStorage", memoryStorage());
  vi.spyOn(document, "hasFocus").mockReturnValue(true);
});

afterEach(async () => {
  await act(async () => root.unmount());
  container.remove();
  vi.useRealTimers();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe("InAppOfferDialog", () => {
  it("opens only after the page has stayed idle for the settle window", async () => {
    vi.spyOn(offers, "getInAppOffer").mockResolvedValue({ in_app_offer: OFFER });
    await mount();

    await act(async () => vi.advanceTimersByTimeAsync(IDLE_SETTLE_MS - 1));
    expect(dialog()).toBeNull();

    await act(async () => vi.advanceTimersByTimeAsync(1));
    expect(dialog()?.textContent).toContain("Welcome back");
    // Focus is on the panel, not on a button a stray keypress would activate.
    expect(document.activeElement?.tagName).toBe("DIV");
  });

  it("restarts the clock on every keypress", async () => {
    vi.spyOn(offers, "getInAppOffer").mockResolvedValue({ in_app_offer: OFFER });
    await mount();

    await act(async () => vi.advanceTimersByTimeAsync(IDLE_SETTLE_MS - 10));
    document.dispatchEvent(new KeyboardEvent("keydown", { key: "a" }));
    await act(async () => vi.advanceTimersByTimeAsync(IDLE_SETTLE_MS - 10));
    expect(dialog()).toBeNull();

    await act(async () => vi.advanceTimersByTimeAsync(20));
    expect(dialog()).not.toBeNull();
  });

  it("waits while another modal dialog is open", async () => {
    vi.spyOn(offers, "getInAppOffer").mockResolvedValue({ in_app_offer: OFFER });
    const other = document.createElement("div");
    other.setAttribute("role", "dialog");
    other.setAttribute("aria-modal", "true");
    document.body.append(other);
    await mount();

    await act(async () => vi.advanceTimersByTimeAsync(IDLE_SETTLE_MS * 3));
    expect(dialog()).toBeNull();

    other.remove();
    await act(async () => vi.advanceTimersByTimeAsync(IDLE_SETTLE_MS));
    expect(dialog()).not.toBeNull();
  });

  it("closing it records the dismissal so it never returns", async () => {
    vi.spyOn(offers, "getInAppOffer").mockResolvedValue({ in_app_offer: OFFER });
    await mount();
    await act(async () => vi.advanceTimersByTimeAsync(IDLE_SETTLE_MS));

    const later = [...document.querySelectorAll("button")].find((b) => b.textContent === "Maybe later");
    await act(async () => later?.click());

    expect(dialog()).toBeNull();
    expect(localStorage.getItem(DISMISSED_STORAGE_KEY)).toContain("offer_1");
  });

  it("asks for a refresh after boot settles", async () => {
    const spy = vi.spyOn(offers, "getInAppOffer").mockResolvedValue({ in_app_offer: null });
    await mount();

    expect(spy).toHaveBeenCalledWith(false);
    expect(spy).not.toHaveBeenCalledWith(true);

    await act(async () => vi.advanceTimersByTimeAsync(FIRST_REFRESH_DELAY_MS));
    expect(spy).toHaveBeenCalledWith(true);
  });

  it("stays silent on an older backend without the route", async () => {
    vi.spyOn(offers, "getInAppOffer").mockRejectedValue(new Error("404"));
    await mount();

    await act(async () => vi.advanceTimersByTimeAsync(FIRST_REFRESH_DELAY_MS + IDLE_SETTLE_MS));
    expect(dialog()).toBeNull();
  });

  it("closes when the Portal withdraws the offer", async () => {
    const spy = vi.spyOn(offers, "getInAppOffer").mockResolvedValue({ in_app_offer: OFFER });
    await mount();
    await act(async () => vi.advanceTimersByTimeAsync(IDLE_SETTLE_MS));
    expect(dialog()).not.toBeNull();

    spy.mockResolvedValue({ in_app_offer: null });
    await act(async () => vi.advanceTimersByTimeAsync(FIRST_REFRESH_DELAY_MS));

    expect(dialog()).toBeNull();
    // Withdrawn is not dismissed: a later campaign with the same id could still show.
    expect(localStorage.getItem(DISMISSED_STORAGE_KEY)).toBeNull();
  });
});
