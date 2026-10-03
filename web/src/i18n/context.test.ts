// @vitest-environment jsdom
import { act, useEffect, type ReactNode } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { I18nProvider, useI18n } from "@/i18n/context";
import type { Locale } from "@/i18n/types";

let container: HTMLDivElement;
let root: Root;

async function render(ui: ReactNode) {
  container = document.createElement("div");
  document.body.append(container);
  root = createRoot(container);
  await act(async () => root.render(ui));
}

function stubBrowserLanguage(tag: string) {
  Object.defineProperty(window.navigator, "language", {
    value: tag,
    configurable: true,
  });
}

function Probe({ onLocale }: { onLocale: (l: Locale) => void }) {
  const { locale } = useI18n();
  useEffect(() => onLocale(locale), [locale, onLocale]);
  return null;
}

async function renderLocale(navLanguage: string): Promise<Locale> {
  stubBrowserLanguage(navLanguage);
  let seen: Locale = "en";
  await render(
    <I18nProvider>
      <Probe onLocale={(l) => (seen = l)} />
    </I18nProvider>,
  );
  return seen;
}

beforeEach(() => {
  localStorage.clear();
  // ponytail: fixed stub, per-width matchMedia if the hook ever needs it
  vi.stubGlobal(
    "matchMedia",
    (query: string) => ({
      matches: false,
      media: query,
      onchange: null,
      addListener: () => {},
      removeListener: () => {},
      addEventListener: () => {},
      removeEventListener: () => {},
      dispatchEvent: () => false,
    }),
  );
});

afterEach(async () => {
  await act(async () => root?.unmount());
  container?.remove();
  vi.unstubAllGlobals();
});

describe("getInitialLocale", () => {
  it("uses the browser language when nothing is stored", async () => {
    expect(await renderLocale("ja-JP")).toBe("ja");
  });

  it("falls back to the base subtag", async () => {
    expect(await renderLocale("pt-BR")).toBe("pt");
  });

  it("lets an explicit stored choice win", async () => {
    localStorage.setItem("hermes-locale", "de");
    expect(await renderLocale("ja-JP")).toBe("de");
  });

  it("defaults to English when nothing matches", async () => {
    expect(await renderLocale("xx-YY")).toBe("en");
  });
});

describe("LanguageSwitcher trigger", () => {
  it("renders an icon so the button is visible below sm", async () => {
    const { LanguageSwitcher } = await import("@/components/LanguageSwitcher");
    await render(
      <I18nProvider>
        <LanguageSwitcher />
      </I18nProvider>,
    );
    const trigger = container.querySelector(
      'button[aria-haspopup="listbox"]',
    );
    expect(trigger?.querySelector("svg")).not.toBeNull();
  });
});
