// @vitest-environment jsdom

import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";

import { I18nProvider, useI18n } from "./context";

let container: HTMLDivElement;
let root: Root;
let originalLanguage: PropertyDescriptor | undefined;
const storage = new Map<string, string>();

Object.defineProperty(globalThis, "localStorage", {
  configurable: true,
  value: {
    clear: () => storage.clear(),
    getItem: (key: string) => storage.get(key) ?? null,
    setItem: (key: string, value: string) => storage.set(key, value),
  },
});
globalThis.IS_REACT_ACT_ENVIRONMENT = true;

function LocaleProbe() {
  return <output>{useI18n().locale}</output>;
}

function setBrowserLanguage(language: string) {
  Object.defineProperty(navigator, "language", {
    configurable: true,
    value: language,
  });
}

async function renderLocale() {
  container = document.createElement("div");
  document.body.append(container);
  root = createRoot(container);
  await act(async () => root.render(<I18nProvider><LocaleProbe /></I18nProvider>));
  return container.querySelector("output")?.textContent;
}

beforeEach(() => {
  localStorage.clear();
  originalLanguage = Object.getOwnPropertyDescriptor(navigator, "language");
});

afterEach(async () => {
  await act(async () => root?.unmount());
  container?.remove();
  if (originalLanguage) Object.defineProperty(navigator, "language", originalLanguage);
});

describe("I18nProvider initial locale", () => {
  it("uses a supported browser language when no preference is stored", async () => {
    setBrowserLanguage("ja-JP");

    expect(await renderLocale()).toBe("ja");
  });

  it("keeps a stored preference ahead of the browser language", async () => {
    localStorage.setItem("hermes-locale", "de");
    setBrowserLanguage("ja-JP");

    expect(await renderLocale()).toBe("de");
  });
});
