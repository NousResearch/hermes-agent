// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { I18nProvider, useI18n } from "./context";

// jsdom runs without an origin here (per-file @vitest-environment jsdom on a
// node-default config), so localStorage is undefined. Stub it before the
// provider's useState initializer runs.
const localStorageMock = (() => {
  let store: Record<string, string> = {};
  return {
    getItem: (key: string) => store[key] ?? null,
    setItem: (key: string, value: string) => {
      store[key] = String(value);
    },
    removeItem: (key: string) => {
      delete store[key];
    },
    clear: () => {
      store = {};
    },
  };
})();

// React only routes updates through act() when this flag is set.
(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT =
  true;

let container: HTMLDivElement;
let root: Root;

function Probe() {
  const { locale, setLocale } = useI18n();
  return (
    <button onClick={() => setLocale("ja")} type="button">
      {locale}
    </button>
  );
}

async function render() {
  container = document.createElement("div");
  document.body.appendChild(container);
  root = createRoot(container);
  await act(async () => {
    root.render(
      <I18nProvider>
        <Probe />
      </I18nProvider>,
    );
  });
}

function setBrowserLanguages(languages: string[]) {
  Object.defineProperty(window.navigator, "languages", {
    configurable: true,
    value: languages,
  });
  Object.defineProperty(window.navigator, "language", {
    configurable: true,
    value: languages[0] ?? "en-US",
  });
}

beforeEach(() => {
  vi.stubGlobal("localStorage", localStorageMock);
  localStorageMock.clear();
  setBrowserLanguages(["en-US"]);
});

afterEach(async () => {
  await act(async () => root?.unmount());
  container?.remove();
  vi.unstubAllGlobals();
});

describe("I18nProvider locale selection", () => {
  it("prefers a stored choice over the browser language", async () => {
    localStorageMock.setItem("hermes-locale", "de");
    setBrowserLanguages(["zh-CN"]);

    await render();

    expect(container.textContent).toBe("de");
  });

  it("falls back to the browser language on a first visit", async () => {
    setBrowserLanguages(["zh-CN", "en"]);

    await render();

    expect(container.textContent).toBe("zh");
  });

  it("maps Traditional Chinese tags to zh-hant", async () => {
    setBrowserLanguages(["zh-TW"]);

    await render();

    expect(container.textContent).toBe("zh-hant");
  });

  it("matches regional variants through their base language", async () => {
    setBrowserLanguages(["pt-BR"]);

    await render();

    expect(container.textContent).toBe("pt");
  });

  it("ignores unsupported browser languages", async () => {
    setBrowserLanguages(["xx-YY"]);

    await render();

    expect(container.textContent).toBe("en");
  });

  it("persists a manual switch for the next visit", async () => {
    await render();

    const button = container.querySelector("button");
    expect(button).not.toBeNull();
    await act(async () => {
      button?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    });

    expect(container.textContent).toBe("ja");
    expect(localStorageMock.getItem("hermes-locale")).toBe("ja");
  });
});
