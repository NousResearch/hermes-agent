// @vitest-environment jsdom
// I18nProvider behaviour: local-first resolution, the async swap to the server
// preference, and the picker → server write. `fetchJSON` is mocked so the timing
// of the server response is controlled explicitly, and the locale is read back
// from the DOM (no outer-variable reassignment during render).
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const fetchJSONMock = vi.fn();

vi.mock("@/lib/api", () => ({
  fetchJSON: (...args: unknown[]) => fetchJSONMock(...args),
}));

import { I18nProvider, useI18n } from "./context";
import { SUPPORTED_LOCALES } from "./resolve-locale";
import type { Locale } from "./types";

const LOCALE_ENDPOINT = "/api/dashboard/locale";

let container: HTMLDivElement;
let root: Root | undefined;
(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

/** Renders the active locale and one button per locale that calls setLocale. */
function Probe() {
  const { locale, setLocale } = useI18n();
  return (
    <div>
      <span data-testid="locale">{locale}</span>
      {SUPPORTED_LOCALES.map((id) => (
        <button key={id} data-locale={id} onClick={() => setLocale(id)} type="button">
          {id}
        </button>
      ))}
    </div>
  );
}

const currentLocale = (): Locale =>
  (document.querySelector('[data-testid="locale"]')?.textContent ?? "") as Locale;

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((r) => {
    resolve = r;
  });
  return { promise, resolve };
}

async function render() {
  container = document.createElement("div");
  document.body.append(container);
  root = createRoot(container);
  await act(async () => {
    root!.render(
      <I18nProvider>
        <Probe />
      </I18nProvider>,
    );
  });
}

async function pickLocale(id: Locale) {
  await act(async () => {
    document
      .querySelector(`button[data-locale="${id}"]`)!
      .dispatchEvent(new MouseEvent("click", { bubbles: true, cancelable: true }));
  });
}

/** Let the mocked fetch's `.then` continuations run inside act(). */
async function flush() {
  await act(async () => {
    await new Promise((resolve) => setTimeout(resolve, 0));
  });
}

beforeEach(() => {
  fetchJSONMock.mockReset();
  localStorage.clear();
  (window as { __HERMES_SESSION_TOKEN__?: string }).__HERMES_SESSION_TOKEN__ = "test-token";
});

afterEach(async () => {
  await act(async () => root?.unmount());
  root = undefined;
  container?.remove();
  localStorage.clear();
  delete (window as { __HERMES_SESSION_TOKEN__?: string }).__HERMES_SESSION_TOKEN__;
  delete (window as { __HERMES_AUTH_REQUIRED__?: boolean }).__HERMES_AUTH_REQUIRED__;
});

describe("I18nProvider", () => {
  it("resolves localStorage first, then swaps to the server preference", async () => {
    localStorage.setItem("hermes-locale", "zh-hant");
    const server = deferred<{ locale: string | null }>();
    fetchJSONMock.mockReturnValue(server.promise);

    await render();
    // First paint is local — no blank/flash while the server request is in flight.
    expect(currentLocale()).toBe("zh-hant");

    await act(async () => {
      server.resolve({ locale: "ja" });
      await server.promise;
    });
    await flush();
    expect(currentLocale()).toBe("ja");
  });

  it("keeps the local locale when the server has no preference", async () => {
    localStorage.setItem("hermes-locale", "fr");
    fetchJSONMock.mockResolvedValue({ locale: null });

    await render();
    await flush();
    expect(currentLocale()).toBe("fr");
  });

  it("writes an explicit pick to localStorage and PUTs it to the server", async () => {
    fetchJSONMock.mockResolvedValue({ locale: null });
    await render();
    await flush();
    fetchJSONMock.mockClear();

    await pickLocale("de");

    expect(localStorage.getItem("hermes-locale")).toBe("de");
    expect(currentLocale()).toBe("de");
    const [url, init] = fetchJSONMock.mock.calls[0] as [string, RequestInit];
    expect(url).toBe(LOCALE_ENDPOINT);
    expect(init.method).toBe("PUT");
    expect(JSON.parse(String(init.body))).toEqual({ locale: "de" });
  });

  it("does not let a late server response override an explicit pick", async () => {
    localStorage.setItem("hermes-locale", "en");
    const server = deferred<{ locale: string | null }>();
    fetchJSONMock.mockReturnValue(server.promise);

    await render();
    await pickLocale("fr");

    await act(async () => {
      server.resolve({ locale: "ja" });
      await server.promise;
    });
    await flush();

    expect(currentLocale()).toBe("fr");
  });
});
