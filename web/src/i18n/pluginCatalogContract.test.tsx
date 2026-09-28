// @vitest-environment jsdom
import { act } from "react";
import { createRoot } from "react-dom/client";
import { renderToStaticMarkup } from "react-dom/server";
import { afterEach, beforeAll, describe, expect, it, vi } from "vitest";

import { exposePluginSDK, getPluginComponent } from "../plugins/registry";
import { I18nContext, formatTranslation, resolveTranslations } from "./runtime";
import type { Locale } from "./types";

async function loadPlugin(name: string) {
  const url = new URL(
    `../../../plugins/${name}/dashboard/dist/index.js`,
    import.meta.url,
  ).href;
  await import(/* @vite-ignore */ url);
  return getPluginComponent(name)!;
}

beforeAll(() => exposePluginSDK());
afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe("bundled Dashboard plugin localization", () => {
  it.each(["kanban", "hermes-achievements"] as const)(
    "loads and renders the real %s bundle through the host SDK in independent locales",
    async (name) => {
      const Page = await loadPlugin(name);
      expect(Page).toBeDefined();
      if (!Page) throw new Error(`Plugin did not register: ${name}`);
      const render = (locale: Locale) =>
        renderToStaticMarkup(
          <I18nContext.Provider
            value={{
              locale,
              t: resolveTranslations(locale),
              format: formatTranslation,
              setLocale: async () => {},
            }}
          >
            <Page />
          </I18nContext.Provider>,
        );
      const english = render("en");
      const simplified = render("zh");
      expect(english).not.toBe(simplified);
      expect(english).not.toContain("[object Object]");
      expect(simplified).not.toContain("[object Object]");
      const expected =
        name === "hermes-achievements"
          ? resolveTranslations("zh").achievements.hero.title
          : resolveTranslations("zh").kanban.loading;
      expect(simplified).toContain(expected);
      // Another language resolves independently, without mutating either baseline.
      render("zh-hant");
      expect(render("en")).toBe(english);
    },
  );
});

it("keeps literal extension data and loaded state when the plugin language changes", async () => {
  vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
  const Page = await loadPlugin("hermes-achievements");
  const name = "$& {tier_part}";
  const fetch = vi
    .spyOn(window.__HERMES_PLUGIN_SDK__!, "fetchJSON")
    .mockResolvedValue({
      achievements: [
        {
          id: "extension-achievement",
          name,
          category: "Extension",
          state: "unlocked",
          unlocked: true,
          description: "Original extension text",
        },
      ],
      unlocked_count: 1,
      total_count: 1,
    });
  const container = document.createElement("div");
  const root = createRoot(container);
  const render = (locale: Locale) =>
    root.render(
      <I18nContext.Provider
        value={{
          locale,
          t: resolveTranslations(locale),
          format: formatTranslation,
          setLocale: async () => {},
        }}
      >
        <Page />
      </I18nContext.Provider>,
    );
  try {
    await act(async () => render("en"));
    expect(
      container.querySelector(".ha-share-trigger")?.getAttribute("aria-label"),
    ).toBe(`Share ${name}`);
    await act(async () => render("zh"));
    expect(
      container.querySelector(".ha-share-trigger")?.getAttribute("aria-label"),
    ).toBe(
      formatTranslation(
        resolveTranslations("zh").achievements.card.share_label,
        { name },
      ),
    );
    expect(container.textContent).toContain("Original extension text");
    expect(fetch).toHaveBeenCalledTimes(1);
  } finally {
    await act(async () => root.unmount());
    fetch.mockRestore();
  }
});
