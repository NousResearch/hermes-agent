// @vitest-environment jsdom
// WCAG 2.2 SC 2.5.8 (Target Size, Minimum): every non-exempt sidebar
// control must render with at least a 24x24 CSS-px hit area. jsdom cannot
// compute layout, so these tests pin the rendered classes that produce the
// 24px minimum on each target (min-h/min-w on the bare-sized controls,
// py-1.5 on the switcher triggers). The inline `Config` sentence link in
// ModelsPage is exempt under the SC 2.5.8 inline-text exception and is
// intentionally not asserted here. See issue #70983.

import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import type { ReactNode } from "react";

import { I18nProvider } from "@/i18n";
import { ThemeProvider } from "@/themes";
import { AuthWidget } from "./AuthWidget";
import { LanguageSwitcher } from "./LanguageSwitcher";
import { ThemeSwitcher } from "./ThemeSwitcher";
import { SidebarFooter } from "./SidebarFooter";

vi.mock("@/lib/api", () => ({
  api: {
    getAuthMe: () =>
      Promise.resolve({
        user_id: "user-abcdef",
        email: "",
        display_name: "",
        provider: "nous",
      }),
    logout: () => Promise.resolve(),
    getThemes: () => Promise.resolve({ themes: [], active: null }),
    setTheme: () => Promise.resolve(),
    getFontPref: () => Promise.resolve({ fontId: null }),
    setFontPref: () => Promise.resolve(),
  },
}));

let container: HTMLDivElement;
let root: Root;

async function render(ui: ReactNode) {
  container = document.createElement("div");
  document.body.append(container);
  root = createRoot(container);
  await act(async () => root.render(ui));
}

afterEach(async () => {
  await act(async () => root?.unmount());
  container?.remove();
});

/** The fix pins each target's rendered class list, so assert on exact
 *  whitespace-delimited tokens (py-1 must not satisfy a py-1.5 check). */
function hasClass(el: Element, token: string): boolean {
  return el.className.split(/\s+/).includes(token);
}

describe("sidebar control target sizes (WCAG 2.5.8, issue #70983)", () => {
  beforeEach(() => {
    window.sessionStorage.clear();
    window.localStorage.clear();
    Object.defineProperty(window, "__HERMES_AUTH_REQUIRED__", {
      configurable: true,
      value: true,
    });
    vi.stubGlobal(
      "matchMedia",
      () =>
        ({
          matches: false,
          media: "",
          addEventListener() {},
          removeEventListener() {},
        }) as unknown as MediaQueryList,
    );
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("Log out button renders a >=24px hit area (min-h/min-w 24px)", async () => {
    await render(
      <I18nProvider>
        <AuthWidget />
      </I18nProvider>,
    );
    const btn = container.querySelector('button[aria-label="Log out"]');
    expect(btn).not.toBeNull();
    expect(hasClass(btn!, "min-h-[24px]")).toBe(true);
    expect(hasClass(btn!, "min-w-[24px]")).toBe(true);
  });

  it("Switch language trigger uses py-1.5 (>=24px computed height)", async () => {
    await render(
      <I18nProvider>
        <LanguageSwitcher />
      </I18nProvider>,
    );
    const btn = container.querySelector('button[aria-haspopup="listbox"]');
    expect(btn).not.toBeNull();
    expect(hasClass(btn!, "py-1.5")).toBe(true);
    expect(hasClass(btn!, "py-1")).toBe(false);
  });

  it("Switch theme trigger uses py-1.5 (>=24px computed height)", async () => {
    await render(
      <ThemeProvider>
        <I18nProvider>
          <ThemeSwitcher />
        </I18nProvider>
      </ThemeProvider>,
    );
    const btn = container.querySelector('button[aria-haspopup="listbox"]');
    expect(btn).not.toBeNull();
    expect(hasClass(btn!, "py-1.5")).toBe(true);
    expect(hasClass(btn!, "py-1")).toBe(false);
  });

  it("Nous Research footer link renders a >=24px hit area (min-h 24px)", async () => {
    await render(
      <I18nProvider>
        <SidebarFooter status={null} />
      </I18nProvider>,
    );
    const link = container.querySelector('a[href="https://nousresearch.com"]');
    expect(link).not.toBeNull();
    expect(hasClass(link!, "min-h-[24px]")).toBe(true);
  });
});
