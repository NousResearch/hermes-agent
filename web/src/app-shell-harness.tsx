// @vitest-environment jsdom
import { vi } from "vitest";
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { MemoryRouter } from "react-router";

/**
 * Shared harness for app-shell (App.tsx) behaviour tests. The test file owns
 * its `vi.mock("@/lib/api")` (hoisted); this module only wires the provider
 * stack and mounts the shell at a lightweight route so shell-level a11y
 * assertions observe the same tree the browser would.
 */
export async function renderAppShell(
  initialEntry = "/docs",
): Promise<{ container: HTMLDivElement; root: Root }> {
  const [
    { default: App },
    { I18nProvider },
    { ThemeProvider },
    { SystemActionsProvider },
    { ProfileProvider },
  ] = await Promise.all([
    import("./App"),
    import("@/i18n"),
    import("@/themes"),
    import("@/contexts/SystemActions"),
    import("@/contexts/ProfileProvider"),
  ]);

  const container = document.createElement("div");
  document.body.append(container);
  const root = createRoot(container);
  await act(async () => {
    root.render(
      <MemoryRouter initialEntries={[initialEntry]}>
        <I18nProvider>
          <ThemeProvider>
            <SystemActionsProvider>
              <ProfileProvider>
                <App />
              </ProfileProvider>
            </SystemActionsProvider>
          </ThemeProvider>
        </I18nProvider>
      </MemoryRouter>,
    );
  });
  // Let lazy route chunks and async provider bootstraps settle.
  for (let i = 0; i < 10; i++) {
    await act(async () => {
      await new Promise((resolve) => setTimeout(resolve, 20));
    });
  }
  return { container, root };
}

/** Stub `matchMedia` with a synthetic viewport: below the 1024px breakpoint. */
export function stubMobileViewport() {
  vi.stubGlobal("matchMedia", (query: string) => ({
    media: query,
    matches: query.includes("max-width"),
    addEventListener() {},
    removeEventListener() {},
  }));
}

/** Stub `matchMedia` reporting desktop (≥ 1024px). */
export function stubDesktopViewport() {
  vi.stubGlobal("matchMedia", (query: string) => ({
    media: query,
    matches: query.includes("min-width"),
    addEventListener() {},
    removeEventListener() {},
  }));
}

/** A `vi.fn` resolving to `{}` for any api method the shell happens to call. */
export function apiFallback(base: Record<string, (...args: unknown[]) => unknown>) {
  return new Proxy(base, {
    get(target, prop: string) {
      if (prop in target) return target[prop];
      return vi.fn(async () => ({}));
    },
  });
}

/** Click a plain mouse event (jsdom has no user-event package here). */
export function clickElement(el: Element | null) {
  if (!el) throw new Error("element not rendered");
  el.dispatchEvent(new MouseEvent("click", { bubbles: true, cancelable: true }));
}
