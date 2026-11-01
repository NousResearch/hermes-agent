// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { MemoryRouter } from "react-router";
import { afterEach, describe, expect, it, vi } from "vitest";

const LINES = ["2026-01-01 INFO a: one", "2026-01-01 ERROR b: two"];
const getLogs = vi.hoisted(() => vi.fn());
vi.mock("@/lib/api", () => ({ api: { getLogs } }));
vi.mock("@/plugins", () => ({ PluginSlot: () => null }));

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

let root: Root | null = null;

afterEach(() => {
  act(() => root?.unmount());
  root = null;
  document.body.innerHTML = "";
  vi.unstubAllGlobals();
});

describe("LogsPage copy button", () => {
  it("copies the loaded log lines joined by newlines", async () => {
    getLogs.mockResolvedValue({ file: "agent", lines: LINES });
    const writeText = vi.fn().mockResolvedValue(undefined);
    vi.stubGlobal("navigator", { clipboard: { writeText } } as unknown as Navigator);
    vi.stubGlobal("isSecureContext", true);

    const [{ default: LogsPage }, { I18nProvider }, { PageHeaderProvider }] =
      await Promise.all([
        import("./LogsPage"),
        import("@/i18n"),
        import("@/contexts/PageHeaderProvider"),
      ]);
    const container = document.createElement("div");
    document.body.append(container);
    root = createRoot(container);
    await act(async () =>
      root!.render(
        <I18nProvider>
          <MemoryRouter>
            <PageHeaderProvider pluginTabs={[]}>
              <LogsPage />
            </PageHeaderProvider>
          </MemoryRouter>
        </I18nProvider>,
      ),
    );

    const btn = () => document.querySelector<HTMLButtonElement>('button[aria-label="Copy"]');
    for (let i = 0; i < 50 && (!btn() || btn()!.disabled); i++) {
      await act(async () => {
        await new Promise((r) => setTimeout(r, 20));
      });
    }
    await act(async () => {
      btn()!.click();
    });

    expect(writeText).toHaveBeenCalledWith(LINES.join("\n"));
    expect(document.querySelector('button[aria-label="Copied"]')).not.toBeNull();
  });
});
