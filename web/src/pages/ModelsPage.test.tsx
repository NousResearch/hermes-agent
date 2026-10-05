// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, describe, expect, it, vi } from "vitest";

const apiMocks = vi.hoisted(() => ({
  getAuxiliaryModels: vi.fn(() => new Promise<never>(() => {})),
  getConfig: vi.fn(() => new Promise<never>(() => {})),
  getModelsAnalytics: vi.fn(() => new Promise<never>(() => {})),
  getMoaModels: vi.fn(() => new Promise<never>(() => {})),
}));

vi.mock("@/lib/api", () => ({ api: apiMocks }));
vi.mock("@/contexts/usePageHeader", () => ({
  usePageHeader: () => ({ setAfterTitle: vi.fn(), setEnd: vi.fn() }),
}));
vi.mock("@/i18n", () => ({
  useI18n: () => ({ t: { common: { refresh: "Refresh" } } }),
}));
vi.mock("@/plugins", () => ({ PluginSlot: () => null }));

let container: HTMLDivElement;
let root: Root;
(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

afterEach(async () => {
  await act(async () => root?.unmount());
  container?.remove();
});

describe("ModelsPage auxiliary tasks", () => {
  it("shows micro-compaction inheriting compression while other tasks use main", async () => {
    const { default: ModelsPage } = await import("./ModelsPage");
    container = document.createElement("div");
    document.body.append(container);
    root = createRoot(container);

    await act(async () => root.render(<ModelsPage />));
    const configure = Array.from(container.querySelectorAll("button")).find(
      (button) => button.textContent?.trim() === "Configure" && !button.disabled,
    );
    if (!configure) throw new Error("auxiliary Configure button not rendered");
    await act(async () => configure.dispatchEvent(new MouseEvent("click", { bubbles: true })));

    const labels = Array.from(container.querySelectorAll("span"));
    const microCompaction = labels.find(
      (element) => element.textContent === "Micro-compaction",
    );
    expect(microCompaction).toBeDefined();
    expect(microCompaction?.parentElement?.parentElement?.textContent).toContain(
      "auto (inherit compression)",
    );

    const vision = labels.find((element) => element.textContent === "Vision");
    expect(vision?.parentElement?.parentElement?.textContent).toContain(
      "auto (use main model)",
    );
  });
});
