// @vitest-environment jsdom

import { act, type ReactNode } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { I18nProvider } from "@/i18n";
import { registerSlot, unregisterPluginSlots } from "@/plugins/slots";
import ConfigPage from "./ConfigPage";

const apiMocks = vi.hoisted(() => ({
  getConfig: vi.fn(),
  getSchema: vi.fn(),
  getDefaults: vi.fn(),
  getConfigRaw: vi.fn(),
  getStatus: vi.fn(),
  saveConfig: vi.fn(),
  saveConfigRaw: vi.fn(),
}));
const headerMock = vi.hoisted(() => ({ end: null as ReactNode }));

vi.mock("@/lib/api", () => ({ api: apiMocks }));
vi.mock("@/contexts/usePageHeader", () => ({
  usePageHeader: () => ({
    setEnd: (node: ReactNode) => {
      headerMock.end = node;
    },
  }),
}));
vi.mock("@nous-research/ui/hooks/use-toast", () => ({
  useToast: () => ({ toast: null, showToast: vi.fn() }),
}));

describe("Config plugin sections", () => {
  let container: HTMLDivElement;
  let root: Root;
  let headerContainer: HTMLDivElement;
  let headerRoot: Root;

  beforeEach(() => {
    apiMocks.getConfig.mockResolvedValue({ general: { model: "test" } });
    apiMocks.getSchema.mockResolvedValue({
      fields: {
        "general.model": {
          category: "general",
          type: "string",
          default: "",
          description: "Default model",
        },
      },
      category_order: ["general"],
    });
    apiMocks.getDefaults.mockResolvedValue({ general: { model: "" } });
    apiMocks.getConfigRaw.mockResolvedValue({ path: "/tmp/config.yaml", yaml: "" });
    apiMocks.getStatus.mockResolvedValue({ config_path: "/tmp/config.yaml" });
    registerSlot("buzz-platform", "config:section:buzz", () => (
      <div data-testid="buzz-policy-panel">Buzz policy panel</div>
    ));
    container = document.createElement("div");
    headerContainer = document.createElement("div");
    document.body.append(container);
    document.body.append(headerContainer);
    root = createRoot(container);
    headerRoot = createRoot(headerContainer);
    headerMock.end = null;
  });

  afterEach(async () => {
    await act(async () => root.unmount());
    await act(async () => headerRoot.unmount());
    unregisterPluginSlots("buzz-platform");
    container.remove();
    headerContainer.remove();
    vi.clearAllMocks();
  });

  it("lists a plugin-owned section and renders it in the selected Config pane", async () => {
    await act(async () => root.render(<I18nProvider><ConfigPage /></I18nProvider>));
    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
      await Promise.resolve();
    });

    const buzzSection = Array.from(container.querySelectorAll("button")).find(
      (button) => button.textContent?.includes("Buzz"),
    );
    expect(buzzSection).toBeDefined();
    expect(container.querySelector('[data-testid="buzz-policy-panel"]')).toBeNull();

    await act(async () => buzzSection?.click());

    expect(container.querySelector('[data-testid="buzz-policy-panel"]')?.textContent).toBe(
      "Buzz policy panel",
    );
  });

  it("restores core Save and Reset controls when searching from a plugin section", async () => {
    await act(async () => root.render(<I18nProvider><ConfigPage /></I18nProvider>));
    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
      await Promise.resolve();
    });

    const buzzSection = Array.from(container.querySelectorAll("button")).find(
      (button) => button.textContent?.includes("Buzz"),
    );
    await act(async () => buzzSection?.click());
    expect(container.querySelector('[data-testid="buzz-policy-panel"]')).not.toBeNull();

    await act(async () => headerRoot.render(headerMock.end));
    const search = headerContainer.querySelector("input");
    expect(search).not.toBeNull();
    await act(async () => {
      const setValue = Object.getOwnPropertyDescriptor(
        HTMLInputElement.prototype,
        "value",
      )?.set;
      setValue?.call(search, "model");
      search?.dispatchEvent(new Event("input", { bubbles: true }));
    });

    const buttons = Array.from(container.querySelectorAll("button"));
    expect(buttons.some((button) => button.textContent?.trim() === "Save")).toBe(true);
    expect(
      buttons.some((button) => /reset/i.test(button.getAttribute("aria-label") ?? "")),
    ).toBe(true);
  });
});
