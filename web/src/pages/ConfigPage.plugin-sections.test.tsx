// @vitest-environment jsdom

import { act } from "react";
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

vi.mock("@/lib/api", () => ({ api: apiMocks }));
vi.mock("@/contexts/usePageHeader", () => ({
  usePageHeader: () => ({ setEnd: vi.fn() }),
}));
vi.mock("@nous-research/ui/hooks/use-toast", () => ({
  useToast: () => ({ toast: null, showToast: vi.fn() }),
}));

describe("Config plugin sections", () => {
  let container: HTMLDivElement;
  let root: Root;

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
    document.body.append(container);
    root = createRoot(container);
  });

  afterEach(async () => {
    await act(async () => root.unmount());
    unregisterPluginSlots("buzz-platform");
    container.remove();
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
});
