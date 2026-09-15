// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const apiMocks = vi.hoisted(() => ({
  getConfig: vi.fn(),
  getSchema: vi.fn(),
  getDefaults: vi.fn(),
  getConfigRaw: vi.fn(),
  getStatus: vi.fn(),
}));

vi.mock("@/lib/api", () => ({ api: apiMocks }));
vi.mock("@/contexts/usePageHeader", () => ({
  usePageHeader: () => ({ setEnd: vi.fn() }),
}));
vi.mock("@nous-research/ui/hooks/use-toast", () => ({
  useToast: () => ({ toast: null, showToast: vi.fn() }),
}));
vi.mock("@/i18n", () => ({
  useI18n: () => ({
    t: {
      common: { clear: "Clear", form: "Form", save: "Save", saving: "Saving", search: "Search" },
      config: {
        categories: {},
        configImported: "Imported",
        configPath: "Config path",
        configSaved: "Saved",
        confirmResetScope: "Reset {scope}",
        exportConfig: "Export",
        failedToLoadRaw: "Failed raw",
        failedToSave: "Failed",
        failedToSaveYaml: "YAML failed",
        fields: "{count} fields",
        filters: "Filters",
        importConfig: "Import",
        invalidJson: "Invalid JSON",
        noFieldsMatch: "No fields",
        rawYaml: "Raw YAML",
        resetDefaults: "Reset defaults",
        resetScopeToast: "Reset {scope}",
        resetScopeTooltip: "Reset {scope}",
        searchResults: "Search results",
        sections: "Sections",
        yamlConfigSaved: "YAML saved",
      },
    },
  }),
}));

import ConfigPage from "./ConfigPage";

describe("ConfigPage terminal provider config", () => {
  let container: HTMLDivElement;
  let root: Root;

  beforeEach(() => {
    vi.clearAllMocks();
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
    apiMocks.getConfig.mockResolvedValue({
      terminal: {
        backend: "plugin_a",
        backends: {
          plugin_a: { token: "a-secret" },
          plugin_b: { token: "b-secret" },
        },
      },
    });
    apiMocks.getSchema.mockResolvedValue({
      fields: {
        "terminal.backend": {
          type: "select",
          category: "terminal",
          options: ["plugin_a", "plugin_b"],
        },
        "terminal.backends.plugin_a.token": {
          type: "secret",
          category: "terminal",
          terminal_backend: "plugin_a",
        },
        "terminal.backends.plugin_b.token": {
          type: "secret",
          category: "terminal",
          terminal_backend: "plugin_b",
        },
      },
      category_order: ["terminal"],
    });
    apiMocks.getDefaults.mockResolvedValue({});
    apiMocks.getConfigRaw.mockResolvedValue({ yaml: "", path: "" });
    apiMocks.getStatus.mockResolvedValue({ config_path: "" });
  });

  afterEach(() => {
    act(() => root.unmount());
    container.remove();
  });

  it("shows only the selected backend fields and masks secret values", async () => {
    await act(async () => root.render(<ConfigPage />));

    expect(container.querySelector('input[type="password"][value="a-secret"]')).not.toBeNull();
    expect(container.querySelector('input[value="b-secret"]')).toBeNull();
  });
});
