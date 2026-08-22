// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

type Api = typeof import("@/lib/api").api;
type ConfigResult = Awaited<ReturnType<Api["getConfig"]>>;
type SchemaResult = Awaited<ReturnType<Api["getSchema"]>>;
type ConfigRawResult = Awaited<ReturnType<Api["getConfigRaw"]>>;
type StatusResult = Awaited<ReturnType<Api["getStatus"]>>;
type SaveConfigResult = Awaited<ReturnType<Api["saveConfig"]>>;
type SaveConfigRawResult = Awaited<ReturnType<Api["saveConfigRaw"]>>;

const statusResult = (configPath: string): StatusResult => ({
  active_sessions: 0,
  config_path: configPath,
  config_version: 0,
  env_path: "",
  gateway_exit_reason: null,
  gateway_health_url: null,
  gateway_pid: null,
  gateway_platforms: {},
  gateway_running: false,
  gateway_state: null,
  gateway_updated_at: null,
  hermes_home: "",
  latest_config_version: 0,
  release_date: "",
  version: "",
});

const profileState = vi.hoisted(() => ({ profile: "one" }));
const toastMocks = vi.hoisted(() => ({ showToast: vi.fn() }));
const apiMocks = vi.hoisted(() => ({
  getConfig: vi.fn<
    (...args: Parameters<Api["getConfig"]>) => Promise<ConfigResult>
  >(() => new Promise<ConfigResult>(() => {})),
  getSchema: vi.fn<
    (...args: Parameters<Api["getSchema"]>) => Promise<SchemaResult>
  >(() => new Promise<SchemaResult>(() => {})),
  getDefaults: vi.fn<
    (...args: Parameters<Api["getDefaults"]>) => Promise<ConfigResult>
  >(() => new Promise<ConfigResult>(() => {})),
  getConfigRaw: vi.fn<
    (...args: Parameters<Api["getConfigRaw"]>) => Promise<ConfigRawResult>
  >(() => new Promise<ConfigRawResult>(() => {})),
  getStatus: vi.fn<
    (...args: Parameters<Api["getStatus"]>) => Promise<StatusResult>
  >(() => new Promise<StatusResult>(() => {})),
  saveConfigRaw: vi.fn<
    (...args: Parameters<Api["saveConfigRaw"]>) => Promise<SaveConfigRawResult>
  >(() => new Promise<SaveConfigRawResult>(() => {})),
  saveConfig: vi.fn<
    (...args: Parameters<Api["saveConfig"]>) => Promise<SaveConfigResult>
  >(() => new Promise<SaveConfigResult>(() => {})),
}));

vi.mock("@/lib/api", () => ({ api: apiMocks }));
vi.mock("@/contexts/useProfileScope", () => ({
  useProfileScope: () => ({ profile: profileState.profile }),
}));
vi.mock("@/contexts/usePageHeader", () => ({
  usePageHeader: () => ({ setEnd: vi.fn() }),
}));
vi.mock("@nous-research/ui/hooks/use-toast", () => ({
  useToast: () => ({ toast: null, showToast: toastMocks.showToast }),
}));
vi.mock("@/i18n", () => ({
  useI18n: () => ({
    t: {
      common: {
        clear: "Clear",
        search: "Search",
        form: "Form",
        save: "Save",
        saving: "Saving",
      },
      config: {
        categories: {},
        failedToLoadRaw: "Failed to load raw config",
        configSaved: "Saved",
        failedToSave: "Failed",
        yamlConfigSaved: "YAML saved",
        failedToSaveYaml: "YAML failed",
        searchResults: "Search results",
        resetScopeToast: "Reset {scope}",
        configImported: "Imported",
        invalidJson: "Invalid JSON",
        configPath: "Config path",
        exportConfig: "Export",
        importConfig: "Import",
        resetScopeTooltip: "Reset {scope}",
        rawYaml: "Raw YAML",
        filters: "Filters",
        sections: "Sections",
        fields: "{count} fields",
        noFieldsMatch: "No fields match {query}",
        confirmResetScope: "Reset {scope}",
        resetDefaults: "Reset defaults",
      },
    },
  }),
}));

import ConfigPage from "./ConfigPage";

describe("ConfigPage profile scope", () => {
  let container: HTMLDivElement;
  let root: Root;

  beforeEach(() => {
    profileState.profile = "one";
    vi.clearAllMocks();
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
  });

  afterEach(() => {
    act(() => root.unmount());
    container.remove();
    vi.unstubAllGlobals();
  });

  it("refetches all profile-scoped config resources after a profile switch", async () => {
    await act(async () => {
      root.render(<ConfigPage />);
    });

    expect(apiMocks.getConfig).toHaveBeenCalledTimes(1);
    expect(apiMocks.getSchema).toHaveBeenCalledTimes(1);
    expect(apiMocks.getDefaults).toHaveBeenCalledTimes(1);
    expect(apiMocks.getConfigRaw).toHaveBeenCalledTimes(1);

    profileState.profile = "two";
    await act(async () => {
      root.render(<ConfigPage />);
    });

    expect(apiMocks.getConfig).toHaveBeenCalledTimes(2);
    expect(apiMocks.getSchema).toHaveBeenCalledTimes(2);
    expect(apiMocks.getDefaults).toHaveBeenCalledTimes(2);
    expect(apiMocks.getConfigRaw).toHaveBeenCalledTimes(2);
    expect(apiMocks.getSchema).toHaveBeenNthCalledWith(1, "one");
    expect(apiMocks.getSchema).toHaveBeenNthCalledWith(2, "two");
  });

  it("shows only the selected backend fields and masks secret values", async () => {
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
    apiMocks.getStatus.mockResolvedValue(statusResult(""));

    await act(async () => root.render(<ConfigPage />));

    expect(container.querySelector('input[type="password"][value="a-secret"]')).not.toBeNull();
    expect(container.querySelector('input[value="b-secret"]')).toBeNull();
  });

  it("disables YAML saving until the switched profile raw config loads", async () => {
    const pendingRaw = new Promise<never>(() => {});
    apiMocks.getConfig.mockResolvedValue({});
    apiMocks.getSchema.mockResolvedValue({ fields: {}, category_order: [] });
    apiMocks.getDefaults.mockResolvedValue({});
    apiMocks.getStatus.mockResolvedValue(statusResult(""));
    apiMocks.getConfigRaw
      .mockResolvedValueOnce({ yaml: "profile: A", path: "/profiles/A/config.yaml" })
      .mockResolvedValueOnce({ yaml: "profile: A", path: "/profiles/A/config.yaml" })
      .mockReturnValue(pendingRaw);
    apiMocks.saveConfigRaw.mockResolvedValue({ ok: true });

    await act(async () => root.render(<ConfigPage />));
    const yamlButton = [...container.querySelectorAll("button")].find(
      (button) => button.textContent === "YAML",
    );
    await act(async () => yamlButton?.click());
    const oldSaveButton = [...container.querySelectorAll("button")].find(
      (button) => button.textContent === "Save",
    ) as HTMLButtonElement;
    expect(oldSaveButton.disabled).toBe(false);

    profileState.profile = "two";
    await act(async () => root.render(<ConfigPage />));
    const newSaveButton = [...container.querySelectorAll("button")].find(
      (button) => button.textContent === "Save",
    ) as HTMLButtonElement;
    expect(newSaveButton.disabled).toBe(true);
    await act(async () => newSaveButton.click());

    expect(apiMocks.saveConfigRaw).not.toHaveBeenCalled();
    expect(container.querySelector("textarea")).toBeNull();
  });

  it("ignores an old profile post-YAML-save refresh after switching", async () => {
    let resolveRefresh!: (value: Record<string, unknown>) => void;
    const refresh = new Promise<Record<string, unknown>>((resolve) => {
      resolveRefresh = resolve;
    });
    apiMocks.getConfig
      .mockResolvedValueOnce({ value: "A" })
      .mockReturnValueOnce(refresh)
      .mockResolvedValueOnce({ value: "B" });
    apiMocks.getSchema.mockResolvedValue({
      fields: { value: { type: "string", category: "general" } },
      category_order: ["general"],
    });
    apiMocks.getDefaults.mockResolvedValue({ value: "default" });
    apiMocks.getConfigRaw.mockResolvedValue({ yaml: "value: A", path: "" });
    apiMocks.getStatus.mockResolvedValue(statusResult(""));
    apiMocks.saveConfigRaw.mockResolvedValue({ ok: true });

    await act(async () => {
      root.render(<ConfigPage />);
    });
    const yamlButton = [...container.querySelectorAll("button")].find(
      (button) => button.textContent === "YAML",
    );
    expect(yamlButton).toBeDefined();
    await act(async () => yamlButton?.click());

    const saveButton = [...container.querySelectorAll("button")].find(
      (button) => button.textContent === "Save",
    );
    expect(saveButton).toBeDefined();
    await act(async () => saveButton?.click());
    expect(apiMocks.getConfig).toHaveBeenCalledTimes(2);

    profileState.profile = "two";
    await act(async () => {
      root.render(<ConfigPage />);
    });
    expect(apiMocks.getConfig).toHaveBeenCalledTimes(3);
    await act(async () => resolveRefresh({ value: "stale A" }));
    const formButton = [...container.querySelectorAll("button")].find(
      (button) => button.textContent === "Form",
    );
    expect(formButton).toBeDefined();
    await act(async () => formButton?.click());

    expect(container.querySelector('input[value="B"]')).not.toBeNull();
    expect(container.querySelector('input[value="stale A"]')).toBeNull();
  });

  it("ignores stale initial profile responses after switching", async () => {
    type Deferred<T> = { promise: Promise<T>; resolve: (value: T) => void };
    const deferred = <T,>(): Deferred<T> => {
      let resolve!: (value: T) => void;
      const promise = new Promise<T>((done) => {
        resolve = done;
      });
      return { promise, resolve };
    };
    const aConfig = deferred<Record<string, unknown>>();
    const aSchema = deferred<{ fields: Record<string, unknown>; category_order: string[] }>();
    const aDefaults = deferred<Record<string, unknown>>();
    const aRaw = deferred<ConfigRawResult>();
    const aStatus = deferred<StatusResult>();
    apiMocks.getConfig
      .mockReturnValueOnce(aConfig.promise)
      .mockResolvedValueOnce({ value: "B" });
    apiMocks.getSchema
      .mockReturnValueOnce(aSchema.promise)
      .mockResolvedValueOnce({
        fields: { value: { type: "string", category: "general" } },
        category_order: ["general"],
      });
    apiMocks.getDefaults
      .mockReturnValueOnce(aDefaults.promise)
      .mockResolvedValueOnce({ value: "B default" });
    apiMocks.getConfigRaw
      .mockReturnValueOnce(aRaw.promise)
      .mockResolvedValueOnce({ yaml: "value: B", path: "" });
    apiMocks.getStatus
      .mockReturnValueOnce(aStatus.promise)
      .mockResolvedValueOnce(statusResult("/profiles/B/fallback.yaml"));

    await act(async () => root.render(<ConfigPage />));
    profileState.profile = "two";
    await act(async () => root.render(<ConfigPage />));
    expect(container.querySelector('input[value="B"]')).not.toBeNull();

    await act(async () => {
      aConfig.resolve({ value: "stale A" });
      aSchema.resolve({
        fields: { stale: { type: "string", category: "plugin-a" } },
        category_order: ["plugin-a"],
      });
      aDefaults.resolve({ value: "stale default" });
      aRaw.resolve({ yaml: "value: stale A", path: "/profiles/A/config.yaml" });
      aStatus.resolve(statusResult("/profiles/A/fallback.yaml"));
    });

    expect(container.querySelector('input[value="B"]')).not.toBeNull();
    expect(container.textContent).toContain("/profiles/B/fallback.yaml");
    expect(container.textContent).not.toContain("/profiles/A/config.yaml");
    expect(container.textContent).not.toContain("/profiles/A/fallback.yaml");
    expect(container.textContent).not.toContain("plugin-a");

    const resetButton = container.querySelector<HTMLButtonElement>(
      'button[aria-label="Reset General"]',
    );
    expect(resetButton).not.toBeNull();
    await act(async () => resetButton?.click());
    const confirmButton = [...document.querySelectorAll("button")].find(
      (button) => button.textContent === "Reset defaults",
    );
    expect(confirmButton).toBeDefined();
    await act(async () => confirmButton?.click());

    expect(container.querySelector('input[value="B default"]')).not.toBeNull();
    expect(container.querySelector('input[value="stale default"]')).toBeNull();
  });

  it("does not expose or save the old form while the new profile loads", async () => {
    const pendingConfig = new Promise<never>(() => {});
    const pendingSchema = new Promise<never>(() => {});
    apiMocks.getConfig
      .mockResolvedValueOnce({ value: "A" })
      .mockReturnValueOnce(pendingConfig);
    apiMocks.getSchema
      .mockResolvedValueOnce({
        fields: { value: { type: "string", category: "general" } },
        category_order: ["general"],
      })
      .mockReturnValueOnce(pendingSchema);
    apiMocks.getDefaults.mockResolvedValue({ value: "default" });
    apiMocks.getConfigRaw.mockResolvedValue({ yaml: "value: current", path: "" });
    apiMocks.getStatus.mockResolvedValue(statusResult(""));
    apiMocks.saveConfig.mockResolvedValue({ ok: true });

    await act(async () => root.render(<ConfigPage />));
    const oldSaveButton = [...container.querySelectorAll("button")].find(
      (button) => button.textContent === "Save",
    ) as HTMLButtonElement;
    expect(container.querySelector('input[value="A"]')).not.toBeNull();

    profileState.profile = "two";
    await act(async () => root.render(<ConfigPage />));
    expect(container.querySelector('input[value="A"]')).toBeNull();
    expect(
      [...container.querySelectorAll("button")].find(
        (button) => button.textContent === "Save",
      ),
    ).toBeUndefined();
    await act(async () => oldSaveButton.click());

    expect(apiMocks.saveConfig).not.toHaveBeenCalled();
  });

  it("does not apply normal-save completion state to a later profile", async () => {
    let resolveSave!: (value: SaveConfigResult) => void;
    const save = new Promise<SaveConfigResult>((resolve) => {
      resolveSave = resolve;
    });
    apiMocks.getConfig
      .mockResolvedValueOnce({ value: "A" })
      .mockResolvedValueOnce({ value: "B" });
    apiMocks.getSchema.mockResolvedValue({
      fields: { value: { type: "string", category: "general" } },
      category_order: ["general"],
    });
    apiMocks.getDefaults.mockResolvedValue({ value: "default" });
    apiMocks.getConfigRaw.mockResolvedValue({ yaml: "value: current", path: "" });
    apiMocks.getStatus.mockResolvedValue(statusResult(""));
    apiMocks.saveConfig.mockReturnValue(save);

    await act(async () => root.render(<ConfigPage />));
    const saveButton = [...container.querySelectorAll("button")].find(
      (button) => button.textContent === "Save",
    );
    expect(saveButton).toBeDefined();
    await act(async () => saveButton?.click());

    profileState.profile = "two";
    await act(async () => root.render(<ConfigPage />));
    await act(async () => resolveSave({ ok: true }));

    expect(toastMocks.showToast).not.toHaveBeenCalled();
    expect(container.querySelector('input[value="B"]')).not.toBeNull();
    expect(
      [...container.querySelectorAll("button")].some(
        (button) => button.textContent === "Saving",
      ),
    ).toBe(false);
  });

  it("ignores an old profile FileReader import callback", async () => {
    class FakeFileReader {
      static latest: FakeFileReader;
      result = "";
      onload: null | (() => void) = null;

      constructor() {
        FakeFileReader.latest = this;
      }

      readAsText() {}
    }
    vi.stubGlobal("FileReader", FakeFileReader);
    apiMocks.getConfig
      .mockResolvedValueOnce({ value: "A" })
      .mockResolvedValueOnce({ value: "B" });
    apiMocks.getSchema.mockResolvedValue({
      fields: { value: { type: "string", category: "general" } },
      category_order: ["general"],
    });
    apiMocks.getDefaults.mockResolvedValue({ value: "default" });
    apiMocks.getConfigRaw.mockResolvedValue({ yaml: "value: current", path: "" });
    apiMocks.getStatus.mockResolvedValue(statusResult(""));

    await act(async () => root.render(<ConfigPage />));
    const input = container.querySelector('input[type="file"]') as HTMLInputElement;
    Object.defineProperty(input, "files", {
      configurable: true,
      value: [new File(["{}"], "config.json", { type: "application/json" })],
    });
    await act(async () => input.dispatchEvent(new Event("change", { bubbles: true })));

    profileState.profile = "two";
    await act(async () => root.render(<ConfigPage />));
    FakeFileReader.latest.result = '{"value":"stale import"}';
    await act(async () => FakeFileReader.latest.onload?.());

    expect(container.querySelector('input[value="B"]')).not.toBeNull();
    expect(container.querySelector('input[value="stale import"]')).toBeNull();
    expect(toastMocks.showToast).not.toHaveBeenCalled();
  });
});
