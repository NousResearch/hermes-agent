// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { MemoryRouter } from "react-router";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { I18nProvider } from "@/i18n";
import type {
  MemoryProviderConfig,
  MemoryProviderField,
  PluginsHubResponse,
} from "@/lib/api";
import PluginsPage from "./PluginsPage";

const apiMocks = vi.hoisted(() => ({
  getMemoryProviderConfig: vi.fn(),
  getPluginsHub: vi.fn(),
  setupMemoryProvider: vi.fn(),
  updateMemoryProviderConfig: vi.fn(),
}));
const uiMocks = vi.hoisted(() => ({
  setAfterTitle: vi.fn(),
  showToast: vi.fn(),
}));

vi.mock("@/lib/api", () => ({ api: apiMocks }));
vi.mock("@/plugins", () => ({ PluginSlot: () => null }));
vi.mock("@/contexts/usePageHeader", () => ({
  usePageHeader: () => ({ setAfterTitle: uiMocks.setAfterTitle }),
}));
vi.mock("@nous-research/ui/hooks/use-toast", () => ({
  useToast: () => ({ showToast: uiMocks.showToast, toast: null }),
}));

const hub: PluginsHubResponse = {
  orphan_dashboard_plugins: [],
  plugins: [],
  providers: {
    context_engine: "compressor",
    context_options: [],
    memory_provider: "hindsight",
    memory_options: [
      {
        available: true,
        configured: true,
        description: "Hindsight memory",
        name: "hindsight",
        status: "ready",
      },
    ],
  },
};

function field(
  key: string,
  kind: MemoryProviderField["kind"],
  value: MemoryProviderField["value"],
  when: MemoryProviderField["when"] = null,
): MemoryProviderField {
  return {
    description: key,
    is_set: false,
    key,
    kind,
    label: key,
    options:
      key === "mode"
        ? [
            { label: "cloud", value: "cloud" },
            { label: "local_external", value: "local_external" },
          ]
        : [],
    placeholder: "",
    required: false,
    url: "",
    value,
    when,
  };
}

const config: MemoryProviderConfig = {
  fields: [
    field("mode", "select", "local_external"),
    field("api_url", "text", "https://api.hindsight.vectorize.io", { mode: "cloud" }),
    field("api_key", "secret", "", { mode: "cloud" }),
    field("api_url", "text", "http://localhost:8888", { mode: "local_external" }),
    field("api_key", "secret", "", { mode: "local_external" }),
    field("bank_id", "text", "hermes"),
    field("bank_mission", "text", ""),
  ],
  label: "Hindsight",
  name: "hindsight",
};

let container: HTMLDivElement;
let root: Root;

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

async function flushEffects() {
  await act(async () => {
    await Promise.resolve();
    await Promise.resolve();
  });
}

async function renderPage() {
  container = document.createElement("div");
  document.body.append(container);
  root = createRoot(container);
  await act(async () => {
    root.render(
      <MemoryRouter>
        <I18nProvider>
          <PluginsPage />
        </I18nProvider>
      </MemoryRouter>,
    );
  });
  await flushEffects();
}

async function setInput(id: string, value: string) {
  const input = container.querySelector(`#${id}`) as HTMLInputElement | null;
  expect(input).not.toBeNull();
  const setValue = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value")?.set;
  await act(async () => {
    setValue?.call(input, value);
    input?.dispatchEvent(new Event("input", { bubbles: true }));
  });
}

async function selectOption(id: string, label: string) {
  const select = container.querySelector(`#${id}`);
  const trigger = select?.querySelector('[role="combobox"]') as HTMLButtonElement | null;
  expect(trigger).not.toBeNull();
  await act(async () => trigger?.click());

  const option = Array.from(select?.querySelectorAll('[role="option"]') ?? []).find(
    (candidate) => candidate.textContent?.trim() === label,
  ) as HTMLElement | undefined;
  expect(option).toBeDefined();
  await act(async () => option?.click());
}

beforeEach(() => {
  apiMocks.getPluginsHub.mockReset();
  apiMocks.getPluginsHub.mockResolvedValue(hub);
  apiMocks.getMemoryProviderConfig.mockReset();
  apiMocks.getMemoryProviderConfig.mockResolvedValue(config);
  apiMocks.setupMemoryProvider.mockReset();
  apiMocks.setupMemoryProvider.mockResolvedValue({ ok: true, provider: "hindsight", results: [] });
  apiMocks.updateMemoryProviderConfig.mockReset();
  apiMocks.updateMemoryProviderConfig.mockResolvedValue(undefined);
  uiMocks.setAfterTitle.mockReset();
  uiMocks.showToast.mockReset();
});

afterEach(async () => {
  await act(async () => root?.unmount());
  container?.remove();
});

describe("PluginsPage memory provider config", () => {
  it("keeps duplicate-key drafts scoped when switching provider modes", async () => {
    await renderPage();

    await setInput("memory-api_url", "http://hindsight.internal:8888");
    await setInput("memory-api_key", "local-secret");
    await selectOption("memory-mode", "cloud");

    expect((container.querySelector("#memory-api_url") as HTMLInputElement | null)?.value).toBe(
      "https://api.hindsight.vectorize.io",
    );
    expect((container.querySelector("#memory-api_key") as HTMLInputElement | null)?.value).toBe(
      "",
    );

    const saveButton = Array.from(container.querySelectorAll("button")).find(
      (button) => button.textContent?.trim() === "Save memory provider",
    );
    expect(saveButton).toBeDefined();
    await act(async () => saveButton?.click());
    await flushEffects();
    expect(apiMocks.updateMemoryProviderConfig).toHaveBeenCalledWith("hindsight", {
      api_key: "",
      api_url: "https://api.hindsight.vectorize.io",
      bank_id: "hermes",
      bank_mission: "",
      mode: "cloud",
    });

    await selectOption("memory-mode", "local_external");

    expect((container.querySelector("#memory-api_url") as HTMLInputElement | null)?.value).toBe(
      "http://hindsight.internal:8888",
    );
    expect((container.querySelector("#memory-api_key") as HTMLInputElement | null)?.value).toBe(
      "local-secret",
    );
  });

  it("initializes duplicate-key fields from the active provider mode", async () => {
    apiMocks.getMemoryProviderConfig.mockResolvedValue({
      ...config,
      fields: config.fields.map((candidate) =>
        candidate.key === "mode" ? { ...candidate, value: "cloud" } : candidate,
      ),
    });

    await renderPage();

    const apiUrl = container.querySelector("#memory-api_url") as HTMLInputElement | null;
    expect(apiUrl?.value).toBe("https://api.hindsight.vectorize.io");
  });

  it("keeps an already-set active secret write-only when saving", async () => {
    apiMocks.getMemoryProviderConfig.mockResolvedValue({
      ...config,
      fields: config.fields.map((candidate) =>
        candidate.key === "api_key" && candidate.when?.mode === "local_external"
          ? { ...candidate, is_set: true, value: "" }
          : candidate,
      ),
    });

    await renderPage();

    const apiKey = container.querySelector("#memory-api_key") as HTMLInputElement | null;
    expect(apiKey?.value).toBe("");
    expect(apiKey?.placeholder).toBe("Leave blank to keep existing value");

    const saveButton = Array.from(container.querySelectorAll("button")).find(
      (button) => button.textContent?.trim() === "Save memory provider",
    );
    expect(saveButton).toBeDefined();
    await act(async () => saveButton?.click());
    await flushEffects();

    expect(apiMocks.updateMemoryProviderConfig).toHaveBeenCalledWith("hindsight", {
      api_key: "",
      api_url: "http://localhost:8888",
      bank_id: "hermes",
      bank_mission: "",
      mode: "local_external",
    });
  });

  it("saves duplicate-key fields from the active provider mode", async () => {
    await renderPage();

    await setInput("memory-api_url", "http://hindsight.internal:8888");
    await setInput("memory-api_key", "local-secret");
    await setInput("memory-bank_id", "security-team");
    await setInput("memory-bank_mission", "Retain security decisions");

    const saveButton = Array.from(container.querySelectorAll("button")).find(
      (button) => button.textContent?.trim() === "Save memory provider",
    );
    expect(saveButton).toBeDefined();
    await act(async () => saveButton?.click());
    await flushEffects();

    expect(apiMocks.updateMemoryProviderConfig).toHaveBeenCalledWith("hindsight", {
      api_key: "local-secret",
      api_url: "http://hindsight.internal:8888",
      bank_id: "security-team",
      bank_mission: "Retain security decisions",
      mode: "local_external",
    });
  });

  it("passes duplicate-key fields from the active mode to provider setup", async () => {
    apiMocks.getPluginsHub.mockResolvedValue({
      ...hub,
      providers: {
        ...hub.providers,
        memory_options: [
          {
            ...hub.providers.memory_options[0],
            available: false,
            configured: false,
            setup: {
              dependencies_installed: false,
              external_dependencies: [],
              pip_dependencies: ["hindsight-embed"],
              required_env: [],
            },
            status: "unavailable",
          },
        ],
      },
    });
    await renderPage();

    await setInput("memory-api_url", "http://hindsight.internal:8888");
    await setInput("memory-api_key", "local-secret");
    await setInput("memory-bank_id", "security-team");
    await setInput("memory-bank_mission", "Retain security decisions");

    const setupButton = Array.from(container.querySelectorAll("button")).find(
      (button) => button.textContent?.trim() === "Install provider dependencies",
    );
    expect(setupButton).toBeDefined();
    await act(async () => setupButton?.click());
    await flushEffects();

    expect(apiMocks.setupMemoryProvider).toHaveBeenCalledWith("hindsight", {
      api_key: "local-secret",
      api_url: "http://hindsight.internal:8888",
      bank_id: "security-team",
      bank_mission: "Retain security decisions",
      mode: "local_external",
    });
  });
});
