// @vitest-environment jsdom

import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { en } from "@/i18n/en";
import { EFFORT_OPTIONS } from "@/lib/reasoning-effort";
import { api } from "@/lib/api";
import ConfigPage from "./ConfigPage";
import ProfilesPage from "./ProfilesPage";

vi.mock("@/lib/api", () => ({
  api: {
    getConfig: vi.fn(),
    getSchema: vi.fn(),
    getDefaults: vi.fn(),
    getConfigRaw: vi.fn(),
    getStatus: vi.fn(),
    saveConfig: vi.fn(),
    getProfiles: vi.fn(),
    getActiveProfile: vi.fn(),
    getModelOptions: vi.fn(),
    setProfileModel: vi.fn(),
    setProfileSettings: vi.fn(),
    getProfileFallbacks: vi.fn(),
    updateProfileFallbacks: vi.fn(),
    getProfileSoul: vi.fn(),
    updateProfileSoul: vi.fn(),
    updateProfileDescription: vi.fn(),
    describeProfileAuto: vi.fn(),
    createProfile: vi.fn(),
    setActiveProfile: vi.fn(),
    getProfileSetupCommand: vi.fn(),
    renameProfile: vi.fn(),
    deleteProfile: vi.fn(),
    getSkills: vi.fn(),
  },
}));

vi.mock("@/i18n", () => ({
  useI18n: () => ({ t: en }),
}));
vi.mock("@/contexts/usePageHeader", () => ({
  usePageHeader: () => ({ setEnd: vi.fn() }),
}));
vi.mock("@/contexts/useProfileScope", () => ({
  useProfileScope: () => ({ setProfile: vi.fn() }),
}));
vi.mock("@/hooks/useModalBehavior", () => ({
  useModalBehavior: () => undefined,
}));
vi.mock("@/plugins", () => ({
  PluginSlot: () => null,
}));
vi.mock("@nous-research/ui/hooks/use-toast", () => ({
  useToast: () => ({ toast: null, showToast: vi.fn() }),
}));
vi.mock("@nous-research/ui/hooks/use-confirm-delete", () => ({
  useConfirmDelete: () => ({
    confirmDelete: vi.fn(),
    dialogProps: {},
    isDeleting: false,
  }),
}));
vi.mock("react-router-dom", () => ({
  useNavigate: () => vi.fn(),
}));

const mockedApi = vi.mocked(api);

let container: HTMLDivElement;
let root: Root;

async function renderPage(element: React.ReactElement) {
  container = document.createElement("div");
  document.body.appendChild(container);
  root = createRoot(container);
  await act(async () => {
    root.render(element);
    await Promise.resolve();
    await Promise.resolve();
  });
}

async function settle() {
  await act(async () => {
    await Promise.resolve();
    await Promise.resolve();
  });
}

function openSelect(select: HTMLButtonElement) {
  act(() => select.click());
}

function openOptions(select: HTMLButtonElement) {
  const alreadyOpen = [...container.querySelectorAll<HTMLElement>("[role=option]")];
  if (alreadyOpen.length > 0) return alreadyOpen;
  openSelect(select);
  return [...container.querySelectorAll<HTMLElement>("[role=option]")];
}

async function chooseOption(select: HTMLButtonElement, label: string) {
  const option = openOptions(select).find(
    (candidate) => candidate.textContent?.trim() === label,
  );
  expect(option).toBeDefined();
  await act(async () => {
    option!.click();
    await Promise.resolve();
  });
}

afterEach(() => {
  act(() => root?.unmount());
  container?.remove();
  vi.clearAllMocks();
});

beforeEach(() => {
  vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
  vi.stubGlobal("matchMedia", () => ({
    matches: false,
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
  }));
  Object.defineProperty(window, "matchMedia", {
    configurable: true,
    value: () => ({
      matches: false,
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
    }),
  });
});

describe("ConfigPage reasoning effort selector", () => {
  beforeEach(() => {
    mockedApi.getConfig.mockResolvedValue({ agent: {} });
    mockedApi.getSchema.mockResolvedValue({
      fields: {
        "agent.reasoning_effort": {
          category: "agent",
          type: "select",
          options: ["", ...EFFORT_OPTIONS.map(({ value }) => value)],
          emptyLabel: "Inherit provider default",
        },
      },
      category_order: ["agent"],
    });
    mockedApi.getDefaults.mockResolvedValue({ agent: {} });
    mockedApi.getConfigRaw.mockResolvedValue({ yaml: "agent: {}\n", path: "/tmp/config.yaml" });
    mockedApi.getStatus.mockResolvedValue(
      { config_path: "/tmp/config.yaml" } as unknown as Awaited<
        ReturnType<typeof api.getStatus>
      >,
    );
    mockedApi.saveConfig.mockResolvedValue({ ok: true });
  });

  it("renders every allowed value and the backward-compatible inherit option", async () => {
    await renderPage(<ConfigPage />);

    const select = container.querySelector<HTMLButtonElement>("button[role=combobox]");
    expect(select).not.toBeNull();
    expect(select?.textContent).toContain("Inherit provider default");

    const options = openOptions(select!);
    expect(options.map((option) => option.textContent?.trim())).toEqual([
      "Inherit provider default",
      ...EFFORT_OPTIONS.map(({ label }) => label),
    ]);
  });

  it("updates the displayed value and propagates a changed selection through the ConfigPage save API", async () => {
    await renderPage(<ConfigPage />);

    const select = container.querySelector<HTMLButtonElement>("button[role=combobox]");
    await chooseOption(select!, "High");
    expect(select?.textContent).toContain("High");
    const save = [...container.querySelectorAll<HTMLButtonElement>("button")].find(
      (button) => button.textContent?.trim() === "Save",
    );
    expect(save).toBeDefined();

    await act(async () => {
      save!.click();
      await Promise.resolve();
    });
    expect(mockedApi.saveConfig).toHaveBeenCalledWith({
      agent: { reasoning_effort: "high" },
    });
  });

  it("restores the saved global effort after the page is remounted", async () => {
    let persistedConfig: Record<string, unknown> = { agent: {} };
    mockedApi.getConfig.mockImplementation(async () => structuredClone(persistedConfig));
    mockedApi.saveConfig.mockImplementation(async (nextConfig) => {
      persistedConfig = structuredClone(nextConfig);
      return { ok: true };
    });

    await renderPage(<ConfigPage />);
    let select = container.querySelector<HTMLButtonElement>("button[role=combobox]");
    openSelect(select!);
    const high = [...container.querySelectorAll<HTMLElement>("[role=option]")].find(
      (option) => option.textContent?.trim() === "High",
    );
    expect(high).toBeDefined();
    await act(async () => {
      high!.click();
      await Promise.resolve();
    });

    const save = [...container.querySelectorAll<HTMLButtonElement>("button")].find(
      (button) => button.textContent?.trim() === "Save",
    );
    expect(save).toBeDefined();
    await act(async () => {
      save!.click();
      await Promise.resolve();
    });
    await settle();
    expect(persistedConfig).toEqual({ agent: { reasoning_effort: "high" } });

    act(() => root.unmount());
    container.remove();
    await renderPage(<ConfigPage />);
    select = container.querySelector<HTMLButtonElement>("button[role=combobox]");
    expect(select?.textContent).toContain("High");
  });

  it("returns an existing selection to inherit and saves the cleared value", async () => {
    mockedApi.getConfig.mockResolvedValue({ agent: { reasoning_effort: "high" } });
    await renderPage(<ConfigPage />);

    const select = container.querySelector<HTMLButtonElement>("button[role=combobox]");
    expect(select?.textContent).toContain("High");
    await chooseOption(select!, "Inherit provider default");
    expect(select?.textContent).toContain("Inherit provider default");

    const save = [...container.querySelectorAll<HTMLButtonElement>("button")].find(
      (button) => button.textContent?.trim() === "Save",
    );
    expect(save).toBeDefined();
    await act(async () => {
      save!.click();
      await Promise.resolve();
    });
    expect(mockedApi.saveConfig).toHaveBeenCalledWith({
      agent: { reasoning_effort: "" },
    });
  });

  it.each([
    {
      name: "no selectable effort levels",
      options: [""],
      labels: ["Inherit provider default"],
    },
    {
      name: "one selectable effort level",
      options: ["", "high"],
      labels: ["Inherit provider default", "High"],
    },
    {
      name: "the maximum supported effort levels",
      options: ["", ...EFFORT_OPTIONS.map(({ value }) => value)],
      labels: ["Inherit provider default", ...EFFORT_OPTIONS.map(({ label }) => label)],
    },
  ])(
    "renders the $name option set without adding unsupported values",
    async ({ options, labels }) => {
      mockedApi.getSchema.mockResolvedValue({
        fields: {
          "agent.reasoning_effort": {
            category: "agent",
            type: "select",
            options,
            emptyLabel: "Inherit provider default",
          },
        },
        category_order: ["agent"],
      });
      await renderPage(<ConfigPage />);

      const select = container.querySelector<HTMLButtonElement>("button[role=combobox]");
      expect(select).not.toBeNull();
      expect(openOptions(select!).map((option) => option.textContent?.trim())).toEqual(labels);
    },
  );

  it("does not expose an invalid persisted value as a selectable option", async () => {
    mockedApi.getConfig.mockResolvedValue({ agent: { reasoning_effort: "turbo" } });
    await renderPage(<ConfigPage />);

    const select = container.querySelector<HTMLButtonElement>("button[role=combobox]");
    expect(select?.textContent).toContain("turbo");
    const options = openOptions(select!);
    expect(options.map((option) => option.textContent?.trim())).toEqual([
      "Inherit provider default",
      ...EFFORT_OPTIONS.map(({ label }) => label),
    ]);
    expect(options.some((option) => option.textContent?.trim() === "turbo")).toBe(false);

    await chooseOption(select!, "Medium");
    expect(select?.textContent).toContain("Medium");
  });

  it("keeps only the latest value when the selector is changed rapidly", async () => {
    await renderPage(<ConfigPage />);

    const select = container.querySelector<HTMLButtonElement>("button[role=combobox]");
    await chooseOption(select!, "High");
    await chooseOption(select!, "Low");
    expect(select?.textContent).toContain("Low");

    const save = [...container.querySelectorAll<HTMLButtonElement>("button")].find(
      (button) => button.textContent?.trim() === "Save",
    );
    expect(save).toBeDefined();
    await act(async () => {
      save!.click();
      await Promise.resolve();
    });
    expect(mockedApi.saveConfig).toHaveBeenCalledTimes(1);
    expect(mockedApi.saveConfig).toHaveBeenCalledWith({
      agent: { reasoning_effort: "low" },
    });
  });

  it("does not persist an unsaved selection after leaving and returning to the page", async () => {
    await renderPage(<ConfigPage />);

    const select = container.querySelector<HTMLButtonElement>("button[role=combobox]");
    await chooseOption(select!, "High");
    expect(select?.textContent).toContain("High");
    expect(mockedApi.saveConfig).not.toHaveBeenCalled();

    act(() => root.unmount());
    container.remove();
    await renderPage(<ConfigPage />);

    const remountedSelect = container.querySelector<HTMLButtonElement>(
      "button[role=combobox]",
    );
    expect(remountedSelect?.textContent).toContain("Inherit provider default");
    expect(mockedApi.saveConfig).not.toHaveBeenCalled();
  });

  it("resets the virtual reasoning field to inherit before saving the scoped category", async () => {
    mockedApi.getConfig.mockResolvedValue({ agent: { reasoning_effort: "high" } });

    await renderPage(<ConfigPage />);
    const select = container.querySelector<HTMLButtonElement>("button[role=combobox]");
    expect(select?.textContent).toContain("High");

    const reset = container.querySelector<HTMLButtonElement>(
      "button[aria-label='Reset Agent to defaults']",
    );
    expect(reset).not.toBeNull();
    await act(async () => {
      reset!.click();
      await Promise.resolve();
    });

    const confirm = [...document.body.querySelectorAll<HTMLButtonElement>("button")].find(
      (button) => button.textContent?.trim() === "Reset to defaults",
    );
    expect(confirm).toBeDefined();
    await act(async () => {
      confirm!.click();
      await Promise.resolve();
    });
    expect(select?.textContent).toContain("Inherit provider default");

    const save = [...container.querySelectorAll<HTMLButtonElement>("button")].find(
      (button) => button.textContent?.trim() === "Save",
    );
    expect(save).toBeDefined();
    await act(async () => {
      save!.click();
      await Promise.resolve();
    });
    expect(mockedApi.saveConfig).toHaveBeenCalledWith({
      agent: { reasoning_effort: "" },
    });
  });
});

describe("ProfilesPage reasoning effort selector", () => {
  beforeEach(() => {
    mockedApi.getProfiles.mockResolvedValue({
      profiles: [
        {
          name: "default",
          path: "/tmp/default",
          is_default: true,
          model: "model-a",
          provider: "provider-a",
          reasoning_effort: "",
          has_env: false,
          skill_count: 0,
          gateway_running: false,
          description: "",
          description_auto: false,
          distribution_name: null,
          distribution_version: null,
          distribution_source: null,
          has_alias: false,
        },
      ],
    });
    mockedApi.getActiveProfile.mockResolvedValue({ active: "default", current: "default" });
    mockedApi.getModelOptions.mockResolvedValue({
      providers: [{ slug: "provider-a", name: "Provider A", models: ["model-a"] }],
    });
    mockedApi.setProfileSettings.mockResolvedValue({
      ok: true,
      provider: null,
      model: null,
      reasoning_effort: "high",
    });
    mockedApi.setProfileModel.mockResolvedValue({ ok: true, provider: "provider-a", model: "model-a" });
    mockedApi.getProfileFallbacks.mockResolvedValue({ fallbacks: [] });
    mockedApi.updateProfileFallbacks.mockResolvedValue({ ok: true, fallbacks: [] });
  });

  it("renders the profile selector with inherit plus all allowed values", async () => {
    await renderPage(<ProfilesPage />);

    const actions = container.querySelector<HTMLButtonElement>("button[aria-label='Actions']");
    expect(actions).not.toBeNull();
    await act(async () => {
      actions!.click();
      await Promise.resolve();
    });
    const changeModel = [...container.querySelectorAll<HTMLButtonElement>("[role=menuitem]")].find(
      (button) => button.textContent?.includes("Change model"),
    );
    expect(changeModel).toBeDefined();
    await act(async () => {
      changeModel!.click();
      await Promise.resolve();
    });
    await settle();

    const selects = [...container.querySelectorAll<HTMLButtonElement>("button[role=combobox]")];
    expect(selects).toHaveLength(2);
    expect(selects[1].textContent).toContain("Inherit provider default");
    expect(openOptions(selects[1]).map((option) => option.textContent?.trim())).toEqual([
      "Inherit provider default",
      ...EFFORT_OPTIONS.map(({ label }) => label),
    ]);
  });

  it("loads model options for the profile being edited", async () => {
    mockedApi.getProfiles.mockResolvedValue({
      profiles: [
        {
          name: "default",
          path: "/tmp/default",
          is_default: true,
          model: "model-a",
          provider: "provider-a",
          reasoning_effort: "",
          has_env: false,
          skill_count: 0,
          gateway_running: false,
          description: "",
          description_auto: false,
          distribution_name: null,
          distribution_version: null,
          distribution_source: null,
          has_alias: false,
        },
        {
          name: "other",
          path: "/tmp/other",
          is_default: false,
          model: "other-model",
          provider: "other-provider",
          reasoning_effort: "",
          has_env: false,
          skill_count: 0,
          gateway_running: false,
          description: "",
          description_auto: false,
          distribution_name: null,
          distribution_version: null,
          distribution_source: null,
          has_alias: false,
        },
      ],
    });
    mockedApi.getModelOptions.mockImplementation(async (profile) => ({
      providers: [
        profile === "other"
          ? { slug: "other-provider", name: "Other Provider", models: ["other-model"] }
          : { slug: "provider-a", name: "Provider A", models: ["model-a"] },
      ],
    }));

    await renderPage(<ProfilesPage />);
    const menus = [...container.querySelectorAll<HTMLElement>("[data-profile-actions]")];
    expect(menus).toHaveLength(2);
    const actions = menus[1].querySelector<HTMLButtonElement>(
      "button[aria-label='Actions']",
    );
    expect(actions).not.toBeNull();
    await act(async () => actions!.click());
    const changeModel = [...menus[1].querySelectorAll<HTMLButtonElement>("[role=menuitem]")].find(
      (button) => button.textContent?.includes("Change model"),
    );
    expect(changeModel).toBeDefined();
    await act(async () => changeModel!.click());
    await settle();

    expect(mockedApi.getModelOptions).toHaveBeenCalledWith("other");
    const modelSelect = container.querySelector<HTMLButtonElement>("button[role=combobox]");
    expect(modelSelect).not.toBeNull();
    expect(openOptions(modelSelect!).map((option) => option.textContent?.trim())).toContain(
      "Other Provider · other-model",
    );
  });

  it("sends a changed model and reasoning effort in one settings update", async () => {
    mockedApi.getModelOptions.mockResolvedValue({
      providers: [
        { slug: "provider-a", name: "Provider A", models: ["model-a", "model-b"] },
      ],
    });
    mockedApi.setProfileSettings.mockResolvedValue({
      ok: true,
      provider: "provider-a",
      model: "model-b",
      reasoning_effort: "high",
    });

    await renderPage(<ProfilesPage />);
    const actions = container.querySelector<HTMLButtonElement>("button[aria-label='Actions']");
    await act(async () => actions!.click());
    const changeModel = [...container.querySelectorAll<HTMLButtonElement>("[role=menuitem]")].find(
      (button) => button.textContent?.includes("Change model"),
    );
    await act(async () => changeModel!.click());
    await settle();

    const selects = [...container.querySelectorAll<HTMLButtonElement>("button[role=combobox]")];
    await chooseOption(selects[0], "Provider A · model-b");
    await chooseOption(selects[1], "High");
    const save = [...container.querySelectorAll<HTMLButtonElement>("button")].find(
      (button) => button.textContent?.trim() === "Save",
    );
    expect(save).toBeDefined();
    await act(async () => {
      save!.click();
      await Promise.resolve();
    });
    await settle();

    expect(mockedApi.setProfileModel).not.toHaveBeenCalled();
    expect(mockedApi.setProfileSettings).toHaveBeenCalledWith(
      "default",
      "provider-a",
      "model-b",
      "high",
    );
  });

  it("ignores a late model save after switching to another profile editor", async () => {
    mockedApi.getProfiles.mockResolvedValue({
      profiles: [
        {
          name: "default",
          path: "/tmp/default",
          is_default: true,
          model: "model-a",
          provider: "provider-a",
          reasoning_effort: "",
          has_env: false,
          skill_count: 0,
          gateway_running: false,
          description: "",
          description_auto: false,
          distribution_name: null,
          distribution_version: null,
          distribution_source: null,
          has_alias: false,
        },
        {
          name: "other",
          path: "/tmp/other",
          is_default: false,
          model: "model-b",
          provider: "provider-b",
          reasoning_effort: "",
          has_env: false,
          skill_count: 0,
          gateway_running: false,
          description: "",
          description_auto: false,
          distribution_name: null,
          distribution_version: null,
          distribution_source: null,
          has_alias: false,
        },
      ],
    });
    mockedApi.getModelOptions.mockImplementation(async (profile) => ({
      providers: [
        profile === "other"
          ? { slug: "provider-b", name: "Provider B", models: ["model-b"] }
          : { slug: "provider-a", name: "Provider A", models: ["model-a"] },
      ],
    }));

    type ProfileSettingsResponse = Awaited<ReturnType<typeof api.setProfileSettings>>;
    let resolveFirstSave!: (value: ProfileSettingsResponse) => void;
    const firstSave = new Promise<ProfileSettingsResponse>((resolve) => {
      resolveFirstSave = resolve;
    });
    mockedApi.setProfileSettings.mockImplementationOnce(() => firstSave);

    await renderPage(<ProfilesPage />);
    let menus = [...container.querySelectorAll<HTMLElement>("[data-profile-actions]")];
    await act(async () => menus[0].querySelector<HTMLButtonElement>("button[aria-label='Actions']")!.click());
    const firstChangeModel = [
      ...menus[0].querySelectorAll<HTMLButtonElement>("[role=menuitem]"),
    ].find((button) => button.textContent?.includes("Change model"));
    expect(firstChangeModel).toBeDefined();
    await act(async () => firstChangeModel!.click());
    await settle();

    const selects = [...container.querySelectorAll<HTMLButtonElement>("button[role=combobox]")];
    await chooseOption(selects[1], "High");
    const save = [...container.querySelectorAll<HTMLButtonElement>("button")].find(
      (button) => button.textContent?.trim() === "Save",
    );
    expect(save).toBeDefined();
    await act(async () => {
      save!.click();
      await Promise.resolve();
    });

    await act(async () => container.querySelector<HTMLButtonElement>("button[aria-label='Close']")!.click());
    menus = [...container.querySelectorAll<HTMLElement>("[data-profile-actions]")];
    await act(async () => menus[1].querySelector<HTMLButtonElement>("button[aria-label='Actions']")!.click());
    const secondChangeModel = [
      ...menus[1].querySelectorAll<HTMLButtonElement>("[role=menuitem]"),
    ].find((button) => button.textContent?.includes("Change model"));
    expect(secondChangeModel).toBeDefined();
    await act(async () => secondChangeModel!.click());
    await settle();

    expect(container.querySelector<HTMLElement>("#profile-editor-title")?.textContent).toContain(
      "other",
    );

    await act(async () => {
      resolveFirstSave({
        ok: true,
        provider: null,
        model: null,
        reasoning_effort: "high",
      });
      await Promise.resolve();
    });
    await settle();

    expect(container.querySelector<HTMLElement>("#profile-editor-title")?.textContent).toContain(
      "other",
    );
  });

  it("keeps reasoning editing available when no model choices are configured", async () => {
    mockedApi.getProfiles.mockResolvedValue({
      profiles: [
        {
          name: "default",
          path: "/tmp/default",
          is_default: true,
          model: null,
          provider: null,
          reasoning_effort: "",
          has_env: false,
          skill_count: 0,
          gateway_running: false,
          description: "",
          description_auto: false,
          distribution_name: null,
          distribution_version: null,
          distribution_source: null,
          has_alias: false,
        },
      ],
    });
    mockedApi.getModelOptions.mockResolvedValue({ providers: [] });
    mockedApi.setProfileSettings.mockImplementation(async (_name, _provider, _model, effort) => ({
      ok: true,
      provider: null,
      model: null,
      reasoning_effort: effort,
    }));

    await renderPage(<ProfilesPage />);
    const actions = container.querySelector<HTMLButtonElement>("button[aria-label='Actions']");
    await act(async () => actions!.click());
    const changeModel = [...container.querySelectorAll<HTMLButtonElement>("[role=menuitem]")].find(
      (button) => button.textContent?.includes("Change model"),
    );
    expect(changeModel).toBeDefined();
    await act(async () => changeModel!.click());
    await settle();

    expect(container.textContent).toContain("No authenticated providers");
    const selects = [...container.querySelectorAll<HTMLButtonElement>("button[role=combobox]")];
    expect(selects).toHaveLength(1);
    expect(selects[0].textContent).toContain("Inherit provider default");

    await chooseOption(selects[0], "High");
    const save = [...container.querySelectorAll<HTMLButtonElement>("button")].find(
      (button) => button.textContent?.trim() === "Save",
    );
    expect(save).toBeDefined();
    await act(async () => {
      save!.click();
      await Promise.resolve();
    });
    await settle();
    expect(mockedApi.setProfileModel).not.toHaveBeenCalled();
    expect(mockedApi.setProfileSettings).toHaveBeenCalledWith("default", null, null, "high");
  });

  it("saves reasoning for an existing model when model choices are unavailable", async () => {
    mockedApi.getModelOptions.mockResolvedValue({ providers: [] });
    mockedApi.setProfileSettings.mockImplementation(async (_name, _provider, _model, effort) => ({
      ok: true,
      provider: null,
      model: null,
      reasoning_effort: effort,
    }));

    await renderPage(<ProfilesPage />);
    const actions = container.querySelector<HTMLButtonElement>("button[aria-label='Actions']");
    await act(async () => actions!.click());
    const changeModel = [
      ...container.querySelectorAll<HTMLButtonElement>("[role=menuitem]"),
    ].find((button) => button.textContent?.includes("Change model"));
    expect(changeModel).toBeDefined();
    await act(async () => changeModel!.click());
    await settle();

    expect(container.textContent).toContain("No authenticated providers");
    const selects = [...container.querySelectorAll<HTMLButtonElement>("button[role=combobox]")];
    expect(selects).toHaveLength(1);
    await chooseOption(selects[0], "High");

    const save = [...container.querySelectorAll<HTMLButtonElement>("button")].find(
      (button) => button.textContent?.trim() === "Save",
    );
    expect(save).toBeDefined();
    expect(save?.disabled).toBe(false);
    await act(async () => {
      save!.click();
      await Promise.resolve();
    });
    await settle();

    expect(mockedApi.setProfileModel).not.toHaveBeenCalled();
    expect(mockedApi.setProfileSettings).toHaveBeenCalledWith("default", null, null, "high");
  });

  it.each([
    { initialEffort: "", selectedEffort: "high", label: "sets a concrete effort" },
    { initialEffort: "high", selectedEffort: "", label: "clears the effort" },
    { initialEffort: "high", selectedEffort: "none", label: "disables thinking" },
  ])(
    "allows reasoning-only saves for an unlisted stored model when it $label",
    async ({ initialEffort, selectedEffort }) => {
      mockedApi.getProfiles.mockResolvedValue({
        profiles: [
          {
            name: "default",
            path: "/tmp/default",
            is_default: true,
            model: "provider-model",
            provider: "custom",
            reasoning_effort: initialEffort,
            has_env: false,
            skill_count: 0,
            gateway_running: false,
            description: "",
            description_auto: false,
            distribution_name: null,
            distribution_version: null,
            distribution_source: null,
            has_alias: false,
          },
        ],
      });
      mockedApi.getModelOptions.mockResolvedValue({
        providers: [{ slug: "provider-a", name: "Provider A", models: ["model-a"] }],
      });
      mockedApi.setProfileSettings.mockImplementation(async (_name, _provider, _model, effort) => ({
        ok: true,
        provider: null,
        model: null,
        reasoning_effort: effort,
      }));

      await renderPage(<ProfilesPage />);
      const actions = container.querySelector<HTMLButtonElement>(
        "button[aria-label='Actions']",
      );
      await act(async () => actions!.click());
      const changeModel = [
        ...container.querySelectorAll<HTMLButtonElement>("[role=menuitem]"),
      ].find((button) => button.textContent?.includes("Change model"));
      expect(changeModel).toBeDefined();
      await act(async () => changeModel!.click());
      await settle();

      const selects = [...container.querySelectorAll<HTMLButtonElement>("button[role=combobox]")];
      expect(selects).toHaveLength(2);
      expect(selects[0].textContent).toContain("custom · provider-model");
      await chooseOption(
        selects[1],
        selectedEffort === "high"
          ? "High"
          : selectedEffort === "none"
            ? "Off (no thinking)"
            : "Inherit provider default",
      );
      const save = [...container.querySelectorAll<HTMLButtonElement>("button")].find(
        (button) => button.textContent?.trim() === "Save",
      );
      expect(save).toBeDefined();
      expect(save?.disabled).toBe(false);
      await act(async () => {
        save!.click();
        await Promise.resolve();
      });
      await settle();

      expect(mockedApi.setProfileModel).not.toHaveBeenCalled();
      expect(mockedApi.setProfileSettings).toHaveBeenCalledWith(
        "default",
        null,
        null,
        selectedEffort,
      );
    },
  );

  it("clears a profile reasoning override without model choices", async () => {
    mockedApi.getProfiles.mockResolvedValue({
      profiles: [
        {
          name: "default",
          path: "/tmp/default",
          is_default: true,
          model: null,
          provider: null,
          reasoning_effort: "high",
          has_env: false,
          skill_count: 0,
          gateway_running: false,
          description: "",
          description_auto: false,
          distribution_name: null,
          distribution_version: null,
          distribution_source: null,
          has_alias: false,
        },
      ],
    });
    mockedApi.getModelOptions.mockResolvedValue({ providers: [] });
    mockedApi.setProfileSettings.mockResolvedValue({
      ok: true,
      provider: null,
      model: null,
      reasoning_effort: "",
    });

    await renderPage(<ProfilesPage />);
    const actions = container.querySelector<HTMLButtonElement>("button[aria-label='Actions']");
    await act(async () => actions!.click());
    const changeModel = [
      ...container.querySelectorAll<HTMLButtonElement>("[role=menuitem]"),
    ].find((button) => button.textContent?.includes("Change model"));
    expect(changeModel).toBeDefined();
    await act(async () => changeModel!.click());
    await settle();

    const select = container.querySelector<HTMLButtonElement>(
      "button[role=combobox]",
    );
    expect(select?.textContent).toContain("High");
    await chooseOption(select!, "Inherit provider default");
    const save = [...container.querySelectorAll<HTMLButtonElement>("button")].find(
      (button) => button.textContent?.trim() === "Save",
    );
    expect(save).toBeDefined();
    await act(async () => {
      save!.click();
      await Promise.resolve();
    });
    await settle();
    expect(mockedApi.setProfileSettings).toHaveBeenCalledWith("default", null, null, "");
  });

  it("persists a profile selection through the profile reasoning endpoint", async () => {
    await renderPage(<ProfilesPage />);
    const actions = container.querySelector<HTMLButtonElement>("button[aria-label='Actions']");
    await act(async () => actions!.click());
    const changeModel = [...container.querySelectorAll<HTMLButtonElement>("[role=menuitem]")].find(
      (button) => button.textContent?.includes("Change model"),
    );
    await act(async () => changeModel!.click());
    await settle();

    const selects = [...container.querySelectorAll<HTMLButtonElement>("button[role=combobox]")];
    await chooseOption(selects[1], "High");
    expect(selects[1].textContent).toContain("High");

    const save = [...container.querySelectorAll<HTMLButtonElement>("button")].find(
      (button) => button.textContent?.trim() === "Save",
    );
    expect(save).toBeDefined();
    await act(async () => {
      save!.click();
      await Promise.resolve();
    });
    await settle();
    expect(mockedApi.setProfileSettings).toHaveBeenCalledWith("default", null, null, "high");
  });

  it("keeps only the latest profile value when the selector is changed rapidly", async () => {
    await renderPage(<ProfilesPage />);
    const actions = container.querySelector<HTMLButtonElement>("button[aria-label='Actions']");
    await act(async () => actions!.click());
    const changeModel = [...container.querySelectorAll<HTMLButtonElement>("[role=menuitem]")].find(
      (button) => button.textContent?.includes("Change model"),
    );
    await act(async () => changeModel!.click());
    await settle();

    const selects = [...container.querySelectorAll<HTMLButtonElement>("button[role=combobox]")];
    await chooseOption(selects[1], "High");
    await chooseOption(selects[1], "Low");
    expect(selects[1].textContent).toContain("Low");

    const save = [...container.querySelectorAll<HTMLButtonElement>("button")].find(
      (button) => button.textContent?.trim() === "Save",
    );
    expect(save).toBeDefined();
    await act(async () => {
      save!.click();
      await Promise.resolve();
    });
    await settle();
    expect(mockedApi.setProfileSettings).toHaveBeenCalledTimes(1);
    expect(mockedApi.setProfileSettings).toHaveBeenCalledWith("default", null, null, "low");
  });

  it("restores the persisted profile effort after the page is remounted", async () => {
    const profileResponse = {
      profiles: [
        {
          name: "default",
          path: "/tmp/default",
          is_default: true,
          model: "model-a",
          provider: "provider-a",
          reasoning_effort: "",
          has_env: false,
          skill_count: 0,
          gateway_running: false,
          description: "",
          description_auto: false,
          distribution_name: null,
          distribution_version: null,
          distribution_source: null,
          has_alias: false,
        },
      ],
    };
    mockedApi.getProfiles.mockResolvedValue(profileResponse);
    mockedApi.setProfileSettings.mockImplementation(async (_name, _provider, _model, effort) => {
      profileResponse.profiles[0].reasoning_effort = effort;
      return { ok: true, provider: null, model: null, reasoning_effort: effort };
    });

    await renderPage(<ProfilesPage />);
    const actions = container.querySelector<HTMLButtonElement>("button[aria-label='Actions']");
    await act(async () => actions!.click());
    const changeModel = [...container.querySelectorAll<HTMLButtonElement>("[role=menuitem]")].find(
      (button) => button.textContent?.includes("Change model"),
    );
    await act(async () => changeModel!.click());
    await settle();

    const selects = [...container.querySelectorAll<HTMLButtonElement>("button[role=combobox]")];
    openSelect(selects[1]);
    const high = [...container.querySelectorAll<HTMLElement>("[role=option]")].find(
      (option) => option.textContent?.trim() === "High",
    );
    expect(high).toBeDefined();
    await act(async () => {
      high!.click();
      await Promise.resolve();
    });
    const save = [...container.querySelectorAll<HTMLButtonElement>("button")].find(
      (button) => button.textContent?.trim() === "Save",
    );
    expect(save).toBeDefined();
    await act(async () => {
      save!.click();
      await Promise.resolve();
    });
    await settle();
    expect(profileResponse.profiles[0].reasoning_effort).toBe("high");

    act(() => root.unmount());
    container.remove();
    await renderPage(<ProfilesPage />);
    await settle();
    expect(container.textContent).toContain("Reasoning effort: high");
  });

  it("keeps an unknown persisted profile value visible but allows replacing it", async () => {
    mockedApi.getProfiles.mockResolvedValue({
      profiles: [
        {
          name: "default",
          path: "/tmp/default",
          is_default: true,
          model: "model-a",
          provider: "provider-a",
          reasoning_effort: "turbo",
          has_env: false,
          skill_count: 0,
          gateway_running: false,
          description: "",
          description_auto: false,
          distribution_name: null,
          distribution_version: null,
          distribution_source: null,
          has_alias: false,
        },
      ],
    });

    await renderPage(<ProfilesPage />);
    expect(container.textContent).toContain("Reasoning effort: turbo");

    const actions = container.querySelector<HTMLButtonElement>("button[aria-label='Actions']");
    await act(async () => actions!.click());
    const changeModel = [...container.querySelectorAll<HTMLButtonElement>("[role=menuitem]")].find(
      (button) => button.textContent?.includes("Change model"),
    );
    await act(async () => changeModel!.click());
    await settle();

    const selects = [...container.querySelectorAll<HTMLButtonElement>("button[role=combobox]")];
    expect(selects[1].textContent).toContain("turbo");
    expect(openOptions(selects[1]).map((option) => option.textContent?.trim())).toEqual([
      "Inherit provider default",
      ...EFFORT_OPTIONS.map(({ label }) => label),
    ]);

    await chooseOption(selects[1], "Medium");
    const save = [...container.querySelectorAll<HTMLButtonElement>("button")].find(
      (button) => button.textContent?.trim() === "Save",
    );
    expect(save).toBeDefined();
    await act(async () => {
      save!.click();
      await Promise.resolve();
    });
    expect(mockedApi.setProfileSettings).toHaveBeenCalledWith("default", null, null, "medium");
  });

  it("does not persist an unsaved profile selection after leaving and returning", async () => {
    await renderPage(<ProfilesPage />);

    const actions = container.querySelector<HTMLButtonElement>("button[aria-label='Actions']");
    await act(async () => actions!.click());
    const changeModel = [...container.querySelectorAll<HTMLButtonElement>("[role=menuitem]")].find(
      (button) => button.textContent?.includes("Change model"),
    );
    await act(async () => changeModel!.click());
    await settle();

    let selects = [...container.querySelectorAll<HTMLButtonElement>("button[role=combobox]")];
    await chooseOption(selects[1], "High");
    expect(selects[1].textContent).toContain("High");
    expect(mockedApi.setProfileSettings).not.toHaveBeenCalled();

    act(() => root.unmount());
    container.remove();
    await renderPage(<ProfilesPage />);
    await settle();

    expect(container.textContent).toContain("Reasoning effort: provider default");
    expect(mockedApi.setProfileSettings).not.toHaveBeenCalled();
    selects = [...container.querySelectorAll<HTMLButtonElement>("button[role=combobox]")];
    expect(selects).toHaveLength(0);
  });
});
