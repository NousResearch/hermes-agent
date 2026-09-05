// @vitest-environment jsdom
import { act, type ReactNode } from "react";
import { createRoot, type Root } from "react-dom/client";
import { MemoryRouter } from "react-router";
import { afterEach, describe, expect, it, vi } from "vitest";

const apiMocks = vi.hoisted(() => ({
  getSkills: vi.fn(),
  getToolsets: vi.fn(),
}));

vi.mock("@/lib/api", () => ({
  api: apiMocks,
}));

vi.mock("@/contexts/useProfileScope", () => ({
  useProfileScope: () => ({ profile: "" }),
}));

vi.mock("@/contexts/usePageHeader", () => ({
  usePageHeader: () => ({ setAfterTitle: vi.fn(), setEnd: vi.fn() }),
}));

vi.mock("@/components/ToolsetConfigDrawer", () => ({
  ToolsetConfigDrawer: () => null,
}));

vi.mock("@/components/SkillEditorDialog", () => ({
  SkillEditorDialog: () => null,
}));

vi.mock("@/plugins", () => ({
  PluginSlot: () => null,
}));

vi.mock("@nous-research/ui/hooks/use-toast", () => ({
  useToast: () => ({ toast: null, showToast: vi.fn() }),
}));

vi.mock("@/i18n", () => {
  const labels = new Proxy(
    {},
    { get: (_target, key: string | symbol) => String(key) },
  );
  return {
    useI18n: () => ({ t: { common: labels, skills: labels } }),
  };
});

let container: HTMLDivElement;
let root: Root;

async function render(ui: ReactNode) {
  container = document.createElement("div");
  document.body.append(container);
  root = createRoot(container);
  await act(async () => {
    root.render(ui);
  });
}

afterEach(() => {
  act(() => root?.unmount());
  container?.remove();
  vi.clearAllMocks();
});

describe("SkillsPage accessibility", () => {
  it("names an enabled skill switch with the skill name", async () => {
    apiMocks.getSkills.mockResolvedValue([
      {
        name: "airtable",
        description: "Manage Airtable data",
        enabled: true,
        category: null,
      },
    ]);
    apiMocks.getToolsets.mockResolvedValue([]);

    const { default: SkillsPage } = await import("./SkillsPage");
    await render(
      <MemoryRouter>
        <SkillsPage />
      </MemoryRouter>,
    );

    const skillSwitch = container.querySelector('[role="switch"]');
    expect(skillSwitch).not.toBeNull();
    expect(skillSwitch?.getAttribute("aria-label")).toBe("Disable airtable");
    expect(skillSwitch?.getAttribute("aria-checked")).toBe("true");
  });
});
