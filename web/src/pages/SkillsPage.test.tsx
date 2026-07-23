/** @vitest-environment jsdom */

import { act, type ReactNode } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { SkillEditorDialog } from "@/components/SkillEditorDialog";
import { api, type SkillHubResult, type SkillInfo } from "@/lib/api";
import {
  HubBrowser,
  HubResultCard,
  MobileCategoryFilters,
  SkillDetailDialog,
  SkillRow,
} from "./SkillsPage";

const SKILL: SkillInfo = {
  name: "a-very-long-mobile-skill-name-that-must-wrap",
  description: "A long skill description that remains readable on a phone.",
  category: "mobile-development",
  enabled: true,
};

const HUB_RESULT: SkillHubResult = {
  name: "Mobile-safe hub skill",
  description: "A hub skill with long metadata for narrow screens.",
  source: "community",
  identifier: "github:example/a-very-long-repository-and-skill-identifier",
  trust_level: "community",
  repo: "example/a-very-long-repository-name",
  tags: ["android", "a-very-long-tag-that-needs-to-wrap"],
};

function buttonWithText(text: string): HTMLButtonElement | undefined {
  return Array.from(document.querySelectorAll<HTMLButtonElement>("button")).find(
    (button) => button.textContent?.includes(text),
  );
}

async function flushEffects() {
  await act(async () => {
    await Promise.resolve();
    await new Promise((resolve) => setTimeout(resolve, 0));
  });
}

function setTextareaValue(textarea: HTMLTextAreaElement, value: string) {
  const setter = Object.getOwnPropertyDescriptor(
    HTMLTextAreaElement.prototype,
    "value",
  )?.set;
  act(() => {
    setter?.call(textarea, value);
    textarea.dispatchEvent(new Event("input", { bubbles: true }));
  });
}

describe("Skills mobile behavior", () => {
  let host: HTMLDivElement;
  let root: Root;

  beforeEach(() => {
    host = document.createElement("div");
    document.body.append(host);
    root = createRoot(host);
  });

  afterEach(() => {
    act(() => root.unmount());
    host.remove();
    vi.restoreAllMocks();
  });

  function render(node: ReactNode) {
    act(() => root.render(node));
  }

  it("renders touch-sized, scroll-snapping category filters with selected state", () => {
    const onChange = vi.fn();
    render(
      <MobileCategoryFilters
        activeCategory="development"
        allLabel="All skills"
        categories={[
          { key: "development", name: "Development", count: 3 },
          { key: "long", name: "A category name that can wrap", count: 1 },
        ]}
        label="Skill categories"
        onChange={onChange}
        total={4}
      />,
    );

    const group = host.querySelector<HTMLElement>("[role='group']");
    const buttons = Array.from(group?.querySelectorAll("button") ?? []);

    expect(group?.getAttribute("aria-labelledby")).toBeTruthy();
    expect(group?.className).toContain("snap-x");
    expect(group?.className).toContain("overflow-x-auto");
    expect(buttons).toHaveLength(3);
    expect(buttons.every((button) => button.className.includes("min-h-11"))).toBe(
      true,
    );
    expect(buttons[0].getAttribute("aria-pressed")).toBe("false");
    expect(buttons[1].getAttribute("aria-pressed")).toBe("true");

    act(() => buttons[1].click());
    expect(onChange).toHaveBeenLastCalledWith(null);
    act(() => buttons[2].click());
    expect(onChange).toHaveBeenLastCalledWith("long");
  });

  it("keeps the edit action visible and touch-sized on phones", () => {
    const onEdit = vi.fn();
    const onToggle = vi.fn();
    render(
      <SkillRow
        skill={SKILL}
        toggling={false}
        onToggle={onToggle}
        onEdit={onEdit}
        noDescriptionLabel="No description"
      />,
    );

    const edit = host.querySelector<HTMLButtonElement>(
      `button[aria-label='Edit ${SKILL.name}']`,
    );
    const toggle = host.querySelector<HTMLButtonElement>("[role='switch']");

    expect(edit?.className).toContain("opacity-100");
    expect(edit?.className).toContain("sm:opacity-0");
    expect(edit?.className).toContain("min-h-11");
    act(() => edit?.click());
    expect(onEdit).toHaveBeenCalledTimes(1);

    act(() => toggle?.click());
    expect(onToggle).toHaveBeenCalledTimes(1);
  });

  it("moves Hub card actions below content on phones without changing actions", () => {
    const onOpen = vi.fn();
    const onInstall = vi.fn();
    render(
      <HubResultCard
        result={HUB_RESULT}
        installed={false}
        onOpen={onOpen}
        onInstall={onInstall}
      />,
    );

    const actions = host.querySelector<HTMLElement>("[data-hub-card-actions]");
    const identifier = Array.from(host.querySelectorAll("p")).find((node) =>
      node.textContent?.includes(HUB_RESULT.identifier),
    );

    expect(actions?.className).toContain("grid-cols-2");
    expect(actions?.className).toContain("sm:flex");
    expect(identifier?.className).toContain("break-all");
    expect(identifier?.className).toContain("lg:truncate");

    act(() => buttonWithText("Details")?.click());
    expect(onOpen).toHaveBeenCalledTimes(1);
    act(() => buttonWithText("Install")?.click());
    expect(onInstall).toHaveBeenCalledTimes(1);
  });

  it("provides a mobile-safe detail dialog with synchronized tab semantics", async () => {
    vi.spyOn(api, "previewSkillFromHub").mockResolvedValue({
      ...HUB_RESULT,
      skill_md: "# Mobile-safe skill",
      files: ["SKILL.md", "references/a-very-long-reference-file.md"],
    });
    vi.spyOn(api, "scanSkillFromHub").mockResolvedValue({
      name: HUB_RESULT.name,
      identifier: HUB_RESULT.identifier,
      source: HUB_RESULT.source,
      trust_level: HUB_RESULT.trust_level,
      verdict: "safe",
      summary: "Safe",
      policy: "allow",
      policy_reason: "No risky patterns detected.",
      findings: [],
      severity_counts: {},
    });
    const onClose = vi.fn();
    render(
      <SkillDetailDialog
        result={HUB_RESULT}
        installed={false}
        onClose={onClose}
        onInstall={() => {}}
        showToast={() => {}}
      />,
    );
    await flushEffects();

    const dialog = document.querySelector<HTMLElement>("[role='dialog']");
    const tabs = Array.from(
      document.querySelectorAll<HTMLButtonElement>("[role='tab']"),
    );
    let panel = document.querySelector<HTMLElement>("[role='tabpanel']");

    expect(dialog?.className).toContain("hermes-modal-panel-viewport");
    expect(dialog?.className).toContain("overflow-hidden");
    expect(tabs).toHaveLength(2);
    expect(tabs[0].getAttribute("aria-selected")).toBe("true");
    expect(tabs[0].getAttribute("aria-controls")).toBe(panel?.id);
    expect(panel?.getAttribute("aria-labelledby")).toBe(tabs[0].id);

    await act(async () => {
      tabs[1].click();
      await Promise.resolve();
    });
    panel = document.querySelector<HTMLElement>("[role='tabpanel']");
    expect(tabs[1].getAttribute("aria-selected")).toBe("true");
    expect(tabs[1].getAttribute("aria-controls")).toBe(panel?.id);
    expect(api.scanSkillFromHub).toHaveBeenCalledWith(HUB_RESULT.identifier);

    act(() => {
      document.dispatchEvent(
        new KeyboardEvent("keydown", { bubbles: true, key: "Escape" }),
      );
    });
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it("keeps editor save behavior while using a keyboard-safe mobile layout", async () => {
    vi.spyOn(api, "getSkillContent").mockResolvedValue({
      name: SKILL.name,
      content: "# Existing skill",
      path: "/skills/example/SKILL.md",
    });
    vi.spyOn(api, "updateSkillContent").mockResolvedValue({ success: true });
    const onSaved = vi.fn();
    render(
      <SkillEditorDialog
        open
        editName={SKILL.name}
        profile="mobile"
        onClose={() => {}}
        onSaved={onSaved}
      />,
    );
    await flushEffects();

    const dialog = document.querySelector<HTMLElement>("[role='dialog']");
    const textarea = document.querySelector<HTMLTextAreaElement>(
      "#skill-editor-content",
    );
    const save = buttonWithText("Save changes");

    expect(dialog?.className).toContain("hermes-modal-panel-viewport");
    expect(textarea?.className).toContain("max-h-[45dvh]");
    expect(save?.className).toContain("min-h-11");

    if (!textarea) throw new Error("Skill editor textarea was not rendered");
    setTextareaValue(textarea, "# Updated skill");
    await act(async () => {
      save?.click();
      await Promise.resolve();
    });

    expect(api.updateSkillContent).toHaveBeenCalledWith(
      SKILL.name,
      "# Updated skill",
      "mobile",
    );
    expect(onSaved).toHaveBeenCalledWith(SKILL.name);
  });

  it("preserves Hub loading, install, and update API flows", async () => {
    vi.spyOn(api, "getSkillHubSources").mockResolvedValue({
      sources: [{ id: "hermes-index", label: "Hermes Index", available: true }],
      index_available: true,
      featured: [HUB_RESULT],
      installed: {},
    });
    vi.spyOn(api, "installSkillFromHub").mockResolvedValue({
      ok: true,
      name: "install-action",
      pid: 101,
    });
    vi.spyOn(api, "updateSkillsFromHub").mockResolvedValue({
      ok: true,
      name: "update-action",
      pid: 102,
    });
    vi.spyOn(api, "getActionStatus").mockResolvedValue({
      exit_code: 0,
      lines: ["done"],
      name: "action",
      pid: null,
      running: false,
    });
    render(<HubBrowser profile="mobile" showToast={() => {}} />);
    await flushEffects();

    expect(api.getSkillHubSources).toHaveBeenCalledWith("mobile");
    const actionLayout = host.querySelector<HTMLElement>(
      "[data-skills-hub-search-actions]",
    );
    expect(actionLayout?.className).toContain("grid-cols-2");

    await act(async () => {
      buttonWithText("Update all")?.click();
      await Promise.resolve();
    });
    expect(api.updateSkillsFromHub).toHaveBeenCalledWith("mobile");

    await act(async () => {
      buttonWithText("Install")?.click();
      await Promise.resolve();
    });
    expect(api.installSkillFromHub).toHaveBeenCalledWith(
      HUB_RESULT.identifier,
      "mobile",
    );
  });
});
