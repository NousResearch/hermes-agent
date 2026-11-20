// @vitest-environment jsdom
import { act, type ReactNode } from "react";
import { createRoot, type Root } from "react-dom/client";
import { MemoryRouter, Route, Routes, useLocation } from "react-router";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const apiMocks = vi.hoisted(() => ({
  getSkills: vi.fn(async () => []),
  getToolsets: vi.fn(async () => []),
  getSkillsHubInstalled: vi.fn(async () => ({ skills: [] })),
  getSkillsHubTaps: vi.fn(async () => ({ taps: [] })),
}));

const profileScopeState = vi.hoisted(() => ({
  selectedProfile: "",
}));

vi.mock("@/lib/api", () => ({
  api: apiMocks,
}));
vi.mock("@/contexts/useProfileScope", () => ({
  useProfileScope: () => ({
    profile: profileScopeState.selectedProfile,
    currentProfile: "default",
    profiles: [],
    setProfile: vi.fn(),
  }),
}));
vi.mock("@/contexts/usePageHeader", () => ({
  usePageHeader: () => ({
    setAfterTitle: vi.fn(),
    setEnd: vi.fn(),
    setTitle: vi.fn(),
  }),
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
vi.mock("@nous-research/ui/ui/components/dialog", () => ({
  Dialog: ({ open, children }: { open?: boolean; children?: ReactNode }) =>
    open ? <div data-testid="learn-dialog">{children}</div> : null,
  DialogContent: ({ children }: { children?: ReactNode }) => <div>{children}</div>,
  DialogDescription: ({ children }: { children?: ReactNode }) => <p>{children}</p>,
  DialogHeader: ({ children }: { children?: ReactNode }) => <div>{children}</div>,
  DialogTitle: ({ children }: { children?: ReactNode }) => <h2>{children}</h2>,
}));

import { I18nProvider } from "@/i18n";
import SkillsPage from "./SkillsPage";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT =
  true;

let container: HTMLDivElement;
let root: Root;

beforeEach(() => {
  vi.clearAllMocks();
  profileScopeState.selectedProfile = "";
  container = document.createElement("div");
  document.body.append(container);
  root = createRoot(container);
});

afterEach(async () => {
  await act(async () => root.unmount());
  container.remove();
});

function setNativeValue(el: HTMLInputElement | HTMLTextAreaElement, value: string) {
  const proto =
    el instanceof HTMLTextAreaElement
      ? HTMLTextAreaElement.prototype
      : HTMLInputElement.prototype;
  const setter = Object.getOwnPropertyDescriptor(proto, "value")?.set;
  setter?.call(el, value);
  el.dispatchEvent(new Event("input", { bubbles: true }));
}

describe("SkillsPage Learn a skill handoff", () => {
  it("navigates to /chat with normalized router state and no ?learn= query parameter", async () => {
    let chatLocation: { pathname: string; search: string; state: unknown } | null = null;
    function ChatSpy() {
      const loc = useLocation();
      chatLocation = { pathname: loc.pathname, search: loc.search, state: loc.state };
      return <div data-testid="chat-spy" />;
    }

    await act(async () => {
      root.render(
        <I18nProvider>
          <MemoryRouter initialEntries={["/skills"]}>
            <Routes>
              <Route path="/skills" element={<SkillsPage />} />
              <Route path="/chat" element={<ChatSpy />} />
            </Routes>
          </MemoryRouter>
        </I18nProvider>,
      );
    });

    const openBtn = Array.from(container.querySelectorAll("button")).find((btn) =>
      btn.textContent?.includes("Learn a skill"),
    );
    expect(openBtn).toBeTruthy();

    await act(async () => {
      openBtn!.click();
    });

    const textarea = container.querySelector("textarea");
    expect(textarea).toBeTruthy();
    await act(async () => {
      setNativeValue(
        textarea!,
        "workflow \u001b[31mnotes\u001b[0m\nsecond \u0015line",
      );
    });

    const submitBtn = Array.from(container.querySelectorAll("button")).find((btn) =>
      btn.textContent?.includes("Learn it"),
    );
    expect(submitBtn).toBeTruthy();

    await act(async () => {
      submitBtn!.click();
    });

    expect(chatLocation).toEqual({
      pathname: "/chat",
      search: "",
      state: { learnSeed: "workflow notes second line" },
    });
  });

  it("preserves ?profile= on /chat navigation when a non-empty profile is selected", async () => {
    profileScopeState.selectedProfile = "work";
    let chatLocation: { pathname: string; search: string; state: unknown } | null = null;
    function ChatSpy() {
      const loc = useLocation();
      chatLocation = { pathname: loc.pathname, search: loc.search, state: loc.state };
      return <div data-testid="chat-spy" />;
    }

    await act(async () => {
      root.render(
        <I18nProvider>
          <MemoryRouter initialEntries={["/skills?profile=work"]}>
            <Routes>
              <Route path="/skills" element={<SkillsPage />} />
              <Route path="/chat" element={<ChatSpy />} />
            </Routes>
          </MemoryRouter>
        </I18nProvider>,
      );
    });

    const openBtn = Array.from(container.querySelectorAll("button")).find((btn) =>
      btn.textContent?.includes("Learn a skill"),
    );
    await act(async () => openBtn!.click());

    const textarea = container.querySelector("textarea")!;
    await act(async () => setNativeValue(textarea, "profile scoped skill"));

    const submitBtn = Array.from(container.querySelectorAll("button")).find((btn) =>
      btn.textContent?.includes("Learn it"),
    );
    await act(async () => submitBtn!.click());

    expect(chatLocation).toEqual({
      pathname: "/chat",
      search: "?profile=work",
      state: { learnSeed: "profile scoped skill" },
    });
  });
});
