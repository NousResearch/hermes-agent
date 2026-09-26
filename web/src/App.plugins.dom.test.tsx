// @vitest-environment jsdom
import { act, type ReactNode } from "react";
import { createRoot, type Root } from "react-dom/client";
import { MemoryRouter } from "react-router";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { api } from "@/lib/api";
import { cacheManifests } from "@/plugins/usePlugins";
import type { PluginManifest } from "@/plugins/types";
import App from "./App";

vi.mock("@nous-research/ui/hooks/use-below-breakpoint", () => ({
  useBelowBreakpoint: () => false,
}));
vi.mock("@nous-research/ui/ui/components/selection-switcher", () => ({
  SelectionSwitcher: () => null,
}));
vi.mock("@nous-research/ui/ui/components/spinner", () => ({ Spinner: () => null }));
vi.mock("@/components/AuthWidget", () => ({ AuthWidget: () => null }));
vi.mock("@/components/LanguageSwitcher", () => ({ LanguageSwitcher: () => null }));
vi.mock("@/components/MemoryPressureBanner", () => ({ MemoryPressureBanner: () => null }));
vi.mock("@/components/ProfileScopeBanner", () => ({ ProfileScopeBanner: () => null }));
vi.mock("@/components/ProfileSwitcher", () => ({ ProfileSwitcher: () => null }));
vi.mock("@/components/SidebarFooter", () => ({ SidebarFooter: () => null }));
vi.mock("@/components/SidebarStatusStrip", () => ({
  SidebarStatusStrip: () => null,
  gatewayLine: () => ({ label: "", tone: "text-muted-foreground" }),
}));
vi.mock("@/components/ThemeSwitcher", () => ({ ThemeSwitcher: () => null }));
vi.mock("@/contexts/ProfileProvider", () => ({
  ProfileProvider: ({ children }: { children: ReactNode }) => children,
}));
vi.mock("@/contexts/useProfileScope", () => ({
  useProfileScope: () => ({ profile: "" }),
}));
vi.mock("@/contexts/useSystemActions", () => ({
  useSystemActions: () => ({
    activeAction: null,
    isBusy: false,
    isRunning: false,
    pendingAction: null,
    runAction: async () => {},
  }),
}));
vi.mock("@/hooks/useSidebarStatus", () => ({ useSidebarStatus: () => null }));
vi.mock("@/pages/ChatPage", () => ({
  default: () => "Built-in chat fallback",
}));
vi.mock("@/themes", () => ({ useTheme: () => ({ theme: {} }) }));

const cachedChatOverride: PluginManifest = {
  name: "cached-chat",
  label: "Cached chat",
  description: "",
  icon: "Puzzle",
  version: "1",
  tab: { path: "/cached-chat", override: "/chat" },
  entry: "index.js",
  has_api: false,
  source: "local",
};

function deferred<T>() {
  let reject!: (error: Error) => void;
  const promise = new Promise<T>((_resolve, no) => { reject = no; });
  return { promise, reject };
}

let root: Root | undefined;
let container: HTMLDivElement | undefined;

beforeEach(() => {
  vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
  window.matchMedia = vi.fn().mockReturnValue({
    matches: false,
    media: "",
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
  });
  sessionStorage.clear();
  localStorage.clear();
});

afterEach(async () => {
  if (root) await act(async () => root?.unmount());
  container?.remove();
  root = undefined;
  container = undefined;
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe("plugin route authority", () => {
  it("settles cached /chat override to the built-in chat after current manifest fetch failure", async () => {
    cacheManifests([cachedChatOverride]);
    const request = deferred<PluginManifest[]>();
    vi.spyOn(api, "getPlugins").mockReturnValue(request.promise);
    vi.spyOn(api, "getConfig").mockResolvedValue({ dashboard: {} });

    container = document.createElement("div");
    document.body.append(container);
    root = createRoot(container);
    await act(async () => {
      root?.render(
        <MemoryRouter initialEntries={["/chat"]}>
          <App />
        </MemoryRouter>,
      );
    });

    expect(container.textContent).toContain("Loading");
    expect(container.querySelector('[data-chat-active="true"]')).toBeNull();
    expect(document.querySelector("script[data-hermes-plugin]")).toBeNull();
    expect(document.querySelector('link[href*="/dashboard-plugins/"]')).toBeNull();

    await act(async () => request.reject(new Error("offline")));

    await vi.waitFor(() => {
      expect(container?.querySelector('[data-chat-active="true"]')).not.toBeNull();
      expect(container?.textContent).toContain("Built-in chat fallback");
    });
    expect(container.textContent).not.toContain("Loading");
    expect(document.querySelector("script[data-hermes-plugin]")).toBeNull();
    expect(document.querySelector('link[href*="/dashboard-plugins/"]')).toBeNull();
  });
});
