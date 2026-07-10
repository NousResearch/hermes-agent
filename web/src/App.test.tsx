import { act, type ReactNode, useEffect } from "react";
import { createRoot, type Root } from "react-dom/client";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const chatPageMock = vi.hoisted(() => {
  let resolve!: (module: { default: (props: { isActive?: boolean }) => ReactNode }) => void;
  const promise = new Promise<{ default: (props: { isActive?: boolean }) => ReactNode }>((done) => {
    resolve = done;
  });

  return {
    mountCount: 0,
    promise,
    resolve,
    unmountCount: 0,
  };
});

vi.mock("@/pages/ChatPage", () => chatPageMock.promise);

vi.mock("@/pages/SessionsPage", () => ({
  default: () => <div data-testid="sessions-page">Sessions page</div>,
}));

vi.mock("@/pages/FilesPage", () => ({ default: () => <div>FilesPage</div> }));
vi.mock("@/pages/AnalyticsPage", () => ({ default: () => <div>AnalyticsPage</div> }));
vi.mock("@/pages/ModelsPage", () => ({ default: () => <div>ModelsPage</div> }));
vi.mock("@/pages/LogsPage", () => ({ default: () => <div>LogsPage</div> }));
vi.mock("@/pages/CronPage", () => ({ default: () => <div>CronPage</div> }));
vi.mock("@/pages/SkillsPage", () => ({ default: () => <div>SkillsPage</div> }));
vi.mock("@/pages/PluginsPage", () => ({ default: () => <div>PluginsPage</div> }));
vi.mock("@/pages/McpPage", () => ({ default: () => <div>McpPage</div> }));
vi.mock("@/pages/PairingPage", () => ({ default: () => <div>PairingPage</div> }));
vi.mock("@/pages/ChannelsPage", () => ({ default: () => <div>ChannelsPage</div> }));
vi.mock("@/pages/WebhooksPage", () => ({ default: () => <div>WebhooksPage</div> }));
vi.mock("@/pages/SystemPage", () => ({ default: () => <div>SystemPage</div> }));
vi.mock("@/pages/ProfilesPage", () => ({ default: () => <div>ProfilesPage</div> }));
vi.mock("@/pages/ProfileBuilderPage", () => ({ default: () => <div>ProfileBuilderPage</div> }));
vi.mock("@/pages/ConfigPage", () => ({ default: () => <div>ConfigPage</div> }));
vi.mock("@/pages/EnvPage", () => ({ default: () => <div>EnvPage</div> }));
vi.mock("@/pages/DocsPage", () => ({ default: () => <div>DocsPage</div> }));

vi.mock("@nous-research/ui/ui/components/button", () => ({
  Button: ({ children, ...props }: { children?: ReactNode }) => <button {...props}>{children}</button>,
}));
vi.mock("@nous-research/ui/ui/components/selection-switcher", () => ({
  SelectionSwitcher: () => null,
}));
vi.mock("@nous-research/ui/ui/components/spinner", () => ({
  Spinner: () => <span data-testid="spinner" />,
}));
vi.mock("@nous-research/ui/ui/components/typography/index", () => ({
  Typography: ({ children }: { children?: ReactNode }) => <span>{children}</span>,
}));
vi.mock("@nous-research/ui/ui/components/confirm-dialog", () => ({
  ConfirmDialog: () => null,
}));
vi.mock("@nous-research/ui/hooks/use-below-breakpoint", () => ({
  useBelowBreakpoint: () => false,
}));

vi.mock("@/components/AuthWidget", () => ({ AuthWidget: () => null }));
vi.mock("@/components/LanguageSwitcher", () => ({ LanguageSwitcher: () => null }));
vi.mock("@/components/ProfileScopeBanner", () => ({ ProfileScopeBanner: () => null }));
vi.mock("@/components/ProfileSwitcher", () => ({ ProfileSwitcher: () => null }));
vi.mock("@/components/SidebarFooter", () => ({ SidebarFooter: () => null }));
vi.mock("@/components/SidebarStatusStrip", () => ({
  SidebarStatusStrip: () => null,
  gatewayLine: () => ({ label: "Running", tone: "ok" }),
}));
vi.mock("@/components/ThemeSwitcher", () => ({ ThemeSwitcher: () => null }));
vi.mock("@/contexts/PageHeaderProvider", () => ({
  PageHeaderProvider: ({ children }: { children: ReactNode }) => <>{children}</>,
}));
vi.mock("@/contexts/ProfileProvider", () => ({
  ProfileProvider: ({ children }: { children: ReactNode }) => <>{children}</>,
}));
vi.mock("@/contexts/useProfileScope", () => ({
  useProfileScope: () => ({
    currentProfile: "default",
    profile: "",
    profiles: ["default"],
    setProfile: vi.fn(),
  }),
}));
vi.mock("@/contexts/useSystemActions", () => ({
  useSystemActions: () => ({
    activeAction: null,
    isBusy: false,
    isRunning: false,
    pendingAction: null,
    runAction: vi.fn(),
  }),
}));
vi.mock("@/hooks/useSidebarStatus", () => ({
  useSidebarStatus: () => null,
}));
vi.mock("@/i18n", () => ({
  useI18n: () => ({
    t: {
      app: {
        activeSessionsLabel: "Active sessions",
        brand: "Hermes",
        closeNavigation: "Close navigation",
        gatewayStrip: {
          failed: "Failed",
          off: "Off",
          running: "Running",
          starting: "Starting",
          stopped: "Stopped",
        },
        gatewayStatusLabel: "Gateway",
        nav: {
          analytics: "Analytics",
          chat: "Chat",
          channels: "Channels",
          config: "Config",
          cron: "Cron",
          docs: "Docs",
          env: "Env",
          files: "Files",
          logs: "Logs",
          mcp: "MCP",
          models: "Models",
          pairing: "Pairing",
          plugins: "Plugins",
          profiles: "Profiles",
          sessions: "Sessions",
          skills: "Skills",
          system: "System",
          webhooks: "Webhooks",
        },
        navigation: "Navigation",
        openNavigation: "Open navigation",
        pluginNavSection: "Plugins",
        statusOverview: "Status",
        system: "System",
      },
      common: {
        cancel: "Cancel",
        collapse: "Collapse",
        expand: "Expand",
        loading: "Loading",
      },
      language: {
        switchTo: "Switch language",
      },
      status: {
        restartGateway: "Restart gateway",
        restartingGateway: "Restarting gateway",
        updateHermes: "Update Hermes",
        updatingHermes: "Updating Hermes",
      },
      theme: {
        switchTheme: "Switch theme",
      },
    },
  }),
}));
vi.mock("@/lib/api", () => ({
  HERMES_BASE_PATH: "",
  api: {
    checkHermesUpdate: vi.fn(),
    getConfig: vi.fn().mockResolvedValue({ dashboard: {} }),
  },
}));
vi.mock("@/plugins", () => ({
  PluginPage: () => <div data-testid="plugin-page" />,
  PluginSlot: () => null,
  usePlugins: () => ({ loading: false, manifests: [], plugins: [] }),
}));
vi.mock("@/themes", () => ({
  useTheme: () => ({ theme: { layoutVariant: "standard" } }),
}));

import App from "./App";

function ChatPage({ isActive = true }: { isActive?: boolean }) {
  useEffect(() => {
    chatPageMock.mountCount += 1;

    return () => {
      chatPageMock.unmountCount += 1;
    };
  }, []);

  return <div data-testid="chat-page" data-active={String(isActive)} />;
}

describe("App persistent chat route", () => {
  let container: HTMLDivElement;
  let root: Root;

  beforeEach(() => {
    chatPageMock.mountCount = 0;
    chatPageMock.unmountCount = 0;
    Object.defineProperty(window, "matchMedia", {
      configurable: true,
      value: vi.fn(() => ({
        addEventListener: vi.fn(),
        matches: false,
        removeEventListener: vi.fn(),
      })),
    });
    container = document.createElement("div");
    document.body.append(container);
    root = createRoot(container);
  });

  afterEach(() => {
    act(() => {
      root.unmount();
    });
    container.remove();
    vi.restoreAllMocks();
  });

  it("keeps the lazy chat host mounted after first /chat navigation", async () => {
    await act(async () => {
      root.render(
        <MemoryRouter initialEntries={["/chat"]}>
          <App />
        </MemoryRouter>,
      );
    });

    expect(container.textContent).toContain("Loading chat");
    expect(container.querySelector('[data-testid="chat-page"]')).toBeNull();

    await act(async () => {
      chatPageMock.resolve({ default: ChatPage });
      await chatPageMock.promise;
    });

    const chatPage = container.querySelector('[data-testid="chat-page"]') as HTMLElement;
    expect(chatPage).not.toBeNull();
    expect(chatPage.dataset.active).toBe("true");
    expect(chatPageMock.mountCount).toBe(1);

    const sessionsLink = container.querySelector('a[href="/sessions"]') as HTMLAnchorElement;
    expect(sessionsLink).not.toBeNull();

    await act(async () => {
      sessionsLink.dispatchEvent(new MouseEvent("click", { bubbles: true, cancelable: true }));
    });

    expect(container.querySelector('[data-testid="sessions-page"]')).not.toBeNull();
    expect(container.querySelector('[data-chat-active="false"] [data-testid="chat-page"]')).not.toBeNull();
    expect(chatPageMock.mountCount).toBe(1);
    expect(chatPageMock.unmountCount).toBe(0);
  });
});
