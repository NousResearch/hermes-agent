// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ProfileContext } from "@/contexts/profile-context";
import { I18nProvider } from "@/i18n/context";
import { WhatsAppProfileRoutesPanel } from "./WhatsAppProfileRoutesPanel";

const scene = vi.hoisted(() => ({
  multiplex: false as boolean,
}));

const api = vi.hoisted(() => ({
  getProfiles: vi.fn(),
  getConfig: vi.fn(),
  getWhatsAppSeenChats: vi.fn(),
  getStatus: vi.fn(),
  saveConfig: vi.fn(),
}));

vi.mock("@/lib/api", () => ({ api }));

const route = {
  platform: "whatsapp",
  chat_id: "15551234567@s.whatsapp.net",
  profile: "ops",
  name: "person",
};

function installApi() {
  api.getProfiles.mockResolvedValue({
    profiles: [
      { name: "default", is_default: true },
      { name: "ops", is_default: false },
    ],
  });
  api.getConfig.mockImplementation(async () => ({
    gateway: {
      multiplex_profiles: scene.multiplex,
      profile_routes: [route],
    },
  }));
  api.getWhatsAppSeenChats.mockResolvedValue({
    chats: [{ chat_id: route.chat_id, name: "unknown number", kind: "dm" }],
  });
  api.getStatus.mockResolvedValue({
    gateway_shared_with: null,
    multiplex_standalone_reason: null,
  });
  api.saveConfig.mockResolvedValue({});
}

async function renderPanel(profile: string) {
  const container = document.createElement("div");
  document.body.appendChild(container);
  const root = createRoot(container);
  const showToast = vi.fn();
  await act(async () => {
    root.render(
      <I18nProvider>
        <ProfileContext.Provider
          value={{
            profile,
            currentProfile: "default",
            profiles: ["default", "ops"],
            setProfile: () => {},
          }}
        >
          <WhatsAppProfileRoutesPanel showToast={showToast} />
        </ProfileContext.Provider>
      </I18nProvider>,
    );
  });
  for (let i = 0; i < 20 && !container.textContent?.includes("15551234567"); i += 1) {
    await act(async () => {
      await new Promise((resolve) => setTimeout(resolve, 0));
    });
  }
  return { container, root, showToast };
}

describe("WhatsApp multiplexer frame", () => {
  let mounted: { root: Root; container: HTMLDivElement } | null = null;

  afterEach(() => {
    if (mounted) {
      act(() => mounted?.root.unmount());
      mounted.container.remove();
      mounted = null;
    }
  });

  it("shows Available while off, On when engaged, and the phone number instead of a placeholder", async () => {
    scene.multiplex = false;
    installApi();
    const off = await renderPanel("default");
    mounted = off;
    expect(off.container.textContent).toContain("Available");
    expect(off.container.textContent).toContain("Linked to");
    expect(off.container.textContent).toContain("default");
    expect(off.container.textContent).toContain("15551234567");
    expect(off.container.textContent?.toLowerCase()).not.toContain("unknown number");
    const offSwitch = off.container.querySelector("[role='switch']");
    expect(offSwitch?.getAttribute("aria-checked")).toBe("false");

    scene.multiplex = true;
    installApi();
    act(() => off.root.unmount());
    off.container.remove();
    mounted = null;
    const on = await renderPanel("default");
    mounted = on;
    expect(on.container.textContent).toContain("On");
    expect(on.container.querySelector("[role='switch']")?.getAttribute("aria-checked")).toBe("true");
  });

  it("holds WhatsApp's destination profile on the managing profile", async () => {
    scene.multiplex = true;
    installApi();
    const view = await renderPanel("ops");
    mounted = view;
    const link = [...view.container.querySelectorAll("button")].find((button) =>
      button.textContent?.includes("default"),
    );
    expect(link).toBeTruthy();
    expect(view.container.textContent).toContain("own WhatsApp stays off");
    const toggle = view.container.querySelector("[role='switch']");
    expect(toggle?.hasAttribute("disabled")).toBe(true);
    expect(toggle?.getAttribute("aria-checked")).toBe("true");
  });
});
