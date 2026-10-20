// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { MessagingPlatform } from "@/lib/api";

const apiMocks = vi.hoisted(() => ({
  updateMessagingPlatform: vi.fn(),
  startWhatsAppOnboarding: vi.fn(),
  getWhatsAppOnboardingStatus: vi.fn(),
  applyWhatsAppOnboarding: vi.fn(),
  cancelWhatsAppOnboarding: vi.fn(),
  getActionStatus: vi.fn(),
}));

vi.mock("@/lib/api", () => ({ api: apiMocks }));

let container: HTMLDivElement;
let root: Root;
(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const showToast = vi.fn();
const onChanged = vi.fn(async () => undefined);
const onRestartNeeded = vi.fn();
const setRestartNeeded = vi.fn();

function makePlatform(overrides: Partial<MessagingPlatform> = {}): MessagingPlatform {
  return {
    id: "whatsapp",
    name: "WhatsApp",
    description: "",
    docs_url: null,
    enabled: true,
    configured: true,
    gateway_running: true,
    state: "running",
    error_code: null,
    error_message: null,
    updated_at: null,
    home_channel: null,
    env_vars: [],
    ingress_url: null,
    whatsapp_setup: { mode: "bot", allowed_users_set: true, home_channel_set: false },
    ...overrides,
  } as MessagingPlatform;
}

async function renderPanel(platform: MessagingPlatform) {
  const { WhatsAppOnboardingPanel } = await import("./ChannelsPage");
  container = document.createElement("div");
  document.body.append(container);
  root = createRoot(container);
  await act(async () =>
    root.render(
      <WhatsAppOnboardingPanel
        onChanged={onChanged}
        onRestartNeeded={onRestartNeeded}
        platform={platform}
        setRestartNeeded={setRestartNeeded}
        showToast={showToast}
      />,
    ),
  );
}

function click(el: Element | null) {
  if (!el) throw new Error("element not rendered");
  el.dispatchEvent(new MouseEvent("click", { bubbles: true, cancelable: true }));
}

const saveButton = () =>
  Array.from(document.querySelectorAll("button")).find((b) =>
    b.textContent?.includes("Save allowlist"),
  ) ?? null;

async function typeAllowedUsers(value: string) {
  const input = document.querySelector<HTMLInputElement>("#whatsapp-allowed-users");
  if (!input) throw new Error("allowed-users input not rendered");
  await act(async () => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value")!.set!.call(input, value);
    input.dispatchEvent(new Event("input", { bubbles: true }));
  });
}

beforeEach(() => {
  for (const fn of Object.values(apiMocks)) fn.mockReset();
  apiMocks.updateMessagingPlatform.mockResolvedValue({ ok: true, platform: "whatsapp", hot_served: true });
  vi.clearAllMocks();
  vi.stubGlobal("fetch", vi.fn(async () => ({ ok: false, status: 500 })));
  vi.stubGlobal("ResizeObserver", class { disconnect() {} observe() {} unobserve() {} });
});

afterEach(async () => {
  await act(async () => root?.unmount());
  container?.remove();
  vi.unstubAllGlobals();
});

describe("WhatsAppOnboardingPanel idle allowlist save (#109776)", () => {
  it("persists a non-empty allowlist through the messaging config endpoint without an onboarding session", async () => {
    await renderPanel(makePlatform());
    expect(saveButton()).not.toBeNull();

    await typeAllowedUsers("15551234567,15557654321");
    await act(async () => click(saveButton()));

    expect(apiMocks.updateMessagingPlatform).toHaveBeenCalledWith("whatsapp", {
      env: { WHATSAPP_ALLOWED_USERS: "15551234567,15557654321" },
    });
    expect(showToast).toHaveBeenCalledWith(
      expect.stringContaining("allowlist saved"),
      "success",
    );
    expect(onChanged).toHaveBeenCalled();
  });

  it("saving an empty field explicitly clears the saved allowlist and asks for a restart", async () => {
    apiMocks.updateMessagingPlatform.mockResolvedValue({ ok: true, platform: "whatsapp", hot_served: false });
    await renderPanel(makePlatform());

    await typeAllowedUsers("   ");
    await act(async () => click(saveButton()));

    expect(apiMocks.updateMessagingPlatform).toHaveBeenCalledWith("whatsapp", {
      clear_env: ["WHATSAPP_ALLOWED_USERS"],
    });
    expect(onRestartNeeded).toHaveBeenCalled();
    expect(showToast).toHaveBeenCalledWith(expect.stringContaining("restart"), "success");
  });

  it("keeps the idle save action hidden while the channel is not configured yet", async () => {
    await renderPanel(makePlatform({ configured: false }));
    expect(saveButton()).toBeNull();
    expect(apiMocks.updateMessagingPlatform).not.toHaveBeenCalled();
  });

  it("surfaces a backend rejection as an error toast without touching the restart state", async () => {
    apiMocks.updateMessagingPlatform.mockRejectedValue(new Error("400: WHATSAPP_ALLOWED_USERS is not configurable"));
    await renderPanel(makePlatform());

    await typeAllowedUsers("15551234567");
    await act(async () => click(saveButton()));

    expect(showToast).toHaveBeenCalledWith(
      expect.stringContaining("Failed to save allowlist"),
      "error",
    );
    expect(onRestartNeeded).not.toHaveBeenCalled();
    expect(onChanged).not.toHaveBeenCalled();
  });
});
