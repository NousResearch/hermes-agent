// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, expect, it, vi } from "vitest";

import type { CatalogMessagingPlatform, MessagingPlatform } from "@/lib/api";
import ChannelsPage from "./ChannelsPage";

const getMessagingPlatforms = vi.hoisted(() => vi.fn());
const installMessagingCatalogPlatform = vi.hoisted(() => vi.fn());
vi.mock("@/lib/api", () => ({ api: { getMessagingPlatforms, installMessagingCatalogPlatform } }));
vi.mock("@/contexts/usePageHeader", () => ({ usePageHeader: () => ({ setEnd: () => undefined }) }));

let root: Root;
let container: HTMLDivElement;
(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

function row(overrides: Partial<CatalogMessagingPlatform>): CatalogMessagingPlatform {
  return {
    name: "yuanbao", title: "Yuanbao", badge: "official", summary: "Chat with Hermes in Yuanbao.",
    disclosure: "", blocked: "", blocked_reason: "", install_command: "hermes plugins install yuanbao",
    needs_confirm: false, ...overrides,
  };
}

const yuanbaoCard: MessagingPlatform = {
  id: "yuanbao", name: "Yuanbao", description: "Yuanbao bot", docs_url: "", enabled: true,
  configured: false, gateway_running: true, state: "not_configured", error_code: null,
  error_message: null, updated_at: null, home_channel: null,
  env_vars: [{
    key: "YUANBAO_TOKEN", required: true, is_set: false, redacted_value: null, description: "",
    prompt: "Yuanbao token", help: "", url: null, is_password: true, advanced: false,
  }],
};

async function render(available: CatalogMessagingPlatform[]) {
  getMessagingPlatforms.mockResolvedValue({
    env_path: "~/.hermes/.env", gateway_start_command: "hermes gateway start", platforms: [], available,
  });
  await act(async () => { root.render(<ChannelsPage />); });
}

function rowEl(name: string): HTMLElement {
  const found = container.querySelector<HTMLElement>(`[data-catalog-platform="${name}"]`);
  if (!found) throw new Error(`Missing catalog row ${name}`);
  return found;
}

function button(scope: ParentNode, label: string): HTMLButtonElement | undefined {
  return [...scope.querySelectorAll("button")].find((b) => b.textContent?.trim() === label);
}

async function click(el: HTMLElement | undefined) {
  if (!el) throw new Error("Missing button");
  await act(async () => { el.click(); });
}

beforeEach(() => {
  container = document.createElement("div");
  document.body.append(container);
  root = createRoot(container);
  installMessagingCatalogPlatform.mockReset();
});

afterEach(async () => {
  await act(async () => { root.unmount(); });
  container.remove();
});

it("lists an available platform with its badge, summary and disclosure", async () => {
  await render([row({ disclosure: "Messages pass through Tencent servers." })]);

  const el = rowEl("yuanbao");
  expect(container.textContent).toContain("Available to install");
  expect(el.textContent).toContain("Yuanbao");
  expect(el.textContent).toContain("Official");
  expect(el.textContent).toContain("Chat with Hermes in Yuanbao.");
  expect(el.textContent).toContain("Disclosure: Messages pass through Tencent servers.");
});

it("installs an official platform directly and opens the new card's credential form", async () => {
  await render([row({})]);
  installMessagingCatalogPlatform.mockResolvedValue({
    ok: true, plugin_name: "yuanbao", platforms: ["yuanbao"], warnings: [],
  });
  getMessagingPlatforms.mockResolvedValue({
    env_path: "~/.hermes/.env", gateway_start_command: "hermes gateway start",
    platforms: [yuanbaoCard], available: [],
  });

  await click(button(rowEl("yuanbao"), "Install"));

  expect(installMessagingCatalogPlatform).toHaveBeenCalledWith("yuanbao", false);
  expect(document.getElementById("channel-config-title")?.textContent).toBe("Configure Yuanbao");
  expect(document.querySelector<HTMLInputElement>("#field-YUANBAO_TOKEN")?.type).toBe("password");
});

it("keeps a form the user opened while the install ran instead of replacing it", async () => {
  const discordCard: MessagingPlatform = {
    ...yuanbaoCard, id: "discord", name: "Discord", description: "Discord bot",
    env_vars: [{ ...yuanbaoCard.env_vars[0], key: "DISCORD_BOT_TOKEN", prompt: "Discord token" }],
  };
  getMessagingPlatforms.mockResolvedValue({
    env_path: "~/.hermes/.env", gateway_start_command: "hermes gateway start",
    platforms: [discordCard], available: [row({})],
  });
  await act(async () => { root.render(<ChannelsPage />); });
  let finishInstall: (value: unknown) => void = () => undefined;
  installMessagingCatalogPlatform.mockReturnValue(new Promise((resolve) => { finishInstall = resolve; }));

  await click(button(rowEl("yuanbao"), "Install"));
  await click(button(container, "Configure"));  // Discord's form, opened while Yuanbao installs
  expect(document.getElementById("channel-config-title")?.textContent).toBe("Configure Discord");

  getMessagingPlatforms.mockResolvedValue({
    env_path: "~/.hermes/.env", gateway_start_command: "hermes gateway start",
    platforms: [discordCard, yuanbaoCard], available: [],
  });
  await act(async () => { finishInstall({ ok: true, plugin_name: "yuanbao", platforms: ["yuanbao"], warnings: [] }); });

  expect(document.getElementById("channel-config-title")?.textContent).toBe("Configure Discord");
  expect(container.textContent).toContain("Yuanbao");
});

it("asks before installing a Hermes Labs platform and sends the confirmation", async () => {
  await render([row({ name: "xchat", title: "X Chat", badge: "labs", needs_confirm: true })]);
  installMessagingCatalogPlatform.mockResolvedValue({
    ok: false, error: "Clone failed", install_command: "hermes plugins install xchat",
  });

  expect(rowEl("xchat").textContent).toContain("Hermes Labs");
  await click(button(rowEl("xchat"), "Install"));
  expect(installMessagingCatalogPlatform).not.toHaveBeenCalled();
  const dialog = document.querySelector('[role="dialog"]');
  expect(dialog?.textContent).toContain("built by Nous Research, not supported, and likely not maintained");

  await click(button(document, "Install anyway"));
  expect(installMessagingCatalogPlatform).toHaveBeenCalledWith("xchat", true);
  // A refusal stays on the row with the command that installs it by hand.
  expect(rowEl("xchat").textContent).toContain("Clone failed");
  expect(rowEl("xchat").textContent).toContain("Run: hermes plugins install xchat");
});

it("shows the command instead of an Install button when lazy installs are off", async () => {
  await render([row({
    blocked: "lazy_installs_off",
    blocked_reason: "security.allow_lazy_installs is off.",
    install_command: "hermes -p work plugins install yuanbao",
  })]);

  const el = rowEl("yuanbao");
  expect(button(el, "Install")).toBeUndefined();
  expect(el.textContent).toContain("Automatic plugin installs are off (security.allow_lazy_installs).");
  expect(el.textContent).toContain("Run: hermes -p work plugins install yuanbao");
});
