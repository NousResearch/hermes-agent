// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, expect, it, vi } from "vitest";

import type { MessagingPlatform } from "@/lib/api";
import ChannelsPage from "./ChannelsPage";

const getMessagingPlatforms = vi.hoisted(() => vi.fn());
vi.mock("@/lib/api", () => ({ api: { getMessagingPlatforms } }));
vi.mock("@/contexts/usePageHeader", () => ({ usePageHeader: () => ({ setEnd: () => undefined }) }));

let root: Root;
let container: HTMLDivElement;
(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

function slack(overrides: Partial<MessagingPlatform>): MessagingPlatform {
  return {
    id: "slack", name: "Slack", description: "Slack bot", docs_url: "", enabled: false,
    configured: true, gateway_running: true, state: "disabled", error_code: null,
    error_message: null, updated_at: null, home_channel: null, env_vars: [], ...overrides,
  };
}

async function render(platform: MessagingPlatform) {
  getMessagingPlatforms.mockResolvedValue({
    env_path: "~/.hermes/.env", gateway_start_command: "hermes gateway start", platforms: [platform],
  });
  await act(async () => { root.render(<ChannelsPage />); });
}

function toggle() {
  const found = container.querySelector<HTMLButtonElement>('[aria-label="Enable Slack"]');
  if (!found) throw new Error("Missing Slack toggle");
  return found;
}

beforeEach(() => {
  container = document.createElement("div");
  document.body.append(container);
  root = createRoot(container);
});

afterEach(async () => {
  await act(async () => { root.unmount(); });
  container.remove();
});

it("names the Team Gateway rule instead of a bare Disabled and locks the toggle", async () => {
  await render(slack({ disabled_by_relay: true }));

  expect(container.textContent).toContain("Managed by Team Gateway");
  expect(container.textContent).not.toContain("Disabled");
  expect(container.textContent).toContain("GATEWAY_RELAY_ALLOW_DIRECT_PLATFORMS=true");
  expect(toggle().disabled).toBe(true);
  const note = document.getElementById(toggle().getAttribute("aria-describedby") ?? "");
  expect(note?.textContent).toContain("Team Gateway");
});

it("keeps the plain Disabled state and a live toggle when the user turned it off", async () => {
  await render(slack({ disabled_by_relay: false }));

  expect(container.textContent).toContain("Disabled");
  expect(container.textContent).not.toContain("Team Gateway");
  expect(toggle().disabled).toBe(false);
});
