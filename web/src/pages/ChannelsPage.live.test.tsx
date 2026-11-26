// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, expect, it, vi } from "vitest";

import type { MessagingPlatform, MessagingPlatformsResponse } from "@/lib/api";
import ChannelsPage from "./ChannelsPage";

const getMessagingPlatforms = vi.hoisted(() => vi.fn());
const getMessagingPlatformsSignature = vi.hoisted(() => vi.fn());
vi.mock("@/lib/api", () => ({ api: { getMessagingPlatforms, getMessagingPlatformsSignature } }));
vi.mock("@/contexts/usePageHeader", () => ({ usePageHeader: () => ({ setEnd: () => undefined }) }));

let root: Root;
let container: HTMLDivElement;
(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const qqCard: MessagingPlatform = {
  id: "qqbot", name: "QQ Bot", description: "QQ Bot", docs_url: "", enabled: false,
  configured: false, gateway_running: true, state: "not_configured", error_code: null,
  error_message: null, updated_at: null, home_channel: null, env_vars: [],
};

function listing(signature: string, platforms: MessagingPlatform[]): MessagingPlatformsResponse {
  return { signature, env_path: "~/.hermes/.env", gateway_start_command: "hermes gateway start", platforms, available: [] };
}

beforeEach(() => {
  vi.useFakeTimers();
  container = document.createElement("div");
  document.body.append(container);
  root = createRoot(container);
});

afterEach(async () => {
  await act(async () => { root.unmount(); });
  container.remove();
  vi.useRealTimers();
});

it("shows a platform plugin installed in the background without a reload", async () => {
  getMessagingPlatforms.mockResolvedValue(listing("before", []));
  getMessagingPlatformsSignature.mockResolvedValue({ signature: "before" });
  await act(async () => { root.render(<ChannelsPage />); });
  expect(container.textContent).not.toContain("QQ Bot");

  // Nothing moved: the poll does not rebuild the list.
  await act(async () => { await vi.advanceTimersByTimeAsync(3_000); });
  expect(getMessagingPlatformsSignature).toHaveBeenCalled();
  expect(getMessagingPlatforms).toHaveBeenCalledTimes(1);

  // The background default install lands the plugin: the list's signature moves.
  getMessagingPlatforms.mockResolvedValue(listing("after", [qqCard]));
  getMessagingPlatformsSignature.mockResolvedValue({ signature: "after" });
  await act(async () => { await vi.advanceTimersByTimeAsync(3_000); });

  expect(getMessagingPlatforms).toHaveBeenCalledTimes(2);
  expect(container.textContent).toContain("QQ Bot");
});
