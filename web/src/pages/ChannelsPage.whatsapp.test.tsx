// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, expect, it, vi } from "vitest";

import type { MessagingPlatform } from "@/lib/api";
import { WhatsAppOnboardingPanel } from "./ChannelsPage";

const start = vi.hoisted(() => vi.fn());
vi.mock("@/lib/api", () => ({ api: { startWhatsAppOnboarding: start } }));

let root: Root;
let container: HTMLDivElement;
(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const platform = {
  id: "whatsapp",
  configured: true,
  whatsapp_setup: { mode: "bot", allowed_users_set: true },
} as MessagingPlatform;

function button(label: string) {
  const found = Array.from(document.querySelectorAll("button")).find(
    (element) => element.textContent?.trim() === label,
  );
  if (!found) throw new Error(`Missing button: ${label}`);
  return found;
}

async function click(label: string) {
  await act(async () => { button(label).click(); });
}

beforeEach(async () => {
  start.mockReset();
  start.mockResolvedValue({
    pairing_id: "fresh", status: "starting", mode: "bot", allowed_users: "",
    expires_at: new Date(Date.now() + 600_000).toISOString(),
  });
  container = document.createElement("div");
  document.body.append(container);
  root = createRoot(container);
  await act(async () => {
    root.render(
      <WhatsAppOnboardingPanel
        platform={platform}
        onChanged={vi.fn(async () => undefined)}
        onRestartNeeded={vi.fn()}
        setRestartNeeded={vi.fn()}
        showToast={vi.fn()}
      />,
    );
  });
});

afterEach(async () => {
  await act(async () => { root.unmount(); });
  container.remove();
});

it("requires explicit confirmation before replacing the saved link", async () => {
  await click("Re-pair with QR");
  expect(start).not.toHaveBeenCalled();
  expect(document.querySelector('[role="dialog"]')?.textContent).toContain("private backup");
  await click("Cancel");
  expect(start).not.toHaveBeenCalled();
  await click("Re-pair with QR");
  await click("Back up and re-pair");
  expect(start).toHaveBeenCalledWith({ mode: "bot", allowed_users: "", reset_session: true });
  expect(button("Re-pair with QR").disabled).toBe(true);
});

it("keeps ordinary pairing non-destructive", async () => {
  await click("Pair with QR");
  expect(start).toHaveBeenCalledWith({ mode: "bot", allowed_users: "", reset_session: false });
});

it("shows an active-bridge refusal and allows retry", async () => {
  start.mockRejectedValueOnce(new Error("The WhatsApp bridge is still running. Stop it before re-pairing."));
  await click("Re-pair with QR");
  await click("Back up and re-pair");
  expect(container.textContent).toContain("Stop it before re-pairing");
  expect(button("Re-pair with QR").disabled).toBe(false);
});
